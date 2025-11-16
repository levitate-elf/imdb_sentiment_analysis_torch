import os
import sys
import logging
import datasets
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split

train = pd.read_csv("./labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./testData.tsv", header=0, delimiter="\t", quoting=3)


def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                    F.softmax(target, dtype=torch.float32), reduction=reduction)
    return loss


class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        kl_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        kl_output = kl_outputs[1]
        kl_output = self.dropout(kl_output)
        kl_logits = self.classifier(kl_output)

        loss = None
        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 第一次前向传播的损失
            loss1 = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 第二次前向传播的损失
            loss2 = loss_fct(kl_logits.view(-1, self.num_labels), labels.view(-1))
            # KL散度损失
            kl_loss = (KL(logits, kl_logits, "batchmean") + KL(kl_logits, logits, "batchmean")) / 2.0
            # 总损失 = 两次CE损失的平均 + KL损失
            total_loss = (loss1 + loss2) / 2.0 + kl_loss
        else:
            # 预测阶段，只使用第一次前向传播的结果
            total_loss = None

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,  # 返回第一次前向传播的logits用于预测
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertScratch.from_pretrained('bert-base-uncased')

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        # 新版 transformers: eval_pred 是 EvalPrediction 对象
        if hasattr(eval_pred, "predictions"):
            preds = np.argmax(eval_pred.predictions, axis=-1)
            labels = eval_pred.label_ids
        else:  # 老版是 (logits, labels) 元组
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/bert_RDrop.csv", index=False, quoting=3)
    logging.info('result saved!')