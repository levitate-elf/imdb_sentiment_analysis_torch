import os
os.environ["UNSLOTH_DISABLE_STATS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import unsloth
from unsloth import FastModel, FastLanguageModel

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import (
    TrainingArguments,
    Trainer,
    training_args,
    BertPreTrainedModel,
    BertModel,
)

from transformers.modeling_outputs import SequenceClassifierOutput

# ======================
#  R-Drop 相关模型定义
# ======================

def KL(input, target, reduction="batchmean"):
    """
    对称 KL 会用到的单向 KL:
      KL(input || target)
    """
    input = input.float()
    target = target.float()
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )


class BertWithRDrop(BertPreTrainedModel):
    """
    带 R-Drop 的 BERT 分类模型：
      - 两次前向（同一 batch，不同 dropout mask）
      - CE1 + CE2 的平均 + 对称 KL
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        # 第一次前向
        outputs1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled1 = outputs1.pooler_output
        pooled1 = self.dropout(pooled1)
        logits1 = self.classifier(pooled1)

        # 第二次前向（R-Drop，用同一输入再过一次，mask 不同）
        outputs2 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled2 = outputs2.pooler_output
        pooled2 = self.dropout(pooled2)
        logits2 = self.classifier(pooled2)

        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
            ce2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            # 对称 KL
            kl = (KL(logits1, logits2) + KL(logits2, logits1)) / 2.0
            total_loss = (ce1 + ce2) / 2.0 + kl

        # 预测时我们只用第一份 logits
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits1,
            hidden_states=outputs1.hidden_states,
            attentions=outputs1.attentions,
        )


# ======================
#  主程序：IMDB + BERT + RDrop + Unsloth + LoRA
# ======================

if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(logging.INFO)
    logger.info("running %s", " ".join(sys.argv))

    # -------- 1. 读 IMDB 数据集 --------
    train_df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test_df = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=3407)

    train_dict = {"label": train_df["sentiment"].astype(int), "text": train_df["review"]}
    val_dict = {"label": val_df["sentiment"].astype(int), "text": val_df["review"]}
    test_dict = {"text": test_df["review"]}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # -------- 2. 用 FastModel 构建 BERT+RDrop 模型并挂 LoRA --------
    model_name = "bert-base-uncased"
    NUM_CLASSES = 2

    # 这里把 auto_model 换成我们自定义的 BertWithRDrop
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,          # 需要的话可以调成 True
        max_seq_length=512,
        dtype=None,
        auto_model=BertWithRDrop,
        num_labels=NUM_CLASSES,
        gpu_memory_utilization=0.8,
    )

    # 在 BERT+RDrop 上加 LoRA
    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
        target_modules="all-linear",  # 对所有 Linear 挂 LoRA
        task_type="SEQ_CLS",
    )

    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    # -------- 3. Tokenize --------
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            max_length=512,
            truncation=True,
            padding=True,
        )
        result["prompt"] = examples["text"]
        return result

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 检查数据集字段
    print("训练集字段:", train_dataset.column_names)
    print("测试集字段:", test_dataset.column_names)

    # 设置格式，确保包含所有必要字段
    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")
    test_dataset.set_format(type="torch")


    # -------- 4. 指标函数 --------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # -------- 5. 训练参数 --------
    train_args = TrainingArguments(
        output_dir="./checkpoint_bert_rdrop_unsloth_lora",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=training_args.OptimizerNames.ADAMW_TORCH,
        learning_rate=2e-5,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        num_train_epochs=3,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
    )

    # -------- 6. 用（被 unsloth patch 的）Trainer 训练 --------
    trainer = Trainer(
        model=model,
        args=train_args,
        processing_class=tokenizer,  # 新版 transformers 用这个字段
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    train_stats = trainer.train()
    print(train_stats)

    # -------- 7. 推理阶段：切到推理模式，走原版 Trainer 做预测 --------
    model.eval()
    FastLanguageModel.for_inference(model)

    try:
        prediction_outputs = trainer.predict(test_dataset)
        test_logits = prediction_outputs.predictions
        test_pred = np.argmax(test_logits, axis=-1).flatten()
        print("使用 Unsloth 预测成功")

    except Exception as e:
        print(f"Unsloth 预测失败: {e}")
        print("切换到手动预测...")

        # 方法2：手动预测（作为备选）
        from torch.utils.data import DataLoader

        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        all_predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.append(predictions.cpu().numpy())

        test_pred = np.concatenate(all_predictions, axis=0).flatten()

    print("Test preds shape:", test_pred.shape)

    # -------- 8. 写结果 --------
    result_output = pd.DataFrame(
        data={
            "id": test_df["id"],
            "sentiment": test_pred,
        }
    )
    result_output.to_csv(
        "./result/bert_rdrop_unsloth_lora.csv",
        index=False,
        quoting=3,
    )
    logging.info('result saved!')

