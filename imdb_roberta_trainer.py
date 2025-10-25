import os
import sys
import logging
import datasets

import pandas as pd
import numpy as np

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import evaluate
train = pd.read_csv("./labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", ''.join(sys.argv))

    train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

    train_ds = datasets.Dataset.from_pandas(
        train_df[["sentiment", "review"]].rename(columns={"sentiment": "label", "review": "text"})
    )
    val_ds = datasets.Dataset.from_pandas(
        val_df[["sentiment", "review"]].rename(columns={"sentiment": "label", "review": "text"})
    )
    test_ds = datasets.Dataset.from_pandas(
        test[["review"]].rename(columns={"review": "text"})
    )

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train =   train_ds.map(preprocess_function, batched=True, remove_columns=["text", "__index_level_0__"])
    tokenized_val = val_ds.map(preprocess_function,   batched=True, remove_columns=["text", "__index_level_0__"])
    tokenized_test = test_ds.map(preprocess_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return metric.compute(predictions=preds, references=eval_pred.label_ids)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        learning_rate=5e-6,
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
    result_output.to_csv("./result/roberta_trainer.csv", index=False, quoting=3)
    logging.info('result saved!')
