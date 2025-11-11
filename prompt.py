import os
import sys
import logging
import numpy as np
import pandas as pd
import datasets
import evaluate

from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from peft import PromptTuningConfig, get_peft_model, TaskType


# -----------------------
# 读 IMDB TSV
# -----------------------
train_df = pd.read_csv("./labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test_df  = pd.read_csv("./testData.tsv",        header=0, delimiter="\t", quoting=3)


def clean_text(s: pd.Series) -> pd.Series:
    """轻清洗：去掉 IMDB 常见的 <br />，不动其他逻辑。"""
    return s.astype(str).str.replace("<br />", " ", regex=False).str.strip()


if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", "".join(sys.argv))

    # -----------------------
    # 拆分训练/验证（分层抽样）
    # -----------------------
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["sentiment"]
    )

    # 注意：Trainer 期望标签列名为 "labels"
    train_dict = {"labels": train_df["sentiment"].astype(int), "text": clean_text(train_df["review"])}
    val_dict   = {"labels": val_df["sentiment"].astype(int),   "text": clean_text(val_df["review"])}
    test_dict  = {"text": clean_text(test_df["review"])}

    train_ds = datasets.Dataset.from_dict(train_dict)
    val_ds   = datasets.Dataset.from_dict(val_dict)
    test_ds  = datasets.Dataset.from_dict(test_dict)

    # -----------------------
    # 模型与分词器
    # -----------------------
    model_id  = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    MAX_LEN = 384  # 长度可按显存调整（256/320/384/512）

    def preprocess_function(examples):
        # 不在这里 padding；交给 DataCollator 动态 pad
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LEN,
        )

    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["text"])
    tokenized_val   = val_ds.map(preprocess_function,   batched=True, remove_columns=["text"])
    tokenized_test  = test_ds.map(preprocess_function,  batched=True, remove_columns=["text"])

    # DataCollator 动态补齐
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 分类模型（2 类）
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        low_cpu_mem_usage=True,
    )

    # -----------------------
    # ① Prompt Tuning（PEFT）— 调整为更合适的超参
    # -----------------------
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        tokenizer_name_or_path=model_id,
        num_virtual_tokens=32,   # 原 16 -> 32（显存允许可到 64）
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # 调试查看可训练参数

    # -----------------------
    # ② 评估指标
    # -----------------------
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        # 兼容不同版本 transformers 的 EvalPrediction/tuple
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # -----------------------
    # ③ 训练参数（PromptTuning 友好配置）
    # -----------------------
    training_args = TrainingArguments(
        output_dir="./checkpoint",
        seed=42,

        # —— PromptTuning 建议更大学习率（只训虚拟 token）——
        learning_rate=2e-3,             # 常见范围 1e-3 ~ 5e-3
        num_train_epochs=3,             # 适度增加训练轮次
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # 有效 batch = 16

        warmup_ratio=0.06,
        weight_decay=0.01,

        # —— 更稳定评估与保存 ——
        eval_strategy="steps",    # 比按 epoch 更稳
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        fp16=True,                      # 显存友好
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
    )

    # 可选的小优化：Ampere+ 上更快更稳（不支持也无妨）
    try:
        import torch
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # -----------------------
    # 预测提交
    # -----------------------
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs.predictions, axis=-1).astype(int)
    print("Pred sample:", test_pred[:20])

    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame({"id": test_df["id"], "sentiment": test_pred})
    result_output.to_csv("./result/deberta_prompt_tuning.csv", index=False, quoting=3)
    logging.info("result saved!")
