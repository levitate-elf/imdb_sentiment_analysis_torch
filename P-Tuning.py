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
    set_seed,
)
from peft import PromptEncoderConfig, get_peft_model, TaskType


# -----------------------
# 基础设置
# -----------------------
SEED = 42
set_seed(SEED)

TRAIN_PATH = "./labeledTrainData.tsv"
TEST_PATH  = "./testData.tsv"
MODEL_ID   = "microsoft/deberta-v3-base"   # 需要 sentencepiece 依赖
MAX_LEN    = 384                           # 文本截断长度；可调 256/384/512
VIRTUAL_TOKENS = 16                        # Prompt token 数；可调 8/16/32

# -----------------------
# 读 IMDB TSV
# -----------------------
train_df = pd.read_csv(TRAIN_PATH, header=0, delimiter="\t", quoting=3)
test_df  = pd.read_csv(TEST_PATH,  header=0, delimiter="\t", quoting=3)

if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", " ".join(sys.argv))

    # -----------------------
    # 划分训练/验证（分层抽样，保证 0/1 比例相近）
    # -----------------------
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=SEED, stratify=train_df["sentiment"]
    )

    # Trainer 期望标签列名为 "labels"
    train_dict = {"labels": train_df["sentiment"].astype(int), "text": train_df["review"].astype(str)}
    val_dict   = {"labels": val_df["sentiment"].astype(int),   "text": val_df["review"].astype(str)}
    test_dict  = {"text": test_df["review"].astype(str)}

    train_ds = datasets.Dataset.from_dict(train_dict)
    val_ds   = datasets.Dataset.from_dict(val_dict)
    test_ds  = datasets.Dataset.from_dict(test_dict)

    # -----------------------
    # 分词器与预处理
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    def preprocess_function(examples):
        # 不在这里 padding；交给 DataCollator 动态 pad 更省显存
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["text"])
    tokenized_val   = val_ds.map(preprocess_function,   batched=True, remove_columns=["text"])
    tokenized_test  = test_ds.map(preprocess_function,  batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -----------------------
    # 基座分类模型（2 类）
    # -----------------------
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        low_cpu_mem_usage=True,
    )

    # 重要：开启梯度检查点，进一步省显存（速度会稍慢）
    # 对 DeBERTa v3，这样启用即可；若报不支持，可改为 TrainingArguments(gradient_checkpointing=True)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    # -----------------------
    # 仅添加 Prompt Tuning（PEFT）
    # -----------------------
    peft_config = PromptEncoderConfig(
        num_virtual_tokens=20,
        encoder_hidden_size=128,
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(base_model, peft_config)
    # 可打印看下只训练的参数量
    model.print_trainable_parameters()

    # -----------------------
    # 指标
    # -----------------------
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # -----------------------
    # 训练参数（显存友好）
    # -----------------------
    training_args = TrainingArguments(
        output_dir="./checkpoint",
        num_train_epochs=4,                 # 训 3~5 epoch 都可；Prompt Tuning 收敛很快
        learning_rate=2e-5,                 # 可在 2e-5 ~ 5e-5 微调
        per_device_train_batch_size=4,      # 小 batch
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,      # 有效 batch = 4 * 4 = 16
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",                 # 如需保存每轮检查点可改 "epoch"
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,                          # FP16 省显存（非 A100 也可开）
        gradient_checkpointing=True,        # 再保险地从 Trainer 层打开
        report_to="none",
        load_best_model_at_end=False,       # 想根据指标挑最好的一轮可设 True + metric_for_best_model
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # -----------------------
    # 训练
    # -----------------------
    trainer.train()

    # -----------------------
    # 预测提交
    # -----------------------
    pred_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(pred_outputs.predictions, axis=-1).astype(int)

    os.makedirs("./result", exist_ok=True)
    pd.DataFrame({"id": test_df["id"], "sentiment": test_pred}).to_csv(
        "./result/deberta_P-Tuning.csv", index=False, quoting=3
    )
    logging.info("result saved! -> ./result/deberta_P-Tuning.csv")
