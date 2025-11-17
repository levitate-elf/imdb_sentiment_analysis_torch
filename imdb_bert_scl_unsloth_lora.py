import os
os.environ["UNSLOTH_DISABLE_STATS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
import logging

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import unsloth
from unsloth import FastModel, FastLanguageModel

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import losses  # 你自己的 SupConLoss

from transformers import (
    TrainingArguments,
    Trainer,
    training_args,
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    DataCollatorWithPadding,
)
from transformers import Trainer as TransformersTrainer
from transformers.modeling_outputs import SequenceClassifierOutput


# ======================
#  DeBERTa-v3 + SupConLoss 模型
# ======================
class DebertaV3ForSequenceClassificationSupCon(DebertaV2PreTrainedModel):
    """
    DeBERTa-v3 二分类 + 监督对比学习 (SupConLoss)
    总 loss = CE + alpha * SupConLoss
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.deberta = DebertaV2Model(config)

        classifier_dropout = (
            getattr(config, "classifier_dropout", None)
            if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # SupConLoss 权重
        self.alpha = 0.2

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        # DeBERTa-v3 backbone
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        # last_hidden_state: [B, L, H]
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # [CLS], [B, H]

        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)   # [B, num_labels]

        loss = None
        if labels is not None:
            # 交叉熵
            ce_fct = nn.CrossEntropyLoss()
            ce_loss = ce_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

            # 监督对比损失 —— 用你自己的 SupConLoss
            scl_fct = losses.SupConLoss()
            # ★ 关键改动：把特征扩展为 [B, 1, H]，符合 SupConLoss 的 [bsz, n_views, ...] 接口
            features = cls_output.unsqueeze(1)   # [B, 1, H]
            scl_loss = scl_fct(features, labels)

            # 可选：简单 NaN 防护，避免一次偶发 NaN 直接把整个训练搞崩
            if torch.isnan(scl_loss) or torch.isinf(scl_loss):
                scl_loss = 0.0 * ce_loss

            loss = ce_loss + self.alpha * scl_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ======================
#  主程序：IMDB + DeBERTa-v3 + SupCon + Unsloth + LoRA
# ======================
if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logging.root.setLevel(logging.INFO)
    logger.info("running %s", " ".join(sys.argv))

    # 1. 读 IMDB 数据
    train_df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test_df = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=3407, shuffle=True
    )

    train_dict = {
        "label": train_df["sentiment"].astype(int),
        "text": train_df["review"],
    }
    val_dict = {
        "label": val_df["sentiment"].astype(int),
        "text": val_df["review"],
    }
    test_dict = {
        "text": test_df["review"],
    }

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # 2. 用 Unsloth + FastModel 加载 DeBERTa-v3-base，并替换为自定义 SupCon 模型
    model_name = "microsoft/deberta-v3-base"
    NUM_CLASSES = 2

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,       # 你要是想省显存可以改 True
        max_seq_length=512,
        dtype=None,
        auto_model=DebertaV3ForSequenceClassificationSupCon,
        num_labels=NUM_CLASSES,
        gpu_memory_utilization=0.8,
    )

    # 3. 在模型上挂 LoRA
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
        target_modules="all-linear",
        task_type="SEQ_CLS",
    )

    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    # 4. Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=512,
            truncation=True,
            padding=True,
        )

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    test_dataset = test_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # 全部列转 tensor（label, input_ids, attention_mask, 可能还会有 token_type_ids）
    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")
    test_dataset.set_format(type="torch")

    # 5. 评估指标：准确率
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # 6. 训练参数（保持和你原 unsloth 脚本风格一致）
    train_args = TrainingArguments(
        output_dir="./checkpoint_deberta_scl_unsloth_lora",
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

    # 7. 用（被 Unsloth patch 的）Trainer 训练
    trainer = Trainer(
        model=model,
        args=train_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    train_stats = trainer.train()
    print(train_stats)

    # 8. 推理：用原版 TransformersTrainer 做预测（避免 Unsloth 的 prompt 钩子）
    model.eval()
    FastLanguageModel.for_inference(model)

    device = next(model.parameters()).device

    # 方法1：使用 DataCollatorWithPadding 确保批次填充
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=data_collator,  # 关键：使用数据收集器进行填充
        shuffle=False
    )
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            # Dataset.set_format(type="torch") 之后，batch 是一个 dict[str, tensor]
            # 只把模型会用到的键搬到 GPU
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "token_type_ids"]
            }
            outputs = model(**batch)
            logits = outputs.logits          # forward 返回的是 SequenceClassifierOutput
            all_logits.append(logits.cpu().numpy())

    test_logits = np.concatenate(all_logits, axis=0)
    test_pred = np.argmax(test_logits, axis=-1).flatten()
    print("Test preds shape:", test_pred.shape)

    # 9. 写结果
    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(
        data={
            "id": test_df["id"],
            "sentiment": test_pred,
        }
    )
    result_output.to_csv(
        "./result/deberta_scl_unsloth_lora.csv",
        index=False,
        quoting=3,
    )
    logging.info("Result saved to ./result/deberta_scl_unsloth_lora.csv")
