import os
import sys
import logging
import datasets
import warnings
import time
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    DebertaV2Tokenizer, 
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from peft import PrefixTuningConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("🚀 开始DeBERTa Prefix Tuning训练方案...")

# 设置随机种子确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 读取数据
print("=== 读取数据 ===")
train = pd.read_csv("./labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./testData.tsv", header=0, delimiter="\t", quoting=3)

# 数据预处理
train, val = train_test_split(train, test_size=0.2, random_state=42)
train['sentiment'] = train['sentiment'].astype(int)
val['sentiment'] = val['sentiment'].astype(int)

print(f"数据加载完成 - 训练集: {len(train)}, 验证集: {len(val)}")

# 加载模型和tokenizer
model_id = "microsoft/deberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建数据集
def preprocess_function(examples):
    tokenized = tokenizer(
        examples['text'], 
        truncation=True, 
        max_length=256,
        padding=False
    )
    tokenized['labels'] = examples['label']
    return tokenized

train_dataset = datasets.Dataset.from_dict({
    'text': train['review'].tolist(),
    'label': train['sentiment'].tolist()
})
val_dataset = datasets.Dataset.from_dict({
    'text': val['review'].tolist(), 
    'label': val['sentiment'].tolist()
})

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

# 加载模型
print("=== 加载模型 ===")
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    torch_dtype=torch.float32,
)

# 使用Prefix Tuning配置
peft_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=20,
    encoder_hidden_size=768
)

# 手动实现Prefix Tuning的前向传播
class CustomPrefixTuningModel(nn.Module):
    def __init__(self, model, peft_config):
        super().__init__()
        self.model = model
        self.peft_config = peft_config
        self.num_virtual_tokens = peft_config.num_virtual_tokens
        
        # 创建prefix embeddings
        self.prefix_embeddings = nn.Embedding(
            self.num_virtual_tokens, 
            self.model.config.hidden_size
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        
        # 创建prefix tokens
        prefix_tokens = torch.arange(self.num_virtual_tokens).repeat(batch_size, 1).to(input_ids.device)
        prefix_embeds = self.prefix_embeddings(prefix_tokens)
        
        # 获取原始输入的embeddings
        inputs_embeds = self.model.deberta.embeddings(input_ids)
        
        # 拼接prefix和原始输入
        combined_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
        
        # 调整attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, self.num_virtual_tokens).to(attention_mask.device)
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # 通过模型
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels
        )
        
        return outputs

# 创建自定义模型
custom_model = CustomPrefixTuningModel(model, peft_config)
custom_model.to(model.device)

# 只训练prefix embeddings
for name, param in custom_model.named_parameters():
    if 'prefix_embeddings' in name:
        param.requires_grad = True
        print(f"训练参数: {name}")
    else:
        param.requires_grad = False

# 计算可训练参数
trainable_params = sum(p.numel() for p in custom_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in custom_model.parameters())
print(f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {100 * trainable_params / total_params:.4f}")

# 移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model.to(device)
print(f"模型已移动到: {device}")

# 创建数据加载器
train_dataloader = DataLoader(
    tokenized_train, 
    batch_size=8,
    shuffle=True, 
    collate_fn=data_collator,
    num_workers=0
)
val_dataloader = DataLoader(
    tokenized_val, 
    batch_size=16,
    collate_fn=data_collator,
    num_workers=0
)

# 优化器和学习率调度器
optimizer = torch.optim.AdamW(
    custom_model.parameters(), 
    lr=1e-3,
    weight_decay=0.01
)

total_steps = len(train_dataloader) * 3
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# 训练函数
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_steps = max(1, len(dataloader) // 10)
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        try:
            # 前向传播
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 进度报告
            if step % progress_steps == 0:
                current_loss = loss.item()
                print(f"📊 Step {step}/{len(dataloader)} - Loss: {current_loss:.4f}")
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    print(f"💾 GPU内存: {memory_used:.2f}GB")
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("💥 内存不足，跳过该batch")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return total_loss / len(dataloader)

# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            try:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("💥 评估时内存不足，跳过该batch")
                    continue
                else:
                    raise e
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

print("=" * 60)
print("🎯 开始自定义Prefix Tuning训练循环")
print("=" * 60)

# 训练循环
num_epochs = 3
for epoch in range(num_epochs):
    print(f"\n🔥 开始第 {epoch+1}/{num_epochs} 个epoch")
    start_time = time.time()
    
    # 训练
    train_loss = train_epoch(custom_model, train_dataloader, optimizer, scheduler, device)
    
    # 评估
    val_accuracy = evaluate(custom_model, val_dataloader, device)
    
    epoch_time = time.time() - start_time
    print(f"✅ Epoch {epoch+1} 完成 - 耗时: {epoch_time:.2f}s")
    print(f"📊 训练损失: {train_loss:.4f}, 验证准确率: {val_accuracy:.4f}")

print("\n🎉 训练完成!")

# 预测函数
def predict(model, test_data, tokenizer, device):
    model.eval()
    all_predictions = []
    
    test_dataset = datasets.Dataset.from_dict({
        'text': test_data['review'].tolist()
    })
    
    tokenized_test = test_dataset.map(
        lambda examples: tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=256,
            padding=False
        ),
        batched=True,
        remove_columns=['text']
    )
    
    test_dataloader = DataLoader(
        tokenized_test, 
        batch_size=16,
        collate_fn=data_collator,
        num_workers=0
    )
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            try:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_predictions.extend(preds.cpu().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("💥 预测时内存不足，跳过该batch")
                    continue
                else:
                    raise e
    
    return all_predictions

print("\n=== 开始预测 ===")
test_predictions = predict(custom_model, test, tokenizer, device)

# 保存结果
result_output = pd.DataFrame({
    "id": test["id"], 
    "sentiment": test_predictions
})
result_output.to_csv("./result/deberta_prefix.csv", index=False)
print("✅ 预测结果保存成功!")

print(f"预测分布:\n{pd.Series(test_predictions).value_counts().sort_index()}")

print("=" * 60)
print("🏁 所有任务完成!")
print("=" * 60)