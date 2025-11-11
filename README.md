# imdb_sentiment_analysis_torch

本 README 汇总了当前提交的所有模型在训练集和验证集上的表现，便于比较不同模型的准确率

## 1. 评测设定

- 数据集：使用相同的数据划分（同一训练集 / 验证集）。
- 指标：
  - tra_acc：训练集准确率（training accuracy）
  - val_acc：验证集准确率（validation accuracy）
- 训练记录：
  - 每个 epoch 记录训练准确率，验证准确率。
  - 部分模型只有3个epoch且无验证准确率。
---

## 2.模型数据及其环境

- 使用的数据集
  - labeledTrainData.tsv
  - unlabeledTrainData.tsv
  - testData.tsv
  - imdb_glove_pickle3
- 运行环境
  - Python 环境
    - python 3.12
  - 核心依赖
      - torch
      - transformers
      - BeautifulSoup
      - pandas
      - scikit-learn
      -  Hugging Face Datasets等
---

## 3. 各模型表现总览

| 模型名              | 最终训练准确率 | 最终验证准确率 | 最佳验证准确率 | 出现的 epoch |
|:-------------------:|:--------------:|:--------------:|:--------------:|:------------:|
| attention_lstm      | 0.92           | 0.87           | 0.87           | 7            |
| bert_native         | 0.98           | 0.91           | 0.92           | 1            |
| bert_scratch        | 0.93           | -              | -              | -            |
| bert_trainer        | 0.94           | -              | -              | -            |
| capsule_lstm        | 0.93           | 0.89           | 0.89           | 6            |
| cnn                 | 0.87           | 0.80           | 0.81           | 6            |
| cnnlstm             | 0.92           | 0.86           | 0.96           | 7            |
| distill_native      | 0.97           | 0.91           | 0.92           | 1            |
| distill_trainer     | 0.93           | -              | -              | -            |
| gru                 | 0.90           | 0.82           | 0.88           | 4            |
| lstm                | 0.82           | 0.82           | 0.82           | 9            |
| roberta_trainer     | 0.94           | -              | -              | -            |
| transformer         | 0.94           | 0.88           | 0.88           | 8            |



说明：
- `-` 表示该模型没有对应的验证集准确率，因此无法统计最终验证准确率，最佳验证准确率和其对应的 epoch。

---

## 4. DeBERTa-v3-base 的 PEFT 结果

> 相同训练轮次下，记录各 epoch 验证准确率（`val_acc`）：

| 方法     | epoch=1 | epoch=2 | epoch=3 |
|:---------|--------:|--------:|--------:|
| Prompt   | 0.5292  | 0.5782  | 0.6379  |
| P-Tuning | 0.5088  | 0.6384  | 0.6745  |
| Prefix   | 0.5198  | 0.5292  | 0.5343  |
| Lora     |0.9073   |0.9284   |0.9346   |

### 4.1 显存与模型选择
- 在同等设定下，**`microsoft/deberta-v2-xxlarge`**在 **16 GB GPU** 上多次 **OOM**（`max_length ≥ 256`、`batch_size ≥ 2`、未量化或仅 4bit 量化时尤甚）。  
- 为确保复现与稳定训练统一以 **`microsoft/deberta-v3-base`** 作为骨干进行 PEFT（LoRA / P-Tuning / Prompt /Prefix）：
  - `deberta-v3-base` 采用8bit量化 可达 **0.93+** 的 `val_acc`，显存压力显著低于 `v2-xxlarge`；
  - 显存更宽裕可尝试 `deberta-v3-large`。

---






