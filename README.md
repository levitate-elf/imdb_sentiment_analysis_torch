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

## 4. 最后结论

- bert_native的训练准确率在它的最后记录点已经达到 0.98，distill_native达到 0.97，说明它们对训练数据拟合得很好。且都在非常早期（epoch=1，也就是只训练了大约两轮之后）就达到了约 0.92 的验证准确率。这说明基于预训练表示的模型在极少的训练轮数内就可以达到很高的性能，收敛速度非常快
- 纯卷积的cnn在验证集上最高为 0.81，明显落后于其他模型。这说明单一cnn结构在该任务上缺乏对长程依赖的表达能力。
- attention_lstm,capsule_lstm,gru这些包含序列建模或注意力机制的结构，验证准确率在 0.87～0.89 区间，稳定可用





