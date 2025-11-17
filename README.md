# imdb_sentiment_analysis_torch

本 README 基于电影评论情感二分类采用了cnn,transformer,gru等经典模型，并进一步在 DeBERTa上尝试多种微调方式（如 Prompt/Prefix/P-Tuning/LoRA 等 PEFT 方法）

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
尝试使用 unsloth 改进微调效率，对 R-Drop、Supervised Contrastive Learning 方法进行实践。

## 模型准确率

| models            | 准确率   |
|-------------------|---------|
| DeBERTa-V2-unsloth | 0.9658 |
| BERT-RDrop         | 0.9374 |
| BERT-SCL           | 0.9137 |
| BERT-RDrop-unsloth-loRA    | 0.9251 |
| BERT-SCL-unsloth-lora | 0.9324 |

### R-Drop介绍

背景：在训练神经网络的过程中，过拟合时有发生，DropOut技术可以解决过拟合问题并且提高模型的泛化能力，但是DropOut的随机性导致了训练和实际应用中模型的不一致性。（即训练阶段采用随机删除单元的方法，而在实际应用的过程中采用的是不删除任何单元的完整模型）本文中介绍了一种简单的方法来正则化由DropOut引起的不一致性，称为R-Drop。

定义：R-Drop通过最小化两个分布之间的双向KL散度，来使得同一份数据的两个子模型输出的两个分布保持一致。与传统的神经网络训练中的DropOut策略相比，R-Drop只是增加了一个没有任何结构改变的KL散度损失。

整体框架结构：R-Drop的总体框架如下，以Transformer为例，左图显示了一个输入x将遍历模型两次，得到两个分布p1和p2，右图显示了由dropout产生的两个不同的子模型。（如图右侧所示，输出预测分布P1和输出分布P2在各层删除的单元各不相同，因此，对于同一输入数据对 (x, y)，P1和P2的分布是不相同的，我们的R-Drop方法试图通过最小化同一样本这两个输出分布之间的双向KL散度来正则化模型预测）。
<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/24da268f-0d33-4f3d-a665-6aa01a2edf93" />

#### R-Drop公式详解
具体来说，以分类问题为例，训练数据为{x_i, y_i} (i=1到n)，模型为 P_θ(y|x)，每个样本的 loss 一般是交叉熵：

```math
\mathcal{L}_i = -\log P_\theta(y_i|x_i) \quad (1)
```

在"Dropout两次"的情况下，其实我们可以认为样本已经通过了两个略有不同的模型，我们分别记为 P_θ^(1)(y|x) 和 P_θ^(2)(y|x)。这时候 R-Drop 的 loss 分为两部分，一部分是常规的交叉熵：

```math
\mathcal{L}^{(CE)}_i = -\log P^{(1)}_\theta(y_i|x_i) - \log P^{(2)}_\theta(y_i|x_i) \quad (2)
```

另一部分则是两个模型之间的对称 KL 散度，它希望不同 Dropout 的模型输出尽可能一致：

```math
\mathcal{L}^{(KL)}_i = \frac{1}{2} \left[ D_{KL} \left( P^{(2)}_\theta(y|x_i) \middle\| P^{(1)}_\theta(y|x_i) \right) + D_{KL} \left( P^{(1)}_\theta(y|x_i) \middle\| P^{(2)}_\theta(y|x_i) \right) \right] \quad (3)
```

最终 loss 就是两个 loss 的加权和：

```math
\mathcal{L}_i = \mathcal{L}^{(CE)}_i + \alpha \mathcal{L}^{(KL)}_i \quad (4)
```

也就是说，它在常规交叉熵的基础上，加了一项强化模型鲁棒性正则项。
#### R-Drop计算流程

**输入：** 训练数据对集合 D = {(x_i, y_i)}^n  
**输出：** 得到模型参数 w  

1. 使用参数 w 来初始化模型
2. 如果没有收敛，则循环执行以下步骤：
   - 随机抽样数据对 (x_i, y_i)
   - 重复输入数据两次，得到两个输出分布
   - 计算对数似然损失函数
   - 计算双向KL散度
   - 通过最小化函数 L 来更新模型参数


