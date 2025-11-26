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
### SCL（Supervised Contrastive Learning）介绍

背景：在传统的监督学习中，交叉熵损失函数被广泛使用，但它主要关注样本与真实标签之间的关系，忽略了样本之间的内在结构信息。监督对比学习（SCL）通过利用标签信息来构建正负样本对，学习更具判别性的特征表示。

定义：SCL通过将同一类别的样本在特征空间中拉近，不同类别的样本推远，来学习更具判别性的特征表示。与传统的交叉熵损失相比，SCL通过对比学习的方式增强了模型的泛化能力和特征表示质量。

整体框架结构：SCL的总体框架如下，对于每个输入样本，通过数据增强得到多个视图，利用标签信息构建正样本对（同类样本）和负样本对（不同类样本），通过对比损失函数来优化特征表示。
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/c3a02223-028f-4816-9474-bc89e9f7037d" />

#### SCL公式详解
具体来说，以分类问题为例，训练数据为{x_i, y_i} (i=1到n)，模型为 f_θ(x)，每个样本的特征表示为 z_i = f_θ(x_i)。

在监督对比学习中，对于每个锚点样本 i，其损失函数定义为：

```math
\mathcal{L}_{SCL}^{(i)} = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)} \quad (1)
```

其中：

- P(i) 是与样本 i 同一类别的正样本集合（不包括 i 自身）
- A(i) 是批次中除 i 外的所有样本集合
- τ 是温度超参数
- z_i · z_j 表示特征向量的点积相似度

  
# unsloth 调用重写的模型

尝试使用 unsloth 调用重写后的模型。

## 1.模型加载阶段

   ```python
   model, tokenizer = FastModel.from_pretrained(
       model_name=model_name,
       max_seq_length=512,
       auto_model=DebertaV3ForSequenceClassificationSupCon,
       num_labels=NUM_CLASSES,
       gpu_memory_utilization=0.8,
   )
```
使用 Unsloth 的 FastModel 加载模型
自动应用 Unsloth 的内存优化和计算优化
## 2.lora配置阶段
  ```python
model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",  # Unsloth 特有的梯度检查点
    target_modules="all-linear",
)
```
使用 Unsloth 的 LoRA 实现，相比原始 PEFT 有额外优化

## 3.训练阶段
 ```python
trainer = Trainer(  # 这是被 Unsloth patch 过的 Trainer
    model=model,
    args=train_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
```

Trainer 已经被 Unsloth 修改过，包含训练加速
## 4.推理阶段
 ```python
model.eval()
FastLanguageModel.for_inference(model)  # Unsloth 的推理优化
```
显式调用 Unsloth 的推理优化方法
即使使用标准 DataLoader，模型本身仍受益于 Unsloth 的优化

# 指令学习在SST-2数据集上的性能总结

## 模型性能对比

| 模型 | SST-2准确度 |  
| :--------:   | :-----: |
|   **Llama-3.3-70B** | 0.9427 |
| **Qwen3-max** | 0.9587 | 
| **Gemma-2B** | 0.9087 | 
| **Phi-3-mini 3.8B** | 0.9415 | 
| **Mistral-7B** | 0.9151 |
| **deepseek-v3.1** | 0.9576 |

## 具体实现
下面以qwen3-max为例，展示prompt构造，api调用，评估逻辑 

### prompt构造

```python
def build_prompt(sentence: str) -> str:
    """
    把一条影评句子包装成指令式 prompt
    """
    return f"""You are a sentiment analysis assistant.
Classify the sentiment of the following movie review as "positive" or "negative".

Review: "{sentence}"

Answer with only one word: positive or negative.
"""
```
由于Prompt 没有任何标注示例，所以严格来说是 zero-shot 指令学习，完全靠自然语言把任务描述清楚，让模型“自己悟出”如何从影评判断情感。

### api调用

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_DASHSCOPE_API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL_NAME = "qwen3-max"   # DashScope 上的 Qwen3 模型


def predict_label(sentence: str) -> int:
    """
    调用 qwen3-max，返回标签 0/1：
      0 -> negative
      1 -> positive
    """
    prompt = build_prompt(sentence)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful sentiment analysis assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,   # 设成 0，让输出尽量稳定
        max_tokens=5,      # 只要 "positive"/"negative" 这种短输出
    )

    text = resp.choices[0].message.content.strip().lower()

    if "positive" in text:
        return 1
    if "negative" in text:
        return 0

    return 0
```

### 评估逻辑

```python
def eval_sst2_on_qwen():
    total = len(valid_ds)
    correct = 0
    for sample in tqdm(valid_ds, desc="Evaluating SST-2 (validation) with qwen3-max"):
        sentence = sample["sentence"]
        gold = sample["label"]   # 0 or 1

        pred = predict_label(sentence)

        if pred == gold:
            correct += 1

    acc = correct / total
    print(f"\nAccuracy on SST-2 (validation): {acc:.4f}  ({correct}/{total})")
```
## 总结
当前为zero-shot 指令学习，模型在 SST-2 上没有经过任何专门训练，只靠指令理解，准确率会受 prompt 写法影响较大，通常略低于精调的 RoBERTa/BERT 模型。


