import logging, os, sys, time, math, re
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

# -------------------- 超参 --------------------
num_epochs   = 10
embed_size   = 300
num_hiddens  = 120
batch_size   = 64
labels       = 2
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN      = 256
LR           = 5e-4
WEIGHT_DECAY = 1e-4
MIN_FREQ     = 2
DROPOUT      = 0.2      # Transformer 层内 dropout
LABEL_SMOOTH = 0.05     # 更稳的分类损失

# -------------------- 数据读取 --------------------
train = pd.read_csv("./labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test  = pd.read_csv("./testData.tsv",        header=0, delimiter="\t", quoting=3)

# -------------------- 文本清洗 --------------------
def review_to_wordlist(review):
    text = BeautifulSoup(review, "lxml").get_text(" ")
    text = re.sub("[^a-zA-Z]", " ", text)
    return text.lower().split()            # 注意：返回“词列表”，不是字符串

# -------------------- 词表 --------------------
class Vocab:
    def __init__(self, tokens, reserved_tokens=None):
        reserved_tokens = (reserved_tokens or [])
        self.idx_to_token, self.token_to_idx = [], {}
        # 先放保留符号（稳定索引）
        for tok in reserved_tokens:
            if tok not in self.token_to_idx:
                self.token_to_idx[tok] = len(self.idx_to_token)
                self.idx_to_token.append(tok)
        # 再放普通词
        for tok in tokens:
            if tok not in self.token_to_idx:
                self.token_to_idx[tok] = len(self.idx_to_token)
                self.idx_to_token.append(tok)
        self.pad = self.token_to_idx["<pad>"]
        self.unk = self.token_to_idx["<unk>"]

    @classmethod
    def build(cls, train_sentences, min_freq=1, reserved_tokens=None):
        cnt = Counter(tok for sent in train_sentences for tok in sent)
        reserved_tokens = (reserved_tokens or ["<pad>", "<unk>"])
        base = [t for t, f in cnt.items()
                if f >= min_freq and t not in set(reserved_tokens)]
        return cls(base, reserved_tokens=reserved_tokens)

    def __len__(self): return len(self.idx_to_token)
    def __getitem__(self, token): return self.token_to_idx.get(token, self.unk)
    def convert_tokens_to_ids(self, tokens): return [self[token] for token in tokens]

# -------------------- 长度 -> 掩码 --------------------
def length_to_mask(lengths):
    max_len = int(lengths.max().item())
    row = torch.arange(max_len, device=lengths.device)
    return row.unsqueeze(0).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)  # True=有效

# -------------------- 位置编码 --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len,1,d)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):               # x: (L,B,d)
        L = x.size(0)
        return self.dropout(x + self.pe[:L])

# -------------------- Transformer 分类器 --------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2,
                 dropout=DROPOUT, max_len=MAX_LEN, activation="relu"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos = PositionalEncoding(embedding_dim, dropout, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_head,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.out = nn.Linear(embedding_dim, num_class)

    def forward(self, inputs, lengths):
        # inputs: (B,L) -> (L,B,d)
        x = self.emb(inputs).transpose(0, 1)
        x = self.pos(x)
        pad_mask = ~length_to_mask(lengths)         # (B,L) True=padding
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x.transpose(0, 1)                       # (B,L,d)

        # 掩码平均池化
        valid = (~pad_mask).unsqueeze(-1)           # (B,L,1)
        summed = (x * valid).sum(1)                 # (B,d)
        denom  = valid.sum(1).clamp(min=1)          # (B,1)
        pooled = summed / denom

        logits = self.out(pooled)                   # ★ 返回原始 logits（不要 log_softmax）
        return logits

# -------------------- Dataset & collate --------------------
class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def head_tail(ids, max_len=MAX_LEN, head_ratio=0.7):
    if len(ids) <= max_len: return ids
    h = int(max_len * head_ratio); t = max_len - h
    return ids[:h] + ids[-t:]

def collate_fn(examples, pad_id=0):
    clipped = []
    for ex in examples:
        if isinstance(ex, tuple):     # (ids, label)
            ids, lab = ex
            ids = head_tail(ids, MAX_LEN)
            clipped.append((ids, lab))
        else:
            ids = head_tail(ex, MAX_LEN)
            clipped.append(ids)

    if isinstance(clipped[0], tuple):
        lengths = torch.tensor([len(ids) for ids, _ in clipped], dtype=torch.long)
        inputs  = [torch.tensor(ids, dtype=torch.long) for ids, _ in clipped]
        labels  = torch.tensor([lab for _, lab in clipped], dtype=torch.long)
        inputs  = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        return inputs, lengths, labels
    else:
        lengths = torch.tensor([len(ids) for ids in clipped], dtype=torch.long)
        inputs  = [torch.tensor(ids, dtype=torch.long) for ids in clipped]
        inputs  = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        return inputs, lengths

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s",
                        level=logging.INFO)
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logger.info("running %s", ''.join(sys.argv))

    # 1) 清洗 & 标签
    clean_train = [review_to_wordlist(r) for r in train["review"]]
    clean_test  = [review_to_wordlist(r) for r in test["review"]]
    train_labels = train["sentiment"].tolist()

    # 2) 词表（只用训练集），保证 <pad>=0
    vocab = Vocab.build(clean_train, min_freq=MIN_FREQ)

    # 3) ID 化
    train_ids = [vocab.convert_tokens_to_ids(s) for s in clean_train]
    test_ids  = [vocab.convert_tokens_to_ids(s) for s in clean_test]

    # 4) 切分
    train_pack, val_pack, y_train, y_val = train_test_split(
        train_ids, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    train_data = list(zip(train_pack, y_train))
    val_data   = list(zip(val_pack,   y_val))

    # 5) DataLoader
    train_loader = torch.utils.data.DataLoader(
        TransformerDataset(train_data), batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_id=vocab.pad)
    )
    val_loader = torch.utils.data.DataLoader(
        TransformerDataset(val_data),   batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_id=vocab.pad)
    )
    test_loader = torch.utils.data.DataLoader(
        TransformerDataset(test_ids),   batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_id=vocab.pad)
    )

    # 6) 模型/优化器/调度器/损失
    net = Transformer(vocab_size=len(vocab), embedding_dim=embed_size, num_class=labels,
                      dropout=DROPOUT, max_len=MAX_LEN).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 7) 早停：按 val loss
    os.makedirs("./checkpoint", exist_ok=True)
    best_val, best_path = float("inf"), "./checkpoint/transformer_best.pt"

    # 8) 训练
    for epoch in range(num_epochs):
        net.train()
        n, train_loss_sum, train_right = 0, 0.0, 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
            for feat, lengths, label in train_loader:
                feat, lengths, label = feat.to(device), lengths.to(device), label.to(device)
                optimizer.zero_grad()
                logits = net(feat, lengths)               # ★ logits（未 softmax）
                loss   = loss_fn(logits, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # 统计
                n += label.size(0)
                train_loss_sum += loss.item() * label.size(0)
                train_right += (logits.argmax(1) == label).sum().item()
                pbar.set_postfix({
                    "train loss": f"{train_loss_sum/n:.4f}",
                    "train acc":  f"{train_right/n:.2f}"
                })
                pbar.update(1)

        # 验证
        net.eval()
        m, val_loss_sum, val_right = 0, 0.0, 0
        with torch.no_grad():
            for vfeat, vlen, vlab in val_loader:
                vfeat, vlen, vlab = vfeat.to(device), vlen.to(device), vlab.to(device)
                vlogits = net(vfeat, vlen)
                vloss   = loss_fn(vlogits, vlab)
                m += vlab.size(0)
                val_loss_sum += vloss.item() * vlab.size(0)
                val_right += (vlogits.argmax(1) == vlab).sum().item()

        train_loss = train_loss_sum / n
        train_acc  = train_right / n
        val_loss   = val_loss_sum / m
        val_acc    = val_right / m
        print(f"[Epoch {epoch}] train loss={train_loss:.4f}, train acc={train_acc:.2f}, "
              f"val loss={val_loss:.4f}, val acc={val_acc:.2f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(net.state_dict(), best_path)
            print(f"[BEST] val_loss={best_val:.4f} saved -> {best_path}")

    # 9) 载入最好模型 → 预测
    state = torch.load(best_path, map_location=device)
    net.load_state_dict(state)
    net.eval()

    preds = []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Prediction") as pbar:
            for tfeat, tlen in test_loader:
                tfeat, tlen = tfeat.to(device), tlen.to(device)
                logits = net(tfeat, tlen)
                preds.extend(logits.argmax(1).cpu().tolist())
                pbar.update(1)

    os.makedirs("./result", exist_ok=True)
    pd.DataFrame({"id": test["id"], "sentiment": preds}).to_csv(
        "./result/transformer.csv", index=False, quoting=3
    )
    logging.info("result saved!")
