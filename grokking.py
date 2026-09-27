"""
Grokking reproduction: train a tiny Transformer on modular addition
(a + b) mod p, and watch for delayed generalization - the phenomenon
where test accuracy stays near-zero long after training accuracy hits
100%, then suddenly jumps up much later.

Paper: "Grokking: Generalization Beyond Overfitting on Small Algorithmic
Datasets" (Power et al., 2022) - https://arxiv.org/abs/2201.02177
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ---------------- CONFIG ----------------
P = 97                  # prime modulus - task is (a + b) mod P
TRAIN_FRACTION = 0.3    # lower fraction = harder task = more pronounced grokking delay
                        # (paper shows dramatic effects often below 40% for mod arithmetic)
N_EMBD = 128
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 512
TOTAL_STEPS = 20000      # need to train LONG PAST train-accuracy saturating
LOG_EVERY = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0        # weight decay matters a lot for grokking to occur
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
# Each example is a sequence: [a, b, '=', result]
# We'll represent tokens as: 0..P-1 are the numbers, P is '=' token,
# P+1 is a padding/start token if needed. Vocab size = P + 1 (just '=').

def make_dataset():
    pairs = [(a, b) for a in range(P) for b in range(P)]
    results = [(a + b) % P for (a, b) in pairs]

    # each input sequence: [a, b, EQUALS] -> predict result
    EQUALS = P  # token id for '='
    inputs = torch.tensor([[a, b, EQUALS] for (a, b) in pairs], dtype=torch.long)
    targets = torch.tensor(results, dtype=torch.long)

    # shuffle and split into train/test
    n = len(inputs)
    perm = torch.randperm(n)
    inputs, targets = inputs[perm], targets[perm]

    n_train = int(n * TRAIN_FRACTION)
    train_x, train_y = inputs[:n_train], targets[:n_train]
    test_x, test_y = inputs[n_train:], targets[n_train:]

    return train_x, train_y, test_x, test_y


# ---------------- TINY TRANSFORMER ----------------
# Reuses the same building blocks you already understand: embedding,
# self-attention, residual connections, layernorm - just much smaller.

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits  # (B, T, vocab_size)


# ---------------- TRAINING LOOP ----------------

def get_accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        # we only care about the prediction at the LAST position
        # (right after the '=' token, predicting the result)
        preds = logits[:, -1, :].argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    model.train()
    return acc


def train():
    train_x, train_y, test_x, test_y = make_dataset()
    train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
    test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)

    vocab_size = P + 1  # numbers 0..P-1, plus '=' token
    block_size = 3       # sequence length: [a, b, '=']

    model = TinyGPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, block_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    steps_log, train_acc_log, test_acc_log = [], [], []

    n_train = train_x.size(0)

    for step in range(TOTAL_STEPS):
        # sample a random batch from the training set
        idx = torch.randint(0, n_train, (min(BATCH_SIZE, n_train),))
        batch_x, batch_y = train_x[idx], train_y[idx]

        logits = model(batch_x)
        loss = F.cross_entropy(logits[:, -1, :], batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            train_acc = get_accuracy(model, train_x, train_y)
            test_acc = get_accuracy(model, test_x, test_y)
            steps_log.append(step)
            train_acc_log.append(train_acc)
            test_acc_log.append(test_acc)
            print(f"Step {step:6d} | loss {loss.item():.4f} | train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    # Plot the results - this is the key output of the whole experiment
    plt.figure(figsize=(10, 6))
    plt.plot(steps_log, train_acc_log, label="Train accuracy")
    plt.plot(steps_log, test_acc_log, label="Test accuracy")
    plt.xlabel("Training step")
    plt.ylabel("Accuracy")
    plt.title(f"Grokking on (a+b) mod {P}")
    plt.legend()
    plt.savefig("grokking_curve.png")
    plt.show()
    print("\nSaved plot to grokking_curve.png")


if __name__ == "__main__":
    train()
