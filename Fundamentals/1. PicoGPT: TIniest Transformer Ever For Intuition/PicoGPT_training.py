import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Settings
vocab_size = 5
d_model = 2
d_ff = 4
seq_len = 2

# Tiny toy dataset: (input_tokens, next_token_target)
toy_dataset = [
    ([1, 2], 3),
    ([0, 1], 2),
    ([2, 3], 4),
    ([3, 4], 1)
]

# Toy Transformer Block
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)  # (B, T, d)
        Q = self.W_Q(emb)
        K = self.W_K(emb)
        V = self.W_V(emb)

        scores = Q @ K.transpose(-2, -1) / d_model**0.5
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        scores += mask

        attn = F.softmax(scores, dim=-1)
        out = attn @ V

        out = self.ffn(out)
        logits = self.output_proj(out)
        return logits

# Model + optimizer + loss
model = TinyTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(300):
    x, y = random.choice(toy_dataset)
    x = torch.tensor([x], dtype=torch.long)     # Shape: (1, seq_len)
    y = torch.tensor([y], dtype=torch.long)     # Shape: (1,)
    
    logits = model(x)                            # Shape: (1, seq_len, vocab_size)
    pred = logits[:, -1, :]                      # Use only last token’s output

    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("Training finished!")