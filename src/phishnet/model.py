"""Attention-based feedforward classifier for tokenized email sequences."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AttentionClassifier(nn.Module):
    """Embedding + self-attention + linear classifier.

    This is a feedforward network with a multi-head self-attention block
    used for both classification and interpretability (the attention
    weights over token positions), not a full transformer encoder stack.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 16, n_heads: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = nn.MultiheadAttention(embed_dim, n_heads)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x.long())
        embedded = embedded.permute(1, 0, 2)  # (seq, batch, embed) for MultiheadAttention
        attn_output, attn_weights = self.attention(embedded, embedded, embedded)
        pooled = attn_output.mean(dim=0)
        logits = self.fc(pooled)
        return torch.sigmoid(logits).squeeze(-1), attn_weights


def train_model(
    model: AttentionClassifier,
    X_train: np.ndarray,
    y_train: list[int],
    epochs: int = 30,
    lr: float = 3e-3,
    seed: int = 42,
) -> list[float]:
    """Train in place and return the per-epoch loss history."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    X = torch.tensor(X_train)
    y = torch.tensor(y_train, dtype=torch.float32)

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds, _ = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


@torch.no_grad()
def predict_proba(model: AttentionClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    preds, _ = model(torch.tensor(X))
    return preds.numpy()


@torch.no_grad()
def attention_weights(model: AttentionClassifier, X: np.ndarray) -> np.ndarray:
    """Return the (batch, seq, seq) self-attention matrix, averaged over heads."""
    model.eval()
    _, attn = model(torch.tensor(X))
    return attn.numpy()
