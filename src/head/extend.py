"""Extension point for custom classification heads.

To add a custom head:
1. Create a class deriving from BaseTokenHead implementing forward(E: Tensor[B,T,D]) -> logits[B,C].
2. Provide required constructor signature: __init__(self, d: int, num_classes: int, **kwargs)
3. Run training/eval with:
   --head custom \
   --custom-head-module src.head.my_head \
   --custom-head-class MyHead

This file also contains a MinimalExampleHead showing the API.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class BaseTokenHead(nn.Module):
    """Abstract token-based head.
    Expected input: token embeddings (B, T, D).
    Implementations can pool, attend, etc. Must return logits (B, C).
    """
    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.d = d
        self.num_classes = num_classes

    def forward(self, E: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

class MinimalExampleHead(BaseTokenHead):
    """A simple attention + projection example head.
    1. Learn a query vector q.
    2. Compute attention over tokens.
    3. Weighted sum -> pooled feature -> linear classifier.
    """
    def __init__(self, d: int, num_classes: int):
        super().__init__(d, num_classes)
        self.query = nn.Parameter(torch.randn(d))
        self.classifier = nn.Linear(d, num_classes)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: (B, T, D)
        q = self.query / (self.query.norm() + 1e-6)
        att = (E @ q)  # (B, T)
        att = torch.softmax(att, dim=1)
        pooled = (E * att.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return self.classifier(pooled)

__all__ = ["BaseTokenHead", "MinimalExampleHead"]
