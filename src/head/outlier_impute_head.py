"""Outlier imputation head using timm features.

This head operates on token embeddings E of shape (B, T, D):
- During training only, it duplicates features n times, detects outlier token
  positions per sample (by largest deviation from the token mean), and performs
  gradual imputation of those tokens toward the mean using linspace steps.
- At eval/test time, it behaves like a standard pooling + linear classifier.

Usage with this repo's custom head mechanism:
  --head custom \
  --custom-head-module src.head.outlier_impute_head \
  --custom-head-class OutlierImputeHead \
  --custom-head-kwargs '{"n": 4, "top_k": 1}'

Constructor signature follows BaseTokenHead:
  OutlierImputeHead(d: int, num_classes: int, n: int = 4, top_k: int = 1,
                    training_only: bool = True, pool: str = "mean")
"""
from __future__ import annotations
import torch
import torch.nn as nn


class OutlierImputeHead(nn.Module):
    """Token head with training-time outlier imputation/duplication.

    Args:
        d: Token embedding dimension
        num_classes: Number of output classes
        n: Number of duplicated versions per sample during training (linspace steps)
        top_k: Number of token positions to treat as outliers per sample
        training_only: Enable the mechanism only during training
        pool: Pooling method over tokens, one of {"mean"}
    """

    def __init__(
        self,
        d: int,
        num_classes: int,
        n: int = 4,
        top_k: int = 1,
        training_only: bool = True,
        pool: str = "mean",
    ):
        super().__init__()
        assert n >= 1, "n must be >= 1"
        assert top_k >= 1, "top_k must be >= 1"
        self.d = d
        self.num_classes = num_classes
        self.n = int(n)
        self.top_k = int(top_k)
        self.training_only = bool(training_only)
        self.pool = pool

        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, num_classes)
        )

    def _pool(self, E: torch.Tensor) -> torch.Tensor:
        # E: (B, T, D) -> (B, D)
        if self.pool == "mean":
            return E.mean(dim=1)
        else:
            # fallback to mean if unknown
            return E.mean(dim=1)

    @torch.no_grad()
    def _select_outlier_mask(self, E: torch.Tensor, top_k: int) -> torch.Tensor:
        """Compute boolean mask of shape (B, T) marking outlier token positions.

        Outliers are positions with largest L2 deviation from per-sample token mean.
        """
        # E: (B, T, D)
        mu = E.mean(dim=1, keepdim=True)  # (B, 1, D)
        dev = (E - mu).pow(2).sum(dim=-1).sqrt()  # (B, T)
        # top_k indices per row
        _, idx = torch.topk(dev, k=min(top_k, E.shape[1]), dim=1, largest=True)
        mask = torch.zeros_like(dev, dtype=torch.bool)
        # scatter to boolean mask
        mask.scatter_(1, idx, True)
        return mask  # (B, T)

    def _duplicate_and_impute(self, E: torch.Tensor) -> torch.Tensor:
        """Duplicate tokens and impute outlier positions toward the mean.

        Returns E_out of shape (B*n, T, D).
        """
        B, T, D = E.shape
        device = E.device
        dtype = E.dtype

        # Identify outlier token positions per sample
        mask = self._select_outlier_mask(E, self.top_k)  # (B, T)
        mask = mask.unsqueeze(-1).to(dtype)  # (B, T, 1) as float 0/1

        mu = E.mean(dim=1, keepdim=True)  # (B, 1, D)
        diff = E - mu  # (B, T, D)

        # Steps from 0 (original) to 1 (fully imputed to mean)
        alphas = torch.linspace(0.0, 1.0, steps=self.n, device=device, dtype=dtype)

        # For each alpha, move only outlier tokens toward the mean
        E_list = []
        for a in alphas:
            # E_a = E - mask * diff * a  -> outlier tokens: E - (E-mu)*a = mu + (1-a)*diff
            E_a = E - mask * diff * a
            E_list.append(E_a)

        E_out = torch.stack(E_list, dim=1)  # (B, n, T, D)
        E_out = E_out.reshape(B * self.n, T, D)
        return E_out

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: (B, T, D)
        if self.training and self.training_only and self.n > 1:
            E_aug = self._duplicate_and_impute(E)
            Z = self._pool(E_aug)  # (B*n, D)
            logits = self.classifier(Z)  # (B*n, C)
            return logits
        else:
            Z = self._pool(E)  # (B, D)
            logits = self.classifier(Z)  # (B, C)
            return logits


__all__ = ["OutlierImputeHead"]
