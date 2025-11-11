import torch
import torch.nn as nn

class SADHead(nn.Module):
    """
    Sparse Additive Decoder (SAD) classification head.

    Accepts a sequence of token embeddings E with shape (B, T, D) and performs:
      1. Query scoring via additive signed aggregation across feature dimension.
      2. Top-m token selection per query.
      3. Grouped additive classification using learned sign weights.

    The design uses only additions and sign flips (no multiplications) aside from
    bias additions, making it potentially more efficient and interpretable.
    """

    def __init__(self, d: int, num_classes: int, K: int = 16, top_m: int = 8):
        super().__init__()
        self.d, self.K, self.top_m = d, K, top_m
        self.num_classes = num_classes
        self.group = (num_classes + K - 1) // K  # ceil for grouped mapping
        # Real-valued parameters whose signs are used in forward (STE in backward)
        self.q_weight = nn.Parameter(torch.randn(K, d))                  # Query sign weights
        self.c_weight = nn.Parameter(torch.randn(K, self.group, d))      # Class sign weights per query
        self.q_bias = nn.Parameter(torch.zeros(K))
        self.cls_bias = nn.Parameter(torch.zeros(K, self.group))

    @staticmethod
    def _sign(x: torch.Tensor) -> torch.Tensor:
        """Binary sign projection with values in {-1, +1}. Gradient passes through (STE)."""
        return (x >= 0).to(x.dtype) * 2 - 1

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            E: Token embeddings of shape (B, T, D)
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        B, T, D = E.shape
        K, G = self.K, self.group

        # A) Query additive scoring
        S_q = self._sign(self.q_weight)                 # (K, D)
        E4 = E[:, :, None, :].expand(B, T, K, D)        # (B, T, K, D)
        Sq4 = S_q[None, None, :, :].expand(B, T, K, D)  # (B, T, K, D)
        signed = torch.where(Sq4 > 0, E4, -E4)          # sign flip instead of mul
        scores = signed.sum(dim=-1) + self.q_bias[None, None, :]  # (B, T, K)

        # Top-m token selection per query
        m = min(self.top_m, T)
        top_idx = scores.topk(m, dim=1).indices         # (B, m, K)
        idx_exp = top_idx[..., None].expand(B, m, K, D)
        E_exp = E[:, :, None, :].expand(B, T, K, D)
        top_tokens = torch.gather(E_exp, 1, idx_exp)    # (B, m, K, D)
        g = top_tokens.sum(dim=1)                       # (B, K, D)

        # B) Grouped additive classification
        S_c = self._sign(self.c_weight)                 # (K, G, D)
        Sc4 = S_c[None, :, :, :].expand(B, K, G, D)     # (B, K, G, D)
        g4  = g[:, :, None, :].expand(B, K, G, D)       # (B, K, G, D)
        signed_g = torch.where(Sc4 > 0, g4, -g4)
        logits_group = signed_g.sum(dim=-1) + self.cls_bias[None, :, :, ]  # (B, K, G)
        logits = logits_group.reshape(B, K * G)[:, :self.num_classes]      # (B, N)
        return logits


class SADModel(nn.Module):
    """
    Wrapper model integrating a timm backbone that produces token features with a SADHead.

    It adapts different backbone output shapes:
      - (B, C, H, W) -> flatten spatial tokens
      - (B, N, D)    -> already tokenized
      - (B, D)       -> treat as single token (T=1)
    """
    def __init__(self, backbone: nn.Module, head: SADHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if isinstance(feats, (list, tuple)):
            # Some models return (x, aux); take the first
            feats = feats[0]
        if feats.dim() == 4:
            # Convolutional features (B, C, H, W) -> (B, T, D)
            B, C, H, W = feats.shape
            tokens = feats.view(B, C, H * W).permute(0, 2, 1)
        elif feats.dim() == 3:
            # Already (B, N, D)
            tokens = feats
        elif feats.dim() == 2:
            # (B, D) -> single token
            tokens = feats.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported feature shape {feats.shape}")
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.extract_tokens(x)
        return self.head(tokens)
