# onion_peel_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OnionPeelHead(nn.Module):
    """
    OnionPeel Orthogonal Residual Head (OP-Head)

    Step-wise process (K steps total):
      1) Define peel vector u_k = norm((sigmoid(m_k)) * v_k) using learnable direction v_k and channel mask m_k
      2) Score tokens with u_k, select top-m tokens
      3) Project these tokens onto u_k, aggregate into step feature z_k, generate logits_k via step-specific classifier
      4) Remove u_k direction component from all tokens (E <- E - beta_k * Proj_{u_k}(E))
      5) Accumulate logits (or use gated weighting)
    Returns: logits: (B, num_classes)
    """

    def __init__(
        self, d: int, num_classes: int, K: int = 4, top_m: int = 8,
        use_token_softmax: bool = True, temperature: float = 0.07, eps: float = 1e-6
    ):
        super().__init__()
        self.d, self.K, self.top_m = d, K, top_m
        self.use_token_softmax = use_token_softmax
        self.temperature = temperature
        self.eps = eps

        # Each step has a "direction" and "channel mask" to learn the subspace to peel/use
        self.v = nn.Parameter(torch.randn(K, d) * 0.02)          # learnable directions v_k
        self.m_logits = nn.Parameter(torch.zeros(K, d))          # channel masks logits -> sigmoid

        # Each step has a linear classifier (shared/step-wise choice; here using step-wise to preserve per-step semantics)
        self.cls_W = nn.Parameter(torch.randn(K, num_classes, d) * (1.0 / (d ** 0.5)))
        self.cls_b = nn.Parameter(torch.zeros(K, num_classes))

        # Peeling strength beta_k and step weights alpha_k (if you want to gate each step's contribution)
        self.beta = nn.Parameter(torch.ones(K))   # how much residual to peel at each step
        self.alpha = nn.Parameter(torch.ones(K))  # per-step logit scaling before sum

        # (Optional) Halt gate: if you want ACT-style early stopping, use these logits to estimate halt probability
        self.halt_logits = nn.Parameter(torch.full((K,), -2.0))  # initially conservative

    def _unit_u(self, k: int) -> torch.Tensor:
        # u_k = normalize(sigmoid(m_k) * v_k)
        mk = torch.sigmoid(self.m_logits[k])                     # (D,)
        vk = self.v[k] * mk                                      # (D,)
        denom = torch.norm(vk, p=2) + self.eps
        return vk / denom

    def _token_scores(self, E: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # E: (B, T, D), u: (D,) -> scores (B, T)
        # Score using scaled dot product
        return (E @ u) / self.temperature

    def _project_onto(self, X: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # X: (..., D), u: (D,) -> Proj_u(X) = (X·u) u
        coeff = (X @ u)                                          # (...,)
        return coeff.unsqueeze(-1) * u                           # (..., D)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E: (B, T, D) tokens
        Returns:
            logits: (B, num_classes)
        """
        B, T, D = E.shape
        assert D == self.d, f"Expect D={self.d}, got {D}"

        logits_sum = 0.0
        tokens = E

        for k in range(self.K):
            u = self._unit_u(k)                                  # (D,)

            # 1) Token scoring + select top-m (or softmax weights)
            scores = self._token_scores(tokens, u)               # (B, T)
            if self.use_token_softmax:
                att = F.softmax(scores, dim=1)                   # (B, T)
                # Expectation weighting (weak sparsity), also do a top-m strict sparsity version for projection
                top_m = min(self.top_m, T)
                top_idx = scores.topk(top_m, dim=1).indices      # (B, m)
            else:
                top_m = min(self.top_m, T)
                att = None
                top_idx = scores.topk(top_m, dim=1).indices

            # 2) Project top-m tokens onto u and aggregate into step feature z_k
            #    (If using softmax, can also do an expectation version z_k_soft)
            idx_exp = top_idx.unsqueeze(-1).expand(B, top_m, D)  # (B, m, D)
            top_tokens = torch.gather(tokens, 1, idx_exp)        # (B, m, D)
            proj_top = self._project_onto(top_tokens, u)         # (B, m, D)
            z_k = proj_top.sum(dim=1)                            # (B, D)
            if att is not None:
                proj_all = self._project_onto(tokens, u)         # (B, T, D)
                z_k_soft = (proj_all * att.unsqueeze(-1)).sum(dim=1)
                # Compromise: average both, having sparse focus while preserving smooth gradients
                z_k = 0.5 * (z_k + z_k_soft)

            # 3) Generate logits_k through k-th step classifier and accumulate with weighting
            Wk = self.cls_W[k]                                    # (C, D)
            bk = self.cls_b[k]                                    # (C,)
            logits_k = z_k @ Wk.t() + bk                          # (B, C)
            logits_sum = logits_sum + self.alpha[k] * logits_k

            # 4) Remove explained components from all tokens (residualize)
            proj_tokens = self._project_onto(tokens, u)          # (B, T, D)
            tokens = tokens - self.beta[k] * proj_tokens

            # 5) (Optional) If you want ACT-style early stopping, use halt prob to decide whether to stop early
            #    During training, can add ponder cost to halt prob; here just keeping the interface.
            # if torch.sigmoid(self.halt_logits[k]) > 0.5: break

        return logits_sum


class OnionPeelModel(nn.Module):
    """
    Wrapper similar to sad.py:
      - (B, C, H, W) -> flatten to (B, T, D)
      - (B, N, D)    -> as is
      - (B, D)       -> add T=1 dimension
    """
    def __init__(self, backbone: nn.Module, head: OnionPeelHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if feats.dim() == 4:
            B, C, H, W = feats.shape
            tokens = feats.view(B, C, H * W).permute(0, 2, 1)    # (B, T, D=C)
        elif feats.dim() == 3:
            tokens = feats
        elif feats.dim() == 2:
            tokens = feats.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported feature shape {feats.shape}")
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.extract_tokens(x)
        return self.head(tokens)