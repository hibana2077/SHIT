# Outlier Imputation Token Head

A training-time augmentation and inference-time classifier head for token embeddings that mitigates outlier tokens by gradually imputing them toward the per-sample token mean. Designed to sit on top of a backbone that outputs token embeddings E of shape (B, T, D), such as ViT/timm models.

- Module: `src.head.outlier_impute_head`
- Class: `OutlierImputeHead`
- Constructor: `OutlierImputeHead(d, num_classes, n=4, top_k=1, training_only=True, pool="mean")`

## Motivation and intuition

Token-level representations often contain a few tokens with unusually large deviations from the sample’s typical feature magnitude or direction (e.g., clutter, occlusions, or spurious patches). These outlier tokens can degrade pooled representations.

This head identifies the `top_k` most deviating token positions per sample and creates `n` progressively imputed versions of the tokens where the selected outliers are moved along a straight line toward the per-sample token mean. During training this acts as a robustness-oriented augmentation; at inference the head reduces to standard pooling plus a small MLP classifier.

## Data shapes and notation

- Input tokens: E ∈ R^{B×T×D}
  - B: batch size
  - T: number of tokens per sample
  - D: token embedding dimension
- Output logits (training): R^{(B·n)×C}
- Output logits (eval): R^{B×C}
- C = `num_classes`

We use the following per-sample statistics:

- Token mean: μ = mean over tokens, shape (B, 1, D)
- Deviation per token: dev = ||E - μ||₂ along D, shape (B, T)

## Behavior by mode

### Training mode (if `training_only=True` and `n>1`)

1. Detect outlier positions per sample by taking the `top_k` tokens with the largest L2 deviation from the sample token mean.
2. Generate `n` equally spaced interpolation coefficients α ∈ linspace(0, 1, n).
3. For each α, impute only the outlier tokens toward μ:
   - Let M be the boolean mask for outliers, broadcast to shape (B, T, 1).
   - Define diff = E - μ.
   - Imputed tokens for step α:  E(α) = E − M ⊙ diff · α = μ + (1−α)·diff on masked positions; non-masked tokens remain unchanged.
4. Stack the `n` versions along batch: E_aug ∈ R^{(B·n)×T×D}.
5. Pool tokens to a single vector per sample and classify with an MLP.

The labels are implicitly duplicated `n` times. Most training loops will average the loss across the expanded batch.

### Evaluation mode (or `training_only=False` or `n==1`)

- No outlier detection or duplication is performed.
- Pool the input tokens once and classify.

## Equations

- Outlier selection (per sample):
  $$\mu = \frac{1}{T} \sum_{t=1}^T E_{:,t,:}, \quad dev_t = \lVert E_{:,t,:} - \mu \rVert_2.$$
  Select indices of the top-k values of dev.

- Imputation steps:
  $$\alpha_i \in \{\text{linspace}(0,1,n)\},\quad diff = E - \mu,$$
  $$E^{(i)} = E - M \odot diff \cdot \alpha_i,$$
  where M is a binary mask broadcastable to (B, T, D), with ones at outlier token positions.

- Pooling (mean):
  $$Z = \frac{1}{T} \sum_{t=1}^T E^{(i)}_{:,t,:}.$$

- Classifier: two-layer MLP with GELU, `D -> D/2 -> C`.

## Pseudocode

```python
# E: (B, T, D)
mu = E.mean(dim=1, keepdim=True)              # (B, 1, D)
dev = (E - mu).pow(2).sum(-1).sqrt()          # (B, T)
idx = topk(dev, k=min(top_k, T))               # (B, top_k)
M = scatter_bool_mask(idx, shape=(B, T))       # (B, T)
M = M.unsqueeze(-1).float()                    # (B, T, 1)
diff = E - mu                                  # (B, T, D)

alphas = linspace(0, 1, steps=n)
E_list = []
for a in alphas:
    E_a = E - M * diff * a                     # move only masked tokens
    E_list.append(E_a)

E_aug = stack(E_list, dim=1).reshape(B*n, T, D)
Z = E_aug.mean(dim=1)                          # pool -> (B*n, D)
logits = classifier(Z)                         # (B*n, C)
```

## API contract

- Inputs
  - `E`: float Tensor of shape (B, T, D)
  - `self.training`: if True enables training-time augmentation when `training_only` and `n > 1`
- Outputs
  - Logits of shape (B·n, C) during training augmentation, else (B, C)
- Errors
  - Asserts `n >= 1` and `top_k >= 1`

## Hyperparameters

- `n` (int, default 4): number of imputation steps. Also the duplication factor for the effective batch size in training. `n=1` disables augmentation but keeps the same code path.
- `top_k` (int, default 1): how many tokens per sample to treat as outliers.
- `training_only` (bool, default True): if False, duplication/imputation also runs at eval; normally keep True.
- `pool` (str, default "mean"): token pooling method; unsupported values fall back to mean.

## Complexity and memory

- Outlier detection: `topk` per sample costs O(T log k) (typically small `k`).
- Imputation: O(B·T·D·n) elementwise ops due to `n` duplicates.
- Memory: approximately `n`× the token memory during training.

## Edge cases and notes

- `n == 1`: alphas = [0]; output equals the original tokens (no change); behaves like a standard pooling+MLP head.
- `top_k >= T`: mask will mark all tokens; the model then imputes all tokens toward μ along α.
- Mixed precision: operations are differentiable; the selection mask is computed under `torch.no_grad()` to avoid gradient through discrete `topk`.
- Gradients: imputation itself is differentiable w.r.t. E; only the mask creation is non-differentiable (by design).
- Pooling: if an unknown `pool` is provided, the implementation falls back to mean pooling silently.

## Practical guidance

- Start with `n ∈ {3,4,5}` and `top_k ∈ {1,2}`. Larger `n` increases compute and memory linearly.
- Use `training_only=True` to keep inference cost equal to a standard head.
- Combine with strong backbone regularization; this head specifically targets token-level outliers.

## Usage examples

### CLI with this repository

```bash
--head custom \
--custom-head-module src.head.outlier_impute_head \
--custom-head-class OutlierImputeHead \
--custom-head-kwargs '{"n": 4, "top_k": 1}'
```

### Python

```python
from src.head.outlier_impute_head import OutlierImputeHead

head = OutlierImputeHead(d=768, num_classes=1000, n=4, top_k=1, training_only=True)
head.train()  # training-time duplication/imputation enabled
logits = head(E)  # E: (B, T, 768) -> logits: (B*n, 1000)

head.eval()   # inference-time: standard pooling+MLP
logits = head(E)  # (B, 1000)
```

## Implementation reference

See `src/head/outlier_impute_head.py` for the full implementation.
