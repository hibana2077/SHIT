"""Quick sanity test for OnionPeelHead integration.
Run: python script/test_onion_head.py --model resnet18 --onion-K 4 --onion-top-m 8
"""
import argparse
import torch
import timm
from src.head.onion import OnionPeelHead, OnionPeelModel
from src.utils import make_timm_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='resnet18')
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--onion-K', type=int, default=4)
    ap.add_argument('--onion-top-m', type=int, default=8)
    ap.add_argument('--onion-temperature', type=float, default=0.07)
    ap.add_argument('--onion-no-softmax', dest='use_softmax', action='store_false')
    ap.set_defaults(use_softmax=True)
    args = ap.parse_args()

    # Resolve transforms & image size from timm model config
    train_t, val_t, data_cfg = make_timm_transforms(args.model, pretrained=args.pretrained)
    img_size = data_cfg['input_size'][-1]
    print(f"Resolved data_cfg: {data_cfg}")

    backbone = timm.create_model(args.model, pretrained=args.pretrained, num_classes=0)
    # Infer embedding dim
    if hasattr(backbone, 'num_features'):
        emb_dim = backbone.num_features
    else:
        dummy = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            feats = backbone.forward_features(dummy)
        if feats.dim() == 4:
            emb_dim = feats.shape[1]
        elif feats.dim() == 3:
            emb_dim = feats.shape[-1]
        elif feats.dim() == 2:
            emb_dim = feats.shape[-1]
        else:
            raise ValueError(f"Unsupported feature shape {feats.shape}")

    head = OnionPeelHead(
        d=emb_dim,
        num_classes=10,  # dummy classes
        K=args.onion_K,
        top_m=args.onion_top_m,
        use_token_softmax=args.use_softmax,
        temperature=args.onion_temperature,
    )
    model = OnionPeelModel(backbone, head)

    # Create dummy input matching resolved size
    x = torch.randn(2, 3, img_size, img_size)
    with torch.no_grad():
        logits = model(x)
    print(f"Logits shape: {logits.shape} (expected (B, num_classes) -> (2,10))")
    print("Forward pass succeeded.")


if __name__ == '__main__':
    main()
