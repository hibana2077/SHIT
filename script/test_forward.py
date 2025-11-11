"""Quick sanity test for timm data config & forward pass.
Run: python script/test_forward.py --model <model_name>
"""
import argparse
import json
from PIL import Image
import numpy as np
import torch
import timm
from src.utils import make_timm_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='resnet50')
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--custom-head-module', type=str, default=None)
    ap.add_argument('--custom-head-class', type=str, default=None)
    ap.add_argument('--custom-head-kwargs', type=str, default=None,
                    help='JSON string for custom head kwargs, e.g. {"n":4,"top_k":1}')
    args = ap.parse_args()

    train_t, val_t, data_cfg = make_timm_transforms(args.model, pretrained=args.pretrained)
    img_size = data_cfg['input_size'][-1]
    print(f"Resolved data_cfg: {data_cfg}")

    # Create random PIL image with correct size to simulate raw input prior to transform
    raw = Image.fromarray((np.random.rand(img_size, img_size, 3) * 255).astype('uint8'))
    tensor = val_t(raw).unsqueeze(0)  # (1,3,H,W)
    print(f"Transformed tensor shape: {tensor.shape}")

    if args.custom_head_module and args.custom_head_class:
        # Build backbone + wrapper like trainer
        backbone = timm.create_model(args.model, pretrained=args.pretrained, num_classes=0)
        if hasattr(backbone, 'num_features'):
            d = backbone.num_features
        else:
            with torch.no_grad():
                tmp = backbone.forward_features(tensor)
            if tmp.dim() == 4:
                d = tmp.shape[1]
            elif tmp.dim() in (2,3):
                d = tmp.shape[-1]
            else:
                raise RuntimeError(f"Cannot infer embedding dim from {tmp.shape}")
        from importlib import import_module
        mod = import_module(args.custom_head_module)
        HeadCls = getattr(mod, args.custom_head_class)
        head_kwargs = json.loads(args.custom_head_kwargs) if args.custom_head_kwargs else {}
        head = HeadCls(d=d, num_classes=10, **head_kwargs)  # dummy num_classes
        class _Wrapper(torch.nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head
            def forward(self, x):
                feats = self.backbone.forward_features(x)
                if feats.dim() == 4:
                    B,C,H,W = feats.shape
                    tokens = feats.view(B,C,H*W).permute(0,2,1)
                elif feats.dim() == 3:
                    tokens = feats
                elif feats.dim() == 2:
                    tokens = feats.unsqueeze(1)
                else:
                    raise ValueError(f"Unsupported feature shape {feats.shape}")
                return self.head(tokens)
        model = _Wrapper(backbone, head)
        model.train()
        out = model(tensor)
        print(f"Custom head train-mode output shape: {out.shape}")
        model.eval()
        out_eval = model(tensor)
        print(f"Custom head eval-mode output shape: {out_eval.shape}")
    else:
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=0)
        with torch.no_grad():
            feats = model.forward_features(tensor)
        if feats.dim() == 4:
            print(f"Feature map shape: {feats.shape}")
        elif feats.dim() == 3:
            print(f"Token shape: {feats.shape}")
        else:
            print(f"Feature vector shape: {feats.shape}")
        print("Forward features completed without size assertion errors.")

if __name__ == '__main__':
    main()
