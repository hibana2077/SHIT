"""Quick sanity test for timm data config & forward pass.
Run: python script/test_forward.py --model <model_name>
"""
import argparse
from PIL import Image
import numpy as np
import torch
import timm
from src.utils import make_timm_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='resnet50')
    ap.add_argument('--pretrained', action='store_true')
    args = ap.parse_args()

    train_t, val_t, data_cfg = make_timm_transforms(args.model, pretrained=args.pretrained)
    img_size = data_cfg['input_size'][-1]
    print(f"Resolved data_cfg: {data_cfg}")

    # Create random PIL image with correct size to simulate raw input prior to transform
    raw = Image.fromarray((np.random.rand(img_size, img_size, 3) * 255).astype('uint8'))
    tensor = val_t(raw).unsqueeze(0)  # (1,3,H,W)
    print(f"Transformed tensor shape: {tensor.shape}")

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
