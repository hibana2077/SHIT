"""
Main training script
Usage: python -m src.train --model resnet50 --dataset cotton80 --epochs 100
"""
import argparse
import sys
from pathlib import Path

from .config import TrainConfig
from .trainer import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a model on UFGVC datasets')
    
    # Dataset settings
    parser.add_argument('--dataset-name', type=str, default='cotton80',
                        help='Dataset name (default: cotton80)')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for data (default: ./data)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-detected if not specified)')
    
    # Model settings
    parser.add_argument('--model-name', type=str, default='resnet50',
                        help='Model name from timm (default: resnet50)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Do not use pretrained weights')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--drop-path-rate', type=float, default=0.0,
                        help='Drop path rate (default: 0.0)')
    # Classification head
    parser.add_argument('--head', type=str, default='fc', choices=['fc', 'custom'],
                        help='Classification head: fc or custom')
    parser.add_argument('--custom-head-module', type=str, default=None,
                        help='Python module path for custom head, e.g., src.head.my_head')
    parser.add_argument('--custom-head-class', type=str, default=None,
                        help='Class name for custom head, e.g., MyHead')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4,
                        dest='learning_rate', help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'],
                        help='Optimizer (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'none'],
                        help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warmup epochs (default: 5)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')
    
    # Data augmentation
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--crop-pct', type=float, default=0.875,
                        help='Crop percentage (default: 0.875)')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Interpolation method (default: bicubic)')
    
    # Training behavior
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping (default: 1.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='Mixup alpha (default: 0.0)')
    parser.add_argument('--cutmix-alpha', type=float, default=0.0,
                        help='Cutmix alpha (default: 0.0)')
    
    # System settings
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    # Checkpoint settings
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory (default: ./outputs)')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--eval-freq', type=int, default=1,
                        help='Evaluate every N epochs (default: 1)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Create config from arguments
    config = TrainConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        num_classes=args.num_classes if args.num_classes else 80,
        model_name=args.model_name,
        pretrained=args.pretrained,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        head=args.head,
        custom_head_module=args.custom_head_module,
        custom_head_class=args.custom_head_class,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        img_size=args.img_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        gradient_clip=args.gradient_clip,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq
    )
    
    # Print configuration
    print("=== Training Configuration ===")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
