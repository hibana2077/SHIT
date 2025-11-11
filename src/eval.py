"""
Standalone evaluation script
Usage: python -m src.eval --checkpoint ./outputs/best_model.pth --dataset cotton80
"""
import argparse
import sys
from pathlib import Path

from .config import EvalConfig
from .evaluator import Evaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate a trained model on UFGVC datasets')
    
    # Required arguments
    parser.add_argument('--checkpoint-path', '--checkpoint', type=str, required=True,
                        dest='checkpoint_path',
                        help='Path to model checkpoint (required)')
    
    # Dataset settings
    parser.add_argument('--dataset-name', type=str, default='cotton80',
                        help='Dataset name (default: cotton80)')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for data (default: ./data)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-detected if not specified)')
    
    # Model settings
    parser.add_argument('--model-name', type=str, default='resnet50',
                        help='Model name from timm (default: resnet50)')
    parser.add_argument('--head', type=str, default='fc', choices=['fc','sad','onion'],
                        help='Classification head: fc (default), sad, or onion')
    parser.add_argument('--sad-K', type=int, default=16,
                        help='Number of query groups (K) for SAD head (default: 16)')
    parser.add_argument('--sad-top-m', type=int, default=8,
                        help='Top-m tokens per query for SAD head (default: 8)')
    # Onion head settings (mirrors training script)
    parser.add_argument('--onion-K', type=int, default=4,
                        help='Number of peel steps K for onion head (default: 4)')
    parser.add_argument('--onion-top-m', type=int, default=8,
                        help='Top-m tokens per step for onion head (default: 8)')
    parser.add_argument('--onion-temperature', type=float, default=0.07,
                        help='Temperature for token scoring in onion head (default: 0.07)')
    parser.add_argument('--onion-softmax', dest='onion_use_token_softmax', action='store_true', default=True,
                        help='Use softmax attention over tokens for onion head (default: True)')
    parser.add_argument('--onion-no-softmax', dest='onion_use_token_softmax', action='store_false',
                        help='Disable softmax attention over tokens for onion head')
    
    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--crop-pct', type=float, default=0.875,
                        help='Crop percentage (default: 0.875)')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Interpolation method (default: bicubic)')
    
    # TTA settings
    parser.add_argument('--tta', action='store_true',
                        help='Use Test-Time Augmentation')
    parser.add_argument('--tta-num', type=int, default=5,
                        help='Number of TTA augmentations (default: 5)')
    
    # System settings
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                        help='Output directory (default: ./eval_results)')
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save predictions')
    parser.add_argument('--no-save-predictions', dest='save_predictions', action='store_false',
                        help='Do not save predictions')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Create config from arguments
    config = EvalConfig(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        split=args.split,
        num_classes=args.num_classes if args.num_classes else 80,
        model_name=args.model_name,
        checkpoint_path=str(checkpoint_path),
        head=args.head,
        sad_K=args.sad_K,
        sad_top_m=args.sad_top_m,
        onion_K=args.onion_K,
        onion_top_m=args.onion_top_m,
        onion_temperature=args.onion_temperature,
        onion_use_token_softmax=args.onion_use_token_softmax,
        batch_size=args.batch_size,
        img_size=args.img_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions
    )
    
    # Print configuration
    print("=== Evaluation Configuration ===")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Create evaluator and run evaluation
    evaluator = Evaluator(config)
    results = evaluator.evaluate()
    
    # Run TTA if requested
    if args.tta:
        print("\n" + "=" * 50)
        tta_results = evaluator.evaluate_with_tta(num_augmentations=args.tta_num)
        results.update(tta_results)
    
    print("\n=== Evaluation Complete ===")
    print(f"Results saved to {config.output_dir}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    if args.tta:
        print(f"Test Accuracy (TTA): {results['accuracy_tta']:.4f}")


if __name__ == '__main__':
    main()
