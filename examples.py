#!/usr/bin/env python3
"""
Quick Start Example
Demonstrates basic usage of the training framework
"""

def example_train():
    """Example: Train a ResNet50 on Cotton80 dataset"""
    print("Example: Training ResNet50 on Cotton80")
    print("=" * 50)
    print("Command:")
    print("""
python -m src.train \\
    --model-name resnet50 \\
    --dataset-name cotton80 \\
    --batch-size 32 \\
    --num-epochs 10 \\
    --lr 1e-4 \\
    --output-dir ./outputs/example_train
    """)
    print("=" * 50)


def example_eval():
    """Example: Evaluate a trained model"""
    print("\nExample: Evaluating trained model")
    print("=" * 50)
    print("Command:")
    print("""
python -m src.eval \\
    --checkpoint ./outputs/example_train/best_model.pth \\
    --dataset-name cotton80 \\
    --split test \\
    --batch-size 64 \\
    --output-dir ./eval_results/example
    """)
    print("=" * 50)


def example_programmatic():
    """Example: Programmatic usage"""
    print("\nExample: Programmatic Usage")
    print("=" * 50)
    
    code = """
from src.config import TrainConfig
from src.trainer import Trainer

# Create configuration
config = TrainConfig(
    dataset_name='cotton80',
    model_name='resnet50',
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4,
    output_dir='./outputs/programmatic'
)

# Create trainer and train
trainer = Trainer(config)
trainer.train()
    """
    print(code)
    print("=" * 50)


def main():
    """Show all examples"""
    print("\n" + "=" * 70)
    print("SHIT Framework - Quick Start Examples")
    print("=" * 70)
    
    example_train()
    example_eval()
    example_programmatic()
    
    print("\n" + "=" * 70)
    print("Available Datasets:")
    print("=" * 70)
    datasets = [
        "cotton80", "soybean", "soy_ageing_r1", "soy_ageing_r3",
        "soy_ageing_r4", "soy_ageing_r5", "soy_ageing_r6",
        "cub_200_2011", "soygene", "soyglobal", "stanford_cars",
        "nabirds", "fgvc_aircraft", "food101", "flowers102", "oxford_pets"
    ]
    for i, ds in enumerate(datasets, 1):
        print(f"{i:2d}. {ds}")
    
    print("\n" + "=" * 70)
    print("Popular Models (from timm):")
    print("=" * 70)
    models = [
        "resnet50", "resnet101", "efficientnet_b0", "efficientnet_b3",
        "vit_base_patch16_224", "vit_large_patch16_224",
        "convnext_base", "swin_base_patch4_window7_224"
    ]
    for i, model in enumerate(models, 1):
        print(f"{i:2d}. {model}")
    
    print("\n" + "=" * 70)
    print("For more information, see README.md")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
