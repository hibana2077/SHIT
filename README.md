# SHIT - Smart Hierarchical Image Trainer

A modular, object-oriented training framework for fine-grained visual classification tasks using UFGVC datasets.

## Features

- ✅ **Modular & Object-Oriented Design**: Clean separation of concerns with dedicated modules for training, evaluation, and utilities
- ✅ **Fixed Random Seed**: Reproducible results with comprehensive seed fixing
- ✅ **Performance Metrics**: Comprehensive model analysis with FLOPs, MACs, and memory usage (fvcore & thop)
- ✅ **Best Model Tracking**: Automatically saves and reports the best model based on validation accuracy
- ✅ **Sklearn Classification Report**: Detailed per-class metrics and confusion matrix
- ✅ **TIMM Integration**: Uses timm models with built-in transforms for state-of-the-art architectures
- ✅ **No TQDM**: Clean epoch-by-epoch reporting without progress bars
- ✅ **Standalone Evaluation**: Separate script for model evaluation on test sets

## Project Structure

```
SHIT/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration classes
│   ├── trainer.py           # Training logic
│   ├── evaluator.py         # Evaluation logic
│   ├── utils.py             # Utility functions
│   ├── train.py             # Training entry point
│   ├── eval.py              # Evaluation entry point
│   └── dataset/
│       └── ufgvc.py         # UFGVC dataset loader
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

```bash
uv pip install -r requirements.txt
```

### Required Dependencies

- torch >= 2.0.0
- timm >= 0.9.0
- fvcore >= 0.1.5
- thop >= 0.1.1
- scikit-learn >= 1.3.0
- psutil >= 5.9.0

## Usage

### Training

Train a model using the command line:

```bash
# Basic training
python3 -m src.train --model-name resnet50 --dataset-name cotton80 --num-epochs 100

# Advanced training with custom parameters
python3 -m src.train \
    --model-name efficientnet_b0 \
    --dataset-name soybean \
    --batch-size 64 \
    --num-epochs 200 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --img-size 224 \
    --seed 42 \
    --output-dir ./outputs/soybean_exp1
```

### Available Arguments

**Dataset Settings:**

- `--dataset-name`: Dataset name (default: cotton80)
  - Available: cotton80, soybean, soy_ageing_r1-r6, soygene, soyglobal, cub_200_2011, stanford_cars, nabirds, fgvc_aircraft, food101, flowers102, oxford_pets
- `--data-root`: Data directory (default: ./data)
- `--num-classes`: Number of classes (auto-detected if not specified)

**Model Settings:**

- `--model-name`: Model from timm (default: resnet50)
- `--pretrained`: Use pretrained weights (default: True)
- `--drop-rate`: Dropout rate (default: 0.0)
- `--drop-path-rate`: Drop path rate (default: 0.0)

**Training Settings:**

- `--batch-size`: Batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 1e-5)
- `--optimizer`: Optimizer [adamw, sgd] (default: adamw)
- `--scheduler`: LR scheduler [cosine, none] (default: cosine)
- `--warmup-epochs`: Warmup epochs (default: 5)
- `--gradient-clip`: Gradient clipping (default: 1.0)
- `--label-smoothing`: Label smoothing (default: 0.1)

**System Settings:**

- `--num-workers`: Data loading workers (default: 4)
- `--seed`: Random seed (default: 42)
- `--device`: Device [cuda, cpu] (default: cuda)
- `--output-dir`: Output directory (default: ./outputs)
- `--save-freq`: Save checkpoint every N epochs (default: 10)
- `--eval-freq`: Evaluate every N epochs (default: 1)

### Evaluation

Evaluate a trained model:

```bash
# Basic evaluation
python -m src.eval --checkpoint ./outputs/best_model.pth --dataset-name cotton80

# Evaluation with Test-Time Augmentation
python -m src.eval \
    --checkpoint ./outputs/best_model.pth \
    --dataset-name cotton80 \
    --split test \
    --tta \
    --tta-num 10 \
    --batch-size 64 \
    --output-dir ./eval_results
```

### Evaluation Arguments

- `--checkpoint-path`: Path to model checkpoint (required)
- `--dataset-name`: Dataset name (default: cotton80)
- `--split`: Dataset split [train, val, test] (default: test)
- `--model-name`: Model name (default: resnet50)
- `--batch-size`: Batch size (default: 64)
- `--tta`: Enable Test-Time Augmentation
- `--tta-num`: Number of TTA augmentations (default: 5)
- `--save-predictions`: Save predictions (default: True)
- `--output-dir`: Output directory (default: ./eval_results)

## Output Files

### Training Outputs

Training produces the following files in the output directory:

```
outputs/
├── best_model.pth                    # Best model checkpoint
├── checkpoint_epoch_10.pth           # Periodic checkpoints
├── checkpoint_epoch_20.pth
├── final_results.json                # Complete training results
└── final_evaluation/
    ├── evaluation_results.json       # Test set evaluation
    ├── classification_report.txt     # Readable classification report
    └── predictions.json              # Test predictions
```

### Evaluation Outputs

```
eval_results/
├── evaluation_results.json           # Evaluation metrics
├── classification_report.txt         # Sklearn classification report
└── predictions.json                  # Predictions and probabilities
```

## Key Features Explained

### 1. Fixed Random Seed
All random operations are seeded for reproducibility:
- Python random
- NumPy random
- PyTorch (CPU and CUDA)
- CUDNN deterministic mode

### 2. Performance Metrics

The framework automatically computes:
- **FLOPs** (Floating Point Operations)
- **MACs** (Multiply-Accumulate Operations)
- **Parameters** (Total and trainable)
- **Memory Usage** (CPU and GPU peak)

Metrics are computed using both `fvcore` and `thop` for accuracy.

### 3. Best Model Selection

The trainer:
- Tracks validation accuracy every epoch
- Saves the best model automatically
- Reports best accuracy and epoch at the end
- Uses best model for final evaluation

### 4. Sklearn Classification Report

Final evaluation includes:
- Per-class precision, recall, F1-score
- Support (number of samples per class)
- Macro and weighted averages
- Confusion matrix
- Saved as both JSON and readable text

### 5. Epoch Reporting

Training reports after each epoch:
```
Epoch [1/100] - Time: 45.23s
  Train Loss: 2.3456 | Train Acc: 0.4521
  Val Loss: 2.1234 | Val Acc: 0.5123
  Learning Rate: 0.000100
  Peak GPU Memory: 3456.78 MB
  *** New best accuracy: 0.5123 ***
```

## Examples

### Train a ResNet50 on Cotton80

```bash
python -m src.train \
    --model-name resnet50 \
    --dataset-name cotton80 \
    --batch-size 32 \
    --num-epochs 100 \
    --lr 1e-4 \
    --output-dir ./outputs/cotton80_resnet50
```

### Train with the SAD classification head

Drop the original FC and use the sparse additive decoder head:

```bash
python -m src.train \
    --model-name resnet50 \
    --dataset-name cotton80 \
    --head sad \
    --sad-K 16 \
    --sad-top-m 8 \
    --batch-size 32 \
    --num-epochs 100 \
    --lr 1e-4 \
    --output-dir ./outputs/cotton80_resnet50_sad
```

### Train a Vision Transformer on CUB-200-2011

```bash
python -m src.train \
    --model-name vit_base_patch16_224 \
    --dataset-name cub_200_2011 \
    --batch-size 64 \
    --num-epochs 150 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --warmup-epochs 10 \
    --img-size 224 \
    --output-dir ./outputs/cub_vit
```

### Evaluate a SAD-head checkpoint

Make sure to pass the same head type and parameters used during training:

```bash
python -m src.eval \
    --checkpoint ./outputs/best_model.pth \
    --dataset-name cotton80 \
    --head sad \
    --sad-K 16 \
    --sad-top-m 8 \
    --split test
```

### Evaluate with Multiple Datasets

```bash
# Evaluate on test set
python -m src.eval \
    --checkpoint ./outputs/best_model.pth \
    --dataset-name cotton80 \
    --split test

# Evaluate on validation set
python -m src.eval \
    --checkpoint ./outputs/best_model.pth \
    --dataset-name cotton80 \
    --split val
```

## Available Datasets

The framework supports 16 datasets from UFGVC:

### Agricultural Datasets

- `cotton80`: Cotton classification (80 classes)
- `soybean`: Soybean classification
- `soy_ageing_r1` to `soy_ageing_r6`: Soybean ageing rounds 1-6
- `soygene`: Soybean gene classification
- `soyglobal`: Global soybean classification

### Fine-Grained Recognition

- `cub_200_2011`: Bird species (200 classes)
- `nabirds`: North American birds
- `stanford_cars`: Car models
- `fgvc_aircraft`: Aircraft variants
- `oxford_pets`: Pet breeds
- `flowers102`: Flower species
- `food101`: Food categories

## Performance Optimization

### GPU Memory Optimization

```bash
# Reduce batch size
--batch-size 16

# Use gradient checkpointing (model-dependent)
# Reduce image size
--img-size 192
```

### Speed Optimization

```bash
# Increase workers (adjust based on CPU cores)
--num-workers 8

# Reduce evaluation frequency during training
--eval-freq 5
```

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Reduce `--img-size`
- Use a smaller model

### Slow Training

- Increase `--num-workers`
- Enable `pin-memory` (default: True)
- Use mixed precision (add to config if needed)

### Poor Accuracy

- Increase `--num-epochs`
- Try different `--lr` values
- Enable data augmentation (already included via timm)
- Try different models from timm

## Citation

If you use this framework, please cite the original UFGVC datasets and timm library.

## License

This project is provided as-is for research and educational purposes.
