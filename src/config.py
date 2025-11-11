"""
Configuration module for training and evaluation
"""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class TrainConfig:
    """Training configuration"""
    
    # Dataset settings
    dataset_name: str = "cotton80"
    data_root: str = "./data"
    num_classes: int = 80
    
    # Model settings
    model_name: str = "resnet50"
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Classification head
    head: str = "fc"  # choices: ['fc', 'sad']
    sad_K: int = 16
    sad_top_m: int = 8
    
    # Training settings
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Data augmentation (handled by timm)
    img_size: int = 224
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    
    # Training behavior
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    # System settings
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    device: str = "cuda"
    
    # Checkpoint settings
    output_dir: str = "./outputs"
    save_freq: int = 10
    eval_freq: int = 1
    
    # Evaluation settings
    eval_metrics: bool = True
    
    # Logging
    log_interval: int = 50
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    
    # Dataset settings
    dataset_name: str = "cotton80"
    data_root: str = "./data"
    split: str = "test"
    num_classes: int = 80
    
    # Model settings
    model_name: str = "resnet50"
    checkpoint_path: str = ""
    # Classification head
    head: str = "fc"  # choices: ['fc', 'sad']
    sad_K: int = 16
    sad_top_m: int = 8
    
    # Evaluation settings
    batch_size: int = 64
    img_size: int = 224
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    
    # System settings
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    device: str = "cuda"
    
    # Output settings
    output_dir: str = "./eval_results"
    save_predictions: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path must be specified for evaluation")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
