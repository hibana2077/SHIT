"""
Utility functions for training and evaluation
"""
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import json
from pathlib import Path
import psutil
import os


def set_seed(seed: int = 42):
    """
    Fix random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_model_complexity(model: nn.Module, input_size: tuple = (1, 3, 224, 224), device: str = 'cuda') -> Dict[str, Any]:
    """
    Calculate model complexity metrics using fvcore and thop
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        device: Device to run on
        
    Returns:
        Dictionary containing complexity metrics
    """
    metrics = {}
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_size).to(device)
    
    # Try fvcore (more accurate for modern architectures)
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        
        flop_counter = FlopCountAnalysis(model, dummy_input)
        flops = flop_counter.total()
        
        metrics['fvcore_flops'] = flops
        metrics['fvcore_gflops'] = flops / 1e9
        metrics['fvcore_params'] = parameter_count(model)[""]
        metrics['fvcore_params_m'] = parameter_count(model)[""] / 1e6
        
        print(f"FVCore - FLOPs: {metrics['fvcore_gflops']:.2f} GFLOPs")
        print(f"FVCore - Params: {metrics['fvcore_params_m']:.2f} M")
        
    except ImportError:
        print("Warning: fvcore not installed, skipping fvcore metrics")
    except Exception as e:
        print(f"Warning: fvcore calculation failed: {e}")
    
    # Try thop
    try:
        from thop import profile, clever_format
        
        # Clone model to avoid modification
        model_copy = type(model)(model.config) if hasattr(model, 'config') else model
        model_copy.load_state_dict(model.state_dict())
        model_copy = model_copy.to(device)
        
        macs, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
        
        metrics['thop_macs'] = macs
        metrics['thop_gmacs'] = macs / 1e9
        metrics['thop_params'] = params
        metrics['thop_params_m'] = params / 1e6
        
        # FLOPs ≈ 2 * MACs for most operations
        metrics['thop_flops'] = 2 * macs
        metrics['thop_gflops'] = 2 * macs / 1e9
        
        print(f"THOP - MACs: {metrics['thop_gmacs']:.2f} GMACs")
        print(f"THOP - FLOPs: {metrics['thop_gflops']:.2f} GFLOPs")
        print(f"THOP - Params: {metrics['thop_params_m']:.2f} M")
        
    except ImportError:
        print("Warning: thop not installed, skipping thop metrics")
    except Exception as e:
        print(f"Warning: thop calculation failed: {e}")
    
    return metrics


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage
    
    Returns:
        Dictionary with memory metrics in MB
    """
    memory_info = {}
    
    # CPU memory
    process = psutil.Process(os.getpid())
    memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
    
    # GPU memory
    if torch.cuda.is_available():
        memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_info['gpu_memory_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return memory_info


def reset_peak_memory_stats():
    """Reset peak memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_metrics(metrics: Dict[str, Any], save_path: str):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    metrics = convert_to_serializable(metrics)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {save_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """
    Load checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"  Best Acc: {checkpoint['best_acc']:.4f}")
    
    return checkpoint


def save_checkpoint(state: Dict[str, Any], save_path: str, is_best: bool = False):
    """
    Save checkpoint
    
    Args:
        state: State dictionary to save
        save_path: Path to save checkpoint
        is_best: Whether this is the best model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path}")
    
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
