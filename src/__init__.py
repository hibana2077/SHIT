"""
SHIT - Smart Hierarchical Image Trainer
A modular training framework for fine-grained visual classification
"""

__version__ = "1.0.0"

from .config import TrainConfig, EvalConfig
from .trainer import Trainer
from .evaluator import Evaluator
from .utils import set_seed, get_model_complexity, get_memory_usage

__all__ = [
    'TrainConfig',
    'EvalConfig',
    'Trainer',
    'Evaluator',
    'set_seed',
    'get_model_complexity',
    'get_memory_usage'
]
