"""
Evaluator class for model evaluation
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from timm.data import create_transform
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import json
from typing import Dict, Any

from .config import EvalConfig
from .utils import set_seed, get_model_complexity, get_memory_usage, load_checkpoint, save_metrics
from .dataset.ufgvc import UFGVCDataset


class Evaluator:
    """Object-oriented evaluator for model testing"""
    
    def __init__(self, config: EvalConfig):
        """
        Initialize evaluator
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.test_loader = None
        self.class_names = None
        
        # Setup
        self._setup_data()
        self._setup_model()
        
    def _setup_data(self):
        """Setup test data loader"""
        print("\n=== Setting up test dataset ===")
        
        # Get transforms from timm
        test_transform = create_transform(
            input_size=self.config.img_size,
            is_training=False,
            interpolation=self.config.interpolation,
            crop_pct=self.config.crop_pct,
            mean=timm.data.IMAGENET_DEFAULT_MEAN,
            std=timm.data.IMAGENET_DEFAULT_STD,
        )
        
        # Create dataset
        test_dataset = UFGVCDataset(
            dataset_name=self.config.dataset_name,
            root=self.config.data_root,
            split=self.config.split,
            transform=test_transform,
            download=True
        )
        
        # Update num_classes from dataset
        self.config.num_classes = len(test_dataset.classes)
        self.class_names = test_dataset.classes
        print(f"Number of classes: {self.config.num_classes}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Create data loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
    def _setup_model(self):
        """Setup and load model"""
        print(f"\n=== Loading model: {self.config.model_name} ===")
        
        # Create model
        self.model = timm.create_model(
            self.config.model_name,
            pretrained=False,
            num_classes=self.config.num_classes
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(self.config.checkpoint_path, self.model)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {num_params:,}")
        
        # Get model complexity
        print("\n=== Model Complexity Analysis ===")
        self.model_metrics = get_model_complexity(
            self.model,
            input_size=(1, 3, self.config.img_size, self.config.img_size),
            device=self.device
        )
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation
        
        Returns:
            Dictionary with evaluation results
        """
        print("\n=== Running Evaluation ===")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"\n=== Evaluation Results ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Generate classification report
        print("\n=== Classification Report ===")
        class_report = classification_report(
            all_labels, 
            all_preds,
            target_names=[str(name) for name in self.class_names],
            digits=4,
            output_dict=True
        )
        
        # Print text version
        print(classification_report(
            all_labels, 
            all_preds,
            target_names=[str(name) for name in self.class_names],
            digits=4
        ))
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Get memory stats
        memory_stats = get_memory_usage()
        
        # Compile results
        results = {
            'accuracy': float(accuracy),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'model_complexity': self.model_metrics,
            'memory_stats': memory_stats,
            'num_samples': len(all_labels),
            'num_classes': self.config.num_classes,
            'class_names': [str(name) for name in self.class_names]
        }
        
        # Save results
        save_metrics(results, self.output_dir / 'evaluation_results.json')
        
        # Save predictions if requested
        if self.config.save_predictions:
            predictions = {
                'predictions': all_preds.tolist(),
                'labels': all_labels.tolist(),
                'probabilities': all_probs.tolist()
            }
            save_metrics(predictions, self.output_dir / 'predictions.json')
            print(f"Predictions saved to {self.output_dir / 'predictions.json'}")
        
        # Save classification report as readable text
        report_text_path = self.output_dir / 'classification_report.txt'
        with open(report_text_path, 'w') as f:
            f.write(f"=== Classification Report ===\n\n")
            f.write(f"Dataset: {self.config.dataset_name}\n")
            f.write(f"Split: {self.config.split}\n")
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Checkpoint: {self.config.checkpoint_path}\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write(classification_report(
                all_labels, 
                all_preds,
                target_names=[str(name) for name in self.class_names],
                digits=4
            ))
        print(f"Classification report saved to {report_text_path}")
        
        return results
    
    def evaluate_with_tta(self, num_augmentations: int = 5) -> Dict[str, Any]:
        """
        Evaluate with Test-Time Augmentation
        
        Args:
            num_augmentations: Number of augmentations to use
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n=== Running Evaluation with TTA (n={num_augmentations}) ===")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                batch_probs = []
                
                for _ in range(num_augmentations):
                    images_aug = images.to(self.device)
                    outputs = self.model(images_aug)
                    probs = torch.softmax(outputs, dim=1)
                    batch_probs.append(probs)
                
                # Average probabilities
                avg_probs = torch.stack(batch_probs).mean(dim=0)
                _, predicted = avg_probs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"Test Accuracy (TTA): {accuracy:.4f}")
        
        return {
            'accuracy_tta': float(accuracy),
            'num_augmentations': num_augmentations
        }
