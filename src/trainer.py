"""
Trainer class for model training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from timm.data import create_transform
from timm.scheduler import CosineLRScheduler
from pathlib import Path
import time
from typing import Dict, Any, Optional
import numpy as np

from .config import TrainConfig
from .utils import (
    set_seed, 
    get_model_complexity, 
    get_memory_usage, 
    reset_peak_memory_stats,
    AverageMeter,
    save_checkpoint,
    save_metrics,
    EarlyStopping
)
from .dataset.ufgvc import UFGVCDataset
from .evaluator import Evaluator


class Trainer:
    """Object-oriented trainer for model training"""
    
    def __init__(self, config: TrainConfig):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
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
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Setup
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()
        
        # Calculate model complexity
        print("\n=== Model Complexity Analysis ===")
        self.model_metrics = get_model_complexity(
            self.model, 
            input_size=(1, 3, config.img_size, config.img_size),
            device=self.device
        )
        
    def _setup_data(self):
        """Setup data loaders"""
        print("\n=== Setting up datasets ===")
        
        # Get transforms from timm model
        train_transform = create_transform(
            input_size=self.config.img_size,
            is_training=True,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation=self.config.interpolation,
            mean=timm.data.IMAGENET_DEFAULT_MEAN,
            std=timm.data.IMAGENET_DEFAULT_STD,
        )
        
        val_transform = create_transform(
            input_size=self.config.img_size,
            is_training=False,
            interpolation=self.config.interpolation,
            crop_pct=self.config.crop_pct,
            mean=timm.data.IMAGENET_DEFAULT_MEAN,
            std=timm.data.IMAGENET_DEFAULT_STD,
        )
        
        # Create datasets
        train_dataset = UFGVCDataset(
            dataset_name=self.config.dataset_name,
            root=self.config.data_root,
            split='train',
            transform=train_transform,
            download=True
        )
        
        val_dataset = UFGVCDataset(
            dataset_name=self.config.dataset_name,
            root=self.config.data_root,
            split='val',
            transform=val_transform,
            download=True
        )
        
        # Update num_classes from dataset
        self.config.num_classes = len(train_dataset.classes)
        print(f"Number of classes: {self.config.num_classes}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Try to load test dataset
        try:
            test_dataset = UFGVCDataset(
                dataset_name=self.config.dataset_name,
                root=self.config.data_root,
                split='test',
                transform=val_transform,
                download=True
            )
            
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            print("Test dataset loaded successfully")
        except:
            print("No test dataset available, will use validation set for final evaluation")
            self.test_loader = self.val_loader
        
    def _setup_model(self):
        """Setup model (supports standard FC or SAD head)"""
        print(f"\n=== Creating model: {self.config.model_name} (head={self.config.head}) ===")

        if self.config.head == 'sad':
            # Create backbone without classifier so forward_features returns tokens/feature map
            backbone = timm.create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
                num_classes=0,  # remove classifier
                drop_rate=self.config.drop_rate,
                drop_path_rate=self.config.drop_path_rate
            )
            # Determine embedding dimension
            if hasattr(backbone, 'num_features'):
                emb_dim = backbone.num_features
            else:
                # Fallback: try a dummy pass to infer
                dummy = torch.randn(1, 3, self.config.img_size, self.config.img_size)
                with torch.no_grad():
                    feats = backbone.forward_features(dummy)
                if feats.dim() == 4:
                    emb_dim = feats.shape[1]
                elif feats.dim() == 3:
                    emb_dim = feats.shape[-1]
                elif feats.dim() == 2:
                    emb_dim = feats.shape[-1]
                else:
                    raise ValueError(f"Cannot infer embedding dimension from shape {feats.shape}")

            from .head.sad import SADHead, SADModel
            sad_head = SADHead(
                d=emb_dim,
                num_classes=self.config.num_classes,
                K=self.config.sad_K,
                top_m=self.config.sad_top_m
            )
            self.model = SADModel(backbone, sad_head)
        else:
            # Standard fc head
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
                num_classes=self.config.num_classes,
                drop_rate=self.config.drop_rate,
                drop_path_rate=self.config.drop_path_rate
            )

        self.model = self.model.to(self.device)

        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable:,}")
        
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        print(f"\n=== Setting up optimizer: {self.config.optimizer} ===")
        
        if self.config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Setup scheduler
        if self.config.scheduler.lower() == 'cosine':
            self.scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=self.config.num_epochs,
                lr_min=self.config.min_lr,
                warmup_t=self.config.warmup_epochs,
                warmup_lr_init=self.config.min_lr,
            )
        else:
            self.scheduler = None
        
    def _setup_criterion(self):
        """Setup loss function"""
        if self.config.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / labels.size(0)
            
            # Update meters
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(accuracy, labels.size(0))
        
        epoch_time = time.time() - start_time
        
        return {
            'loss': loss_meter.avg,
            'acc': acc_meter.avg,
            'time': epoch_time
        }
    
    def validate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                accuracy = correct / labels.size(0)
                
                loss_meter.update(loss.item(), labels.size(0))
                acc_meter.update(accuracy, labels.size(0))
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return {
            'loss': loss_meter.avg,
            'acc': acc_meter.avg,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
    
    def train(self):
        """Main training loop"""
        print("\n=== Starting Training ===")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Initial LR: {self.config.learning_rate}")
        
        # Reset peak memory stats
        reset_peak_memory_stats()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Validate
            val_metrics = None
            if (epoch + 1) % self.config.eval_freq == 0:
                val_metrics = self.validate(self.val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['learning_rate'].append(current_lr)
            
            if val_metrics is not None:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['acc'])
            
            # Print epoch summary
            print(f"\nEpoch [{epoch + 1}/{self.config.num_epochs}] - Time: {train_metrics['time']:.2f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f}")
            if val_metrics is not None:
                print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Get memory stats
            memory_stats = get_memory_usage()
            if 'gpu_memory_max_allocated_mb' in memory_stats:
                print(f"  Peak GPU Memory: {memory_stats['gpu_memory_max_allocated_mb']:.2f} MB")
            
            # Save checkpoint
            is_best = False
            if val_metrics is not None and val_metrics['acc'] > self.best_acc:
                self.best_acc = val_metrics['acc']
                self.best_epoch = epoch
                is_best = True
                print(f"  *** New best accuracy: {self.best_acc:.4f} ***")
            
            if (epoch + 1) % self.config.save_freq == 0 or is_best:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                    'best_epoch': self.best_epoch,
                    'config': self.config,
                    'history': self.history,
                    'model_metrics': self.model_metrics
                }
                
                save_path = self.output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                save_checkpoint(checkpoint, save_path, is_best=is_best)
        
        print(f"\n=== Training Completed ===")
        print(f"Best Validation Accuracy: {self.best_acc:.4f} at epoch {self.best_epoch + 1}")
        
        # Final evaluation on test set
        print("\n=== Running Final Evaluation ===")
        self.final_evaluation()
        
    def final_evaluation(self):
        """Run final evaluation on best model"""
        # Load best model
        best_checkpoint_path = self.output_dir / 'best_model.pth'
        if not best_checkpoint_path.exists():
            print("No best model found, using current model")
        else:
            checkpoint = torch.load(best_checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        
        # Create evaluator
        from .config import EvalConfig
        eval_config = EvalConfig(
            dataset_name=self.config.dataset_name,
            data_root=self.config.data_root,
            split='test',
            num_classes=self.config.num_classes,
            model_name=self.config.model_name,
            checkpoint_path=str(best_checkpoint_path),
            head=self.config.head,
            sad_K=self.config.sad_K,
            sad_top_m=self.config.sad_top_m,
            batch_size=self.config.batch_size,
            img_size=self.config.img_size,
            output_dir=str(self.output_dir / 'final_evaluation'),
            device=self.config.device,
            seed=self.config.seed
        )
        
        evaluator = Evaluator(eval_config)
        evaluator.model = self.model  # Use already loaded model
        results = evaluator.evaluate()
        
        # Save results
        final_results = {
            'training_history': self.history,
            'best_val_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'model_complexity': self.model_metrics,
            'final_evaluation': results,
            'config': vars(self.config)
        }
        
        save_metrics(final_results, self.output_dir / 'final_results.json')
        
        print(f"\n=== Final Test Accuracy: {results['accuracy']:.4f} ===")
        print(f"Results saved to {self.output_dir}")
