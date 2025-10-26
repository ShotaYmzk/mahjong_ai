"""Supervised training for Mahjong AI.

Trains the model using cross-entropy loss on expert gameplay data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class SupervisedTrainer:
    """Trainer for supervised learning of Mahjong AI."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 checkpoint_dir: str = 'outputs/checkpoints',
                 log_dir: str = 'outputs/logs'):
        """Initialize supervised trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Handle different output types
            if isinstance(outputs, tuple):
                outputs, _ = outputs  # (outputs, attention_weights)
            
            if isinstance(outputs, dict):
                # Hierarchical head
                loss = self.compute_hierarchical_loss(outputs, labels)
                logits = outputs['discard']
            else:
                # Simple discard head
                loss = nn.functional.cross_entropy(outputs, labels)
                logits = outputs
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Compute metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'acc': total_correct / total_samples
            })
        
        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                
                if isinstance(outputs, dict):
                    loss = self.compute_hierarchical_loss(outputs, labels)
                    logits = outputs['discard']
                else:
                    loss = nn.functional.cross_entropy(outputs, labels)
                    logits = outputs
                
                # Compute metrics
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / total_samples,
                    'acc': total_correct / total_samples
                })
        
        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }
        
        return metrics
    
    def compute_hierarchical_loss(self, outputs: Dict[str, torch.Tensor], 
                                  labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for hierarchical head.
        
        Args:
            outputs: Dictionary with 'action', 'claim', 'discard' logits
            labels: Ground truth tile labels
            
        Returns:
            Loss scalar
        """
        # For supervised learning, we assume action is always discard (0)
        # So we only compute discard loss
        discard_loss = nn.functional.cross_entropy(outputs['discard'], labels)
        
        return discard_loss
    
    def train(self, num_epochs: int, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(epoch)
            self.val_history.append(val_metrics)
            
            # Logging
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Save checkpoint if best
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, 'best_acc')
                logger.info(f"Saved best accuracy checkpoint: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, 'best_loss')
                logger.info(f"Saved best loss checkpoint: {val_metrics['loss']:.4f}")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch}')
        
        # Save training history
        self.save_history()
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch: int, name: str):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            name: Checkpoint name
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        checkpoint_path = self.checkpoint_dir / f'{name}.pth'
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {checkpoint['epoch']}")
    
    def save_history(self):
        """Save training history to JSON."""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved training history to {history_path}")


def create_optimizer(model: nn.Module, 
                    optimizer_name: str = 'adam',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-5) -> optim.Optimizer:
    """Create optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_name: str = 'plateau',
                    **kwargs) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler ('plateau', 'cosine', 'step', 'none')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    elif scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

