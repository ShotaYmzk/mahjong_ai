"""Supervised training for Mahjong AI.

Trains the model using cross-entropy loss on expert gameplay data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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
                 log_dir: str = 'outputs/logs',
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """Initialize supervised trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            scheduler: Optional learning rate scheduler
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_dir = self.log_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
        self.lr_history = []
        self.start_epoch = 1
    
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
    
    def train(self, num_epochs: int, save_every: int = 10):
        """Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Starting from epoch {self.start_epoch}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
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
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Plot training curves
            self.plot_training_curves(epoch)
            
            # Save checkpoint if best
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, 'best_acc')
                logger.info(f"✅ Saved best accuracy checkpoint: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, 'best_loss')
                logger.info(f"✅ Saved best loss checkpoint: {val_metrics['loss']:.4f}")
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}')
                logger.info(f"💾 Saved checkpoint at epoch {epoch}")
            
            # Always save latest checkpoint (for resuming)
            self.save_checkpoint(epoch, 'latest')
        
        # Save final training history
        self.save_history()
        
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)
    
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
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'lr_history': self.lr_history
        }
        
        checkpoint_path = self.checkpoint_dir / f'{name}.pth'
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number to resume from
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        self.lr_history = checkpoint.get('lr_history', [])
        self.start_epoch = checkpoint['epoch'] + 1
        
        logger.info(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"   Best val acc: {self.best_val_acc:.4f}")
        logger.info(f"   Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"   Resuming from epoch {self.start_epoch}")
        
        return self.start_epoch
    
    def save_history(self):
        """Save training history to JSON."""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'learning_rate': self.lr_history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved training history to {history_path}")
    
    def plot_training_curves(self, epoch: int):
        """Plot and save training curves.
        
        Args:
            epoch: Current epoch number
        """
        if not self.train_history or not self.val_history:
            return
        
        epochs = list(range(1, len(self.train_history) + 1))
        train_losses = [h['loss'] for h in self.train_history]
        train_accs = [h['accuracy'] for h in self.train_history]
        val_losses = [h['loss'] for h in self.val_history]
        val_accs = [h['accuracy'] for h in self.val_history]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Loss
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate
        if self.lr_history:
            axes[2].plot(range(1, len(self.lr_history) + 1), self.lr_history, 'g-', linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Learning Rate', fontsize=12)
            axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = self.plot_dir / f'training_curves_epoch_{epoch:04d}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Also save as latest
        latest_path = self.plot_dir / 'training_curves_latest.png'
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        if self.lr_history:
            axes[2].plot(range(1, len(self.lr_history) + 1), self.lr_history, 'g-', linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Learning Rate', fontsize=12)
            axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(latest_path, dpi=100, bbox_inches='tight')
        plt.close()


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

