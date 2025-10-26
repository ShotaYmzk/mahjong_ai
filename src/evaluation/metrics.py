"""Evaluation metrics for Mahjong AI.

Implements various metrics:
- Accuracy (exact match)
- Top-k accuracy
- Success Probability (SP)
- Hit Rate (HR)
- Cross-entropy loss
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self, num_classes: int = 34):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes (tile types)
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
              loss: Optional[float] = None):
        """Update metrics with new batch.
        
        Args:
            predictions: Predicted logits or probabilities (batch, num_classes)
            targets: Ground truth labels (batch,)
            loss: Optional loss value
        """
        # Convert to CPU numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
        
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        if not self.predictions:
            return {}
        
        # Concatenate all batches
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = self.compute_accuracy(predictions, targets)
        
        # Top-k accuracy
        for k in [3, 5]:
            metrics[f'top_{k}_accuracy'] = self.compute_top_k_accuracy(predictions, targets, k)
        
        # Cross-entropy loss
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
        
        # Success Probability (SP) - average probability assigned to correct class
        metrics['success_probability'] = self.compute_success_probability(predictions, targets)
        
        # Hit Rate (HR) - percentage of times correct tile is in top-k
        metrics['hit_rate_top3'] = self.compute_hit_rate(predictions, targets, k=3)
        metrics['hit_rate_top5'] = self.compute_hit_rate(predictions, targets, k=5)
        
        # Per-class metrics
        per_class_metrics = self.compute_per_class_metrics(predictions, targets)
        metrics.update(per_class_metrics)
        
        return metrics
    
    @staticmethod
    def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy (exact match).
        
        Args:
            predictions: Predicted logits (N, num_classes)
            targets: Ground truth labels (N,)
            
        Returns:
            Accuracy value
        """
        _, predicted_classes = torch.max(predictions, dim=1)
        correct = (predicted_classes == targets).sum().item()
        total = targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def compute_top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """Compute top-k accuracy.
        
        Args:
            predictions: Predicted logits (N, num_classes)
            targets: Ground truth labels (N,)
            k: Top-k value
            
        Returns:
            Top-k accuracy
        """
        _, top_k_pred = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top_k_pred)
        correct = (top_k_pred == targets_expanded).any(dim=1).sum().item()
        total = targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def compute_success_probability(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Success Probability (average probability of correct class).
        
        Args:
            predictions: Predicted logits (N, num_classes)
            targets: Ground truth labels (N,)
            
        Returns:
            Average success probability
        """
        # Convert logits to probabilities
        probs = torch.softmax(predictions, dim=1)
        
        # Get probability of correct class for each sample
        correct_probs = probs[torch.arange(len(targets)), targets]
        
        return correct_probs.mean().item()
    
    @staticmethod
    def compute_hit_rate(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
        """Compute Hit Rate (percentage of times correct tile is in top-k).
        
        Same as top-k accuracy but with a different name convention.
        
        Args:
            predictions: Predicted logits (N, num_classes)
            targets: Ground truth labels (N,)
            k: Top-k value
            
        Returns:
            Hit rate
        """
        return MetricsCalculator.compute_top_k_accuracy(predictions, targets, k)
    
    def compute_per_class_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, any]:
        """Compute per-class precision, recall, F1.
        
        Args:
            predictions: Predicted logits (N, num_classes)
            targets: Ground truth labels (N,)
            
        Returns:
            Dictionary with per-class metrics
        """
        _, predicted_classes = torch.max(predictions, dim=1)
        
        # Convert to numpy
        y_true = targets.numpy()
        y_pred = predicted_classes.numpy()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # Per-class precision and recall
        precision_per_class = []
        recall_per_class = []
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
        
        # Macro-averaged metrics
        metrics = {
            'precision_macro': np.mean(precision_per_class),
            'recall_macro': np.mean(recall_per_class),
            'f1_macro': 2 * np.mean(precision_per_class) * np.mean(recall_per_class) / 
                       (np.mean(precision_per_class) + np.mean(recall_per_class) + 1e-8)
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix.
        
        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        if not self.predictions:
            return None
        
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        _, predicted_classes = torch.max(predictions, dim=1)
        
        cm = confusion_matrix(
            targets.numpy(), 
            predicted_classes.numpy(), 
            labels=range(self.num_classes)
        )
        
        return cm
    
    def get_classification_report(self) -> str:
        """Get detailed classification report.
        
        Returns:
            Classification report string
        """
        if not self.predictions:
            return "No predictions available"
        
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        _, predicted_classes = torch.max(predictions, dim=1)
        
        report = classification_report(
            targets.numpy(),
            predicted_classes.numpy(),
            labels=range(self.num_classes),
            target_names=[f"Tile_{i}" for i in range(self.num_classes)],
            zero_division=0
        )
        
        return report


class GameplayMetrics:
    """Metrics for gameplay performance."""
    
    def __init__(self):
        """Initialize gameplay metrics."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.games_played = 0
        self.games_won = 0
        self.total_points = 0
        self.riichi_declared = 0
        self.riichi_won = 0
        self.average_placement = []
    
    def update_game(self, placement: int, points: int, won: bool, riichi: bool):
        """Update metrics after a game.
        
        Args:
            placement: Final placement (1-4)
            points: Final points
            won: Whether player won
            riichi: Whether player declared riichi
        """
        self.games_played += 1
        if won:
            self.games_won += 1
        
        self.total_points += points
        self.average_placement.append(placement)
        
        if riichi:
            self.riichi_declared += 1
            if won:
                self.riichi_won += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute gameplay metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.games_played == 0:
            return {}
        
        metrics = {
            'games_played': self.games_played,
            'win_rate': self.games_won / self.games_played,
            'avg_points': self.total_points / self.games_played,
            'avg_placement': np.mean(self.average_placement),
            'riichi_rate': self.riichi_declared / self.games_played,
        }
        
        if self.riichi_declared > 0:
            metrics['riichi_success_rate'] = self.riichi_won / self.riichi_declared
        
        return metrics

