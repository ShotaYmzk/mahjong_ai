"""Attention visualization for XAI.

Provides visualization tools for:
- Attention heatmaps
- Attention flow diagrams
- Feature importance plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Visualizer for attention weights and patterns."""
    
    def __init__(self, output_dir: str = 'outputs/visualizations'):
        """Initialize attention visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
    
    def plot_attention_heatmap(self,
                              attention_weights: np.ndarray,
                              title: str = 'Attention Heatmap',
                              xlabel: str = 'Key',
                              ylabel: str = 'Query',
                              save_name: Optional[str] = None):
        """Plot attention weights as heatmap.
        
        Args:
            attention_weights: Attention matrix (query_len, key_len)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_name: Filename to save (None for display only)
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, annot=False, cmap='viridis', 
                   cbar=True, square=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {save_path}")
        
        plt.close()
    
    def plot_multi_head_attention(self,
                                 attention_weights: np.ndarray,
                                 num_heads: int,
                                 title: str = 'Multi-Head Attention',
                                 save_name: Optional[str] = None):
        """Plot multi-head attention patterns.
        
        Args:
            attention_weights: Attention tensor (num_heads, query_len, key_len)
            num_heads: Number of attention heads
            title: Plot title
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, num_heads // 2, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(num_heads):
            sns.heatmap(attention_weights[i], ax=axes[i], cmap='viridis',
                       cbar=True, square=True)
            axes[i].set_title(f'Head {i+1}')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multi-head attention to {save_path}")
        
        plt.close()
    
    def plot_attention_rollout(self,
                              rolled_attention: np.ndarray,
                              token_names: Optional[List[str]] = None,
                              title: str = 'Attention Rollout',
                              save_name: Optional[str] = None):
        """Plot attention rollout across layers.
        
        Args:
            rolled_attention: Rolled attention matrix (seq_len, seq_len)
            token_names: Names for tokens
            title: Plot title
            save_name: Filename to save
        """
        plt.figure(figsize=(12, 10))
        
        if token_names is not None:
            sns.heatmap(rolled_attention, annot=False, cmap='Blues',
                       xticklabels=token_names, yticklabels=token_names,
                       cbar=True, square=True)
        else:
            sns.heatmap(rolled_attention, annot=False, cmap='Blues',
                       cbar=True, square=True)
        
        plt.title(title)
        plt.xlabel('Source Token')
        plt.ylabel('Target Token')
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention rollout to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self,
                               importance_scores: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               title: str = 'Feature Importance',
                               top_k: int = 20,
                               save_name: Optional[str] = None):
        """Plot feature importance scores.
        
        Args:
            importance_scores: Importance scores (num_features,)
            feature_names: Names for features
            title: Plot title
            top_k: Number of top features to show
            save_name: Filename to save
        """
        # Get top-k features
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        top_scores = importance_scores[top_indices]
        
        if feature_names is not None:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f'Feature {i}' for i in top_indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_k), top_scores)
        plt.yticks(range(top_k), top_names)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {save_path}")
        
        plt.close()
    
    def plot_tile_attention(self,
                           attention_to_tiles: Dict[str, float],
                           title: str = 'Tile Attention',
                           save_name: Optional[str] = None):
        """Plot attention to different tile types.
        
        Args:
            attention_to_tiles: Dictionary mapping tile names to attention scores
            title: Plot title
            save_name: Filename to save
        """
        tiles = list(attention_to_tiles.keys())
        scores = list(attention_to_tiles.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(tiles, scores)
        plt.xlabel('Tile')
        plt.ylabel('Attention Score')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved tile attention to {save_path}")
        
        plt.close()
    
    def create_attention_summary(self,
                                attention_dict: Dict[str, np.ndarray],
                                sample_id: str = 'sample',
                                save_dir: Optional[str] = None):
        """Create summary of all attention visualizations for a sample.
        
        Args:
            attention_dict: Dictionary of attention weights from different layers
            sample_id: Identifier for this sample
            save_dir: Directory to save (uses default if None)
        """
        if save_dir:
            save_path = Path(save_dir)
        else:
            save_path = self.output_dir / sample_id
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating attention summary for {sample_id}")
        
        # Plot each attention layer
        for layer_name, attention in attention_dict.items():
            if attention.ndim == 2:
                # Single attention matrix
                self.plot_attention_heatmap(
                    attention,
                    title=f'{layer_name} Attention',
                    save_name=f'{layer_name}_attention.png'
                )
            elif attention.ndim == 3:
                # Multi-head attention
                num_heads = attention.shape[0]
                self.plot_multi_head_attention(
                    attention,
                    num_heads,
                    title=f'{layer_name} Multi-Head Attention',
                    save_name=f'{layer_name}_multihead.png'
                )
        
        logger.info(f"Saved attention summary to {save_path}")


def visualize_training_metrics(history: Dict[str, List[float]],
                               save_path: str = 'outputs/visualizations/training_metrics.png'):
    """Visualize training metrics over epochs.
    
    Args:
        history: Dictionary with 'train' and 'val' lists of metric dictionaries
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    train_losses = [m['loss'] for m in history['train']]
    val_losses = [m['loss'] for m in history['val']]
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    train_accs = [m['accuracy'] for m in history['train']]
    val_accs = [m['accuracy'] for m in history['val']]
    
    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(val_accs, label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training metrics to {save_path}")


