"""Transformer-in-Transformer (TIT) architecture for Mahjong AI.

Implements a hierarchical transformer structure with:
- Inner Transformer: Processes tiles within a group
- Outer Transformer: Processes relationships between groups

Based on the TIT architecture adapted for Mahjong tile sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class InnerTransformer(nn.Module):
    """Inner transformer for processing tiles within a group."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float = 0.1, num_layers: int = 2):
        """Initialize inner transformer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through inner transformer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x


class OuterTransformer(nn.Module):
    """Outer transformer for processing relationships between tile groups."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float = 0.1, num_layers: int = 4):
        """Initialize outer transformer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through outer transformer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x


class TIT(nn.Module):
    """Transformer-in-Transformer model for Mahjong AI.
    
    Architecture:
    1. Tile embedding layer
    2. Group tiles into patches/groups
    3. Inner transformer processes tiles within each group
    4. Aggregate group representations
    5. Outer transformer processes group relationships
    6. Output layer for action prediction
    """
    
    def __init__(self, 
                 input_dim: int = 340,  # 34 tile types * 10 feature groups
                 d_model: int = 256,
                 nhead_inner: int = 4,
                 nhead_outer: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 num_inner_layers: int = 2,
                 num_outer_layers: int = 4,
                 num_tile_groups: int = 10,  # Number of feature groups
                 tile_group_size: int = 34):  # Size of each group
        """Initialize TIT model.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model hidden dimension
            nhead_inner: Number of heads in inner transformer
            nhead_outer: Number of heads in outer transformer
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            num_inner_layers: Number of inner transformer layers
            num_outer_layers: Number of outer transformer layers
            num_tile_groups: Number of tile groups
            tile_group_size: Size of each tile group
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_tile_groups = num_tile_groups
        self.tile_group_size = tile_group_size
        
        # Input projection for each tile group
        self.tile_embedding = nn.Linear(tile_group_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=num_tile_groups, dropout=dropout)
        
        # Inner transformer (processes tiles within groups)
        self.inner_transformer = InnerTransformer(
            d_model=d_model,
            nhead=nhead_inner,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_inner_layers
        )
        
        # Group aggregation (mean pooling or learned aggregation)
        self.group_aggregation = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Outer transformer (processes relationships between groups)
        self.outer_transformer = OuterTransformer(
            d_model=d_model,
            nhead=nhead_outer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_outer_layers
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head will be added separately (hierarchical head)
        self.feature_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Forward pass through TIT model.
        
        Args:
            x: Input tensor of shape (batch, input_dim) or (batch, seq_len, input_dim)
            mask: Optional mask tensor
            
        Returns:
            Tuple of (feature_vector, attention_weights_dict)
        """
        batch_size = x.size(0)
        
        # Reshape input into tile groups
        # x: (batch, input_dim) -> (batch, num_groups, group_size)
        if x.dim() == 2:
            x = x.view(batch_size, self.num_tile_groups, self.tile_group_size)
        
        # Embed each tile group
        # (batch, num_groups, group_size) -> (batch, num_groups, d_model)
        x = self.tile_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Inner transformer: process tiles within each group
        inner_output = self.inner_transformer(x, mask)
        
        # Aggregate group representations
        group_repr = self.group_aggregation(inner_output)
        
        # Outer transformer: process relationships between groups
        outer_output = self.outer_transformer(group_repr, mask)
        
        # Global pooling
        # (batch, num_groups, d_model) -> (batch, d_model, num_groups) -> (batch, d_model, 1)
        pooled = self.global_pool(outer_output.transpose(1, 2))
        # (batch, d_model, 1) -> (batch, d_model)
        pooled = pooled.squeeze(2)
        
        # Feature projection
        features = self.feature_projection(pooled)
        
        # Collect attention weights for XAI
        attention_weights = {
            'inner_output': inner_output,
            'outer_output': outer_output,
            'group_repr': group_repr
        }
        
        return features, attention_weights
    
    def get_attention_weights(self) -> dict:
        """Extract attention weights from transformer layers.
        
        Returns:
            Dictionary of attention weights
        """
        attention_weights = {}
        
        # Extract from inner transformer
        for i, layer in enumerate(self.inner_transformer.transformer.layers):
            if hasattr(layer, 'self_attn'):
                attention_weights[f'inner_layer_{i}'] = layer.self_attn
        
        # Extract from outer transformer
        for i, layer in enumerate(self.outer_transformer.transformer.layers):
            if hasattr(layer, 'self_attn'):
                attention_weights[f'outer_layer_{i}'] = layer.self_attn
        
        return attention_weights


class SimplifiedMahjongTransformer(nn.Module):
    """Simplified transformer model for Mahjong (alternative to TIT)."""
    
    def __init__(self, 
                 input_dim: int = 340,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        """Initialize simplified transformer.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            mask: Optional mask
            
        Returns:
            Tuple of (features, attention_dict)
        """
        # Project input
        x = self.input_projection(x)
        
        # Add dummy sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Remove sequence dimension and project
        x = x.squeeze(1)
        features = self.output_projection(x)
        
        attention_weights = {'transformer_output': x}
        
        return features, attention_weights


