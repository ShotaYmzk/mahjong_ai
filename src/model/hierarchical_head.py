"""Hierarchical prediction head for Mahjong actions.

Implements a 3-level hierarchical output structure:
1. Action Head: Decide action type (discard, chi, pon, kan, riichi, tsumo, ron)
2. Claim Head: If claiming, decide which meld type
3. Discard Head: If discarding, decide which tile to discard

This hierarchical structure matches the decision-making process in Mahjong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ActionHead(nn.Module):
    """Predicts the type of action to take."""
    
    def __init__(self, d_model: int, num_actions: int = 7, dropout: float = 0.1):
        """Initialize action head.
        
        Args:
            d_model: Input feature dimension
            num_actions: Number of action types
                0: discard
                1: chi
                2: pon
                3: kan
                4: riichi
                5: tsumo
                6: ron
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_actions = num_actions
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Feature tensor of shape (batch, d_model)
            
        Returns:
            Action logits of shape (batch, num_actions)
        """
        return self.head(x)


class ClaimHead(nn.Module):
    """Predicts meld type when claiming (chi/pon/kan)."""
    
    def __init__(self, d_model: int, num_claim_types: int = 3, dropout: float = 0.1):
        """Initialize claim head.
        
        Args:
            d_model: Input feature dimension
            num_claim_types: Number of claim types (chi=0, pon=1, kan=2)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_claim_types = num_claim_types
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_claim_types)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Feature tensor of shape (batch, d_model)
            
        Returns:
            Claim type logits of shape (batch, num_claim_types)
        """
        return self.head(x)


class DiscardHead(nn.Module):
    """Predicts which tile to discard."""
    
    def __init__(self, d_model: int, num_tiles: int = 34, dropout: float = 0.1):
        """Initialize discard head.
        
        Args:
            d_model: Input feature dimension
            num_tiles: Number of tile types (34)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_tiles)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Feature tensor of shape (batch, d_model)
            
        Returns:
            Tile logits of shape (batch, num_tiles)
        """
        return self.head(x)


class HierarchicalHead(nn.Module):
    """Hierarchical prediction head combining all three heads.
    
    The hierarchical structure allows for:
    1. First deciding whether to discard or claim
    2. If claiming, deciding meld type
    3. If discarding, deciding which tile
    
    This matches the natural decision-making flow in Mahjong.
    """
    
    def __init__(self, 
                 d_model: int = 256, 
                 num_actions: int = 7,
                 num_claim_types: int = 3,
                 num_tiles: int = 34,
                 dropout: float = 0.1):
        """Initialize hierarchical head.
        
        Args:
            d_model: Input feature dimension
            num_actions: Number of action types
            num_claim_types: Number of claim types
            num_tiles: Number of tile types
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_actions = num_actions
        self.num_claim_types = num_claim_types
        self.num_tiles = num_tiles
        
        # Three prediction heads
        self.action_head = ActionHead(d_model, num_actions, dropout)
        self.claim_head = ClaimHead(d_model, num_claim_types, dropout)
        self.discard_head = DiscardHead(d_model, num_tiles, dropout)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all heads.
        
        Args:
            x: Feature tensor of shape (batch, d_model)
            
        Returns:
            Dictionary containing:
                - 'action': Action logits (batch, num_actions)
                - 'claim': Claim type logits (batch, num_claim_types)
                - 'discard': Tile logits (batch, num_tiles)
        """
        action_logits = self.action_head(x)
        claim_logits = self.claim_head(x)
        discard_logits = self.discard_head(x)
        
        return {
            'action': action_logits,
            'claim': claim_logits,
            'discard': discard_logits
        }
    
    def predict(self, x: torch.Tensor, return_probs: bool = False) -> Dict[str, torch.Tensor]:
        """Make predictions with probabilities.
        
        Args:
            x: Feature tensor of shape (batch, d_model)
            return_probs: If True, return probabilities instead of logits
            
        Returns:
            Dictionary containing predictions
        """
        outputs = self.forward(x)
        
        if return_probs:
            outputs = {
                'action': F.softmax(outputs['action'], dim=-1),
                'claim': F.softmax(outputs['claim'], dim=-1),
                'discard': F.softmax(outputs['discard'], dim=-1)
            }
        
        return outputs
    
    def predict_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict action and tile to discard.
        
        Args:
            x: Feature tensor of shape (batch, d_model)
            
        Returns:
            Tuple of (action_pred, tile_pred)
        """
        outputs = self.predict(x, return_probs=True)
        
        action_pred = torch.argmax(outputs['action'], dim=-1)
        tile_pred = torch.argmax(outputs['discard'], dim=-1)
        
        return action_pred, tile_pred
    
    def compute_loss(self, 
                     outputs: Dict[str, torch.Tensor],
                     action_labels: torch.Tensor,
                     claim_labels: Optional[torch.Tensor] = None,
                     discard_labels: Optional[torch.Tensor] = None,
                     action_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute hierarchical loss.
        
        Args:
            outputs: Output dictionary from forward()
            action_labels: Ground truth action labels (batch,)
            claim_labels: Ground truth claim type labels (batch,), optional
            discard_labels: Ground truth tile labels (batch,), optional
            action_weights: Weights for each action type (num_actions,)
            
        Returns:
            Dictionary containing:
                - 'total': Total loss
                - 'action': Action loss
                - 'claim': Claim loss (if claim_labels provided)
                - 'discard': Discard loss (if discard_labels provided)
        """
        losses = {}
        
        # Action loss (always computed)
        if action_weights is not None:
            action_loss = F.cross_entropy(outputs['action'], action_labels, weight=action_weights)
        else:
            action_loss = F.cross_entropy(outputs['action'], action_labels)
        losses['action'] = action_loss
        
        total_loss = action_loss
        
        # Claim loss (only for samples where action is chi/pon/kan)
        if claim_labels is not None:
            # Mask for claim actions (actions 1, 2, 3)
            claim_mask = (action_labels >= 1) & (action_labels <= 3)
            if claim_mask.any():
                claim_loss = F.cross_entropy(
                    outputs['claim'][claim_mask], 
                    claim_labels[claim_mask]
                )
                losses['claim'] = claim_loss
                total_loss = total_loss + 0.5 * claim_loss
        
        # Discard loss (only for samples where action is discard or riichi)
        if discard_labels is not None:
            # Mask for discard actions (actions 0, 4)
            discard_mask = (action_labels == 0) | (action_labels == 4)
            if discard_mask.any():
                discard_loss = F.cross_entropy(
                    outputs['discard'][discard_mask],
                    discard_labels[discard_mask]
                )
                losses['discard'] = discard_loss
                total_loss = total_loss + discard_loss
        
        losses['total'] = total_loss
        
        return losses


class SimplifiedDiscardOnlyHead(nn.Module):
    """Simplified head that only predicts tile discards (for supervised learning)."""
    
    def __init__(self, d_model: int = 256, num_tiles: int = 34, dropout: float = 0.1):
        """Initialize simplified discard-only head.
        
        Args:
            d_model: Input feature dimension
            num_tiles: Number of tile types (34)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_tiles)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Feature tensor of shape (batch, d_model)
            
        Returns:
            Tile logits of shape (batch, num_tiles)
        """
        return self.head(x)
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.
        
        Args:
            logits: Predicted logits (batch, num_tiles)
            labels: Ground truth labels (batch,)
            
        Returns:
            Loss scalar
        """
        return F.cross_entropy(logits, labels)


class CompleteMahjongModel(nn.Module):
    """Complete Mahjong AI model combining backbone and heads."""
    
    def __init__(self, backbone: nn.Module, head: nn.Module):
        """Initialize complete model.
        
        Args:
            backbone: Backbone model (e.g., TIT or SimplifiedTransformer)
            head: Prediction head (e.g., HierarchicalHead or SimplifiedDiscardOnlyHead)
        """
        super().__init__()
        
        self.backbone = backbone
        self.head = head
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass through complete model.
        
        Args:
            x: Input tensor
            mask: Optional mask
            
        Returns:
            Predictions from head
        """
        features, attention_weights = self.backbone(x, mask)
        outputs = self.head(features)
        
        return outputs, attention_weights
    
    def get_features(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features from backbone.
        
        Args:
            x: Input tensor
            mask: Optional mask
            
        Returns:
            Feature tensor
        """
        features, _ = self.backbone(x, mask)
        return features


