"""Model module containing TIT architecture and XAI hooks."""

from .transformer_tit import TIT, SimplifiedMahjongTransformer
from .hierarchical_head import (
    HierarchicalHead, 
    SimplifiedDiscardOnlyHead, 
    CompleteMahjongModel
)
from .xai_hooks import (
    XAIHooks, 
    AttentionAnalyzer, 
    GradientAttribution,
    ActivationHook,
    AttentionHook,
    GradientHook
)

__all__ = [
    "TIT", 
    "SimplifiedMahjongTransformer",
    "HierarchicalHead", 
    "SimplifiedDiscardOnlyHead",
    "CompleteMahjongModel",
    "XAIHooks",
    "AttentionAnalyzer",
    "GradientAttribution",
    "ActivationHook",
    "AttentionHook",
    "GradientHook"
]

