"""Preprocessing module for Mahjong game data."""

# Original modules (V1)
from .parse_xml import TenhouXMLParser
from .feature_encoder import TileEncoder, AdvancedFeatureExtractor
from .dataset_builder import MahjongDatasetBuilder, SequenceDatasetBuilder
from .game_state import RoundGameState, DiscardHistory, GameStateManager
from .advanced_game_state import AdvancedGameState

# Enhanced modules (V2)
from .enhanced_parser import EnhancedXMLParser, EnhancedGame, EnhancedGameAction, EnhancedRoundState
from .game_state_tracker import ComprehensiveGameStateTracker, ComprehensiveGameState
from .feature_encoder_v2 import AdvancedFeatureEncoderV2, LabelEncoder
from .data_validator import DataValidator, ValidationReport
from .dataset_builder_v2 import ComprehensiveDatasetBuilder

__all__ = [
    # V1 modules
    "TenhouXMLParser", 
    "TileEncoder", 
    "AdvancedFeatureExtractor",
    "MahjongDatasetBuilder", 
    "SequenceDatasetBuilder",
    "RoundGameState",
    "DiscardHistory",
    "GameStateManager",
    "AdvancedGameState",
    
    # V2 modules (recommended)
    "EnhancedXMLParser",
    "EnhancedGame",
    "EnhancedGameAction",
    "EnhancedRoundState",
    "ComprehensiveGameStateTracker",
    "ComprehensiveGameState",
    "AdvancedFeatureEncoderV2",
    "LabelEncoder",
    "DataValidator",
    "ValidationReport",
    "ComprehensiveDatasetBuilder",
]

