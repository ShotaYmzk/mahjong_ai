"""Feature encoder for Mahjong tiles.

Implements 1x34 tile encoding based on Tjong-Kim-Sang et al.
Each tile type (0-33) is represented by its count (0-4).

Tile types:
- 0-8: Man (characters) 1m-9m
- 9-17: Pin (circles) 1p-9p
- 18-26: Sou (bamboo) 1s-9s
- 27-33: Honors (E, S, W, N, Haku, Hatsu, Chun)
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TileEncoder:
    """Encoder for Mahjong tiles using 1x34 representation."""
    
    # Tile type constants
    NUM_TILE_TYPES = 34
    NUM_TILES_PER_TYPE = 4
    
    # Tile ranges
    MAN_START = 0
    MAN_END = 9
    PIN_START = 9
    PIN_END = 18
    SOU_START = 18
    SOU_END = 27
    HONOR_START = 27
    HONOR_END = 34
    
    def __init__(self):
        """Initialize the tile encoder."""
        pass
    
    @staticmethod
    def tile_id_to_type(tile_id: int) -> int:
        """Convert tile ID (0-135) to tile type (0-33).
        
        Args:
            tile_id: Tile ID from Tenhou (0-135)
            
        Returns:
            Tile type (0-33)
        """
        return tile_id // 4
    
    @staticmethod
    def tile_type_to_name(tile_type: int) -> str:
        """Convert tile type to human-readable name.
        
        Args:
            tile_type: Tile type (0-33)
            
        Returns:
            Tile name (e.g., "1m", "5p", "E")
        """
        if 0 <= tile_type < 9:
            return f"{tile_type + 1}m"
        elif 9 <= tile_type < 18:
            return f"{tile_type - 8}p"
        elif 18 <= tile_type < 27:
            return f"{tile_type - 17}s"
        else:
            honor_names = ['E', 'S', 'W', 'N', 'W', 'G', 'R']
            return honor_names[tile_type - 27]
    
    def encode_tiles(self, tile_ids: List[int]) -> np.ndarray:
        """Encode a list of tile IDs into 1x34 format.
        
        Args:
            tile_ids: List of tile IDs (0-135)
            
        Returns:
            34-dimensional array with counts for each tile type
        """
        encoding = np.zeros(self.NUM_TILE_TYPES, dtype=np.int32)
        
        for tile_id in tile_ids:
            tile_type = self.tile_id_to_type(tile_id)
            if 0 <= tile_type < self.NUM_TILE_TYPES:
                encoding[tile_type] += 1
        
        return encoding
    
    def encode_hand(self, hand: List[int]) -> np.ndarray:
        """Encode a hand (13 or 14 tiles) into 1x34 format.
        
        Args:
            hand: List of tile IDs in hand
            
        Returns:
            34-dimensional array
        """
        return self.encode_tiles(hand)
    
    def encode_discards(self, discards: List[int]) -> np.ndarray:
        """Encode discard pile into 1x34 format.
        
        Args:
            discards: List of tile IDs discarded
            
        Returns:
            34-dimensional array
        """
        return self.encode_tiles(discards)
    
    def encode_melds(self, melds: List[Tuple[str, List[int]]]) -> np.ndarray:
        """Encode melds (chi/pon/kan) into 1x34 format.
        
        Args:
            melds: List of (meld_type, tile_ids) tuples
            
        Returns:
            34-dimensional array
        """
        all_tiles = []
        for meld_type, tiles in melds:
            all_tiles.extend(tiles)
        
        return self.encode_tiles(all_tiles)
    
    def encode_visible_tiles(self, hand: List[int], melds: List[Tuple[str, List[int]]], 
                            discards: List[int]) -> np.ndarray:
        """Encode all visible tiles for a player.
        
        Args:
            hand: Tiles in hand
            melds: List of melds
            discards: Discarded tiles
            
        Returns:
            34-dimensional array
        """
        all_tiles = hand.copy()
        for meld_type, tiles in melds:
            all_tiles.extend(tiles)
        all_tiles.extend(discards)
        
        return self.encode_tiles(all_tiles)
    
    def encode_game_state(self, own_hand: List[int], own_melds: List[Tuple[str, List[int]]],
                         all_discards: List[List[int]], 
                         opponent_melds: List[List[Tuple[str, List[int]]]],
                         dora_indicators: List[int]) -> np.ndarray:
        """Encode complete game state into feature vector.
        
        Feature vector structure (34 * N dimensions):
        - Own hand (34)
        - Own melds (34)
        - Opponent 1 discards (34)
        - Opponent 2 discards (34)
        - Opponent 3 discards (34)
        - Opponent 1 melds (34)
        - Opponent 2 melds (34)
        - Opponent 3 melds (34)
        - Dora indicators (34)
        - Visible opponent tiles estimate (34) - for unseen tiles
        
        Args:
            own_hand: Own hand tiles
            own_melds: Own melds
            all_discards: Discards for all 4 players
            opponent_melds: Melds for 3 opponents
            dora_indicators: Dora indicator tiles
            
        Returns:
            Feature vector of shape (34 * 10,)
        """
        features = []
        
        # Own hand
        features.append(self.encode_hand(own_hand))
        
        # Own melds
        features.append(self.encode_melds(own_melds))
        
        # All discards (4 players)
        for discards in all_discards:
            features.append(self.encode_discards(discards))
        
        # Opponent melds (3 opponents)
        for melds in opponent_melds:
            features.append(self.encode_melds(melds))
        
        # Dora indicators
        features.append(self.encode_tiles(dora_indicators))
        
        return np.concatenate(features)
    
    def encode_action(self, action_type: str, tile: int = None) -> Tuple[int, int]:
        """Encode an action into categorical labels.
        
        Args:
            action_type: Type of action ('discard', 'chi', 'pon', 'kan', 'riichi', 'tsumo', 'ron')
            tile: Tile involved (for discard)
            
        Returns:
            (action_category, tile_type) tuple
            action_category: 0=discard, 1=chi, 2=pon, 3=kan, 4=riichi, 5=tsumo, 6=ron
            tile_type: 0-33 for tile, or -1 if not applicable
        """
        action_map = {
            'discard': 0,
            'chi': 1,
            'pon': 2,
            'kan': 3,
            'riichi': 4,
            'tsumo': 5,
            'ron': 6
        }
        
        action_category = action_map.get(action_type, 0)
        tile_type = self.tile_id_to_type(tile) if tile is not None else -1
        
        return action_category, tile_type
    
    def create_one_hot_tile(self, tile_id: int) -> np.ndarray:
        """Create one-hot encoding for a single tile.
        
        Args:
            tile_id: Tile ID (0-135)
            
        Returns:
            One-hot vector of length 34
        """
        one_hot = np.zeros(self.NUM_TILE_TYPES, dtype=np.float32)
        tile_type = self.tile_id_to_type(tile_id)
        if 0 <= tile_type < self.NUM_TILE_TYPES:
            one_hot[tile_type] = 1.0
        return one_hot
    
    def decode_tiles(self, encoding: np.ndarray) -> List[str]:
        """Decode 1x34 encoding back to tile names.
        
        Args:
            encoding: 34-dimensional array
            
        Returns:
            List of tile names
        """
        tiles = []
        for tile_type in range(self.NUM_TILE_TYPES):
            count = int(encoding[tile_type])
            tile_name = self.tile_type_to_name(tile_type)
            tiles.extend([tile_name] * count)
        
        return tiles
    
    def normalize_encoding(self, encoding: np.ndarray) -> np.ndarray:
        """Normalize encoding by dividing by max tile count.
        
        Args:
            encoding: 34-dimensional array
            
        Returns:
            Normalized array (values 0-1)
        """
        return encoding.astype(np.float32) / self.NUM_TILES_PER_TYPE
    
    def is_valid_hand(self, encoding: np.ndarray) -> bool:
        """Check if an encoding represents a valid hand.
        
        Args:
            encoding: 34-dimensional array
            
        Returns:
            True if valid (13 or 14 tiles, no type exceeds 4)
        """
        total = encoding.sum()
        if total not in [13, 14]:
            return False
        
        if np.any(encoding > self.NUM_TILES_PER_TYPE):
            return False
        
        return True
    
    def compute_remaining_tiles(self, visible_tiles: np.ndarray) -> np.ndarray:
        """Compute remaining unseen tiles from visible tiles.
        
        Args:
            visible_tiles: 34-dimensional array of visible tiles
            
        Returns:
            34-dimensional array of remaining tiles
        """
        all_tiles = np.full(self.NUM_TILE_TYPES, self.NUM_TILES_PER_TYPE, dtype=np.int32)
        remaining = all_tiles - visible_tiles
        remaining = np.maximum(remaining, 0)  # Ensure non-negative
        
        return remaining


class AdvancedFeatureExtractor:
    """Extract advanced features for neural network training."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.tile_encoder = TileEncoder()
    
    def extract_round_features(self, round_state: Dict) -> Dict[str, np.ndarray]:
        """Extract features for a specific round state.
        
        Args:
            round_state: Dictionary containing round information
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic tile features
        features['hand'] = self.tile_encoder.encode_hand(round_state.get('hand', []))
        features['discards'] = self.tile_encoder.encode_discards(round_state.get('discards', []))
        features['melds'] = self.tile_encoder.encode_melds(round_state.get('melds', []))
        
        # Contextual features
        features['is_dealer'] = np.array([round_state.get('is_dealer', 0)], dtype=np.float32)
        features['round_wind'] = np.array([round_state.get('round_wind', 0)], dtype=np.int32)
        features['honba'] = np.array([round_state.get('honba', 0)], dtype=np.int32)
        features['turn_number'] = np.array([round_state.get('turn', 0)], dtype=np.int32)
        
        # Score features (normalized)
        features['score'] = np.array([round_state.get('score', 25000)], dtype=np.float32) / 100000.0
        
        return features
    
    def extract_sequence_features(self, states: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract features for a sequence of states.
        
        Args:
            states: List of state dictionaries
            
        Returns:
            Dictionary of feature arrays (with time dimension)
        """
        all_features = [self.extract_round_features(state) for state in states]
        
        # Stack features along time dimension
        stacked_features = {}
        for key in all_features[0].keys():
            stacked_features[key] = np.stack([f[key] for f in all_features])
        
        return stacked_features


