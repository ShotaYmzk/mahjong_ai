"""Advanced Feature Encoder V2 for Mahjong AI.

高度な特徴量エンコーディング（2,099次元の固定長ベクトル）

特徴量構成:
- 基本特徴量（1x34 × 10）: 340次元
- 時系列特徴量: k×34 + m×39次元（可変）
- メタ特徴量: 31次元
- 待ち・向聴数: 38次元
- 捨て牌メタ情報: 170次元
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .game_state_tracker import (
    ComprehensiveGameState, DrawHistoryEntry, DiscardHistoryEntry,
    MeldHistoryEntry, RiichiInfo, WaitInfo
)

logger = logging.getLogger(__name__)


class AdvancedFeatureEncoderV2:
    """高度な特徴量エンコーダー（V2）
    
    固定長ベクトルを生成し、教師あり学習に適した形式にする。
    """
    
    def __init__(self, 
                 draw_history_length: int = 8,
                 discard_history_length: int = 32,
                 normalize: bool = True):
        """Initialize the encoder.
        
        Args:
            draw_history_length: ツモ履歴の最大長（k）
            discard_history_length: 捨て牌履歴の最大長（m）
            normalize: 正規化を有効にするか
        """
        self.draw_history_length = draw_history_length
        self.discard_history_length = discard_history_length
        self.normalize = normalize
        
        # Calculate feature dimensions
        self.basic_dim = 340  # 34 * 10
        self.draw_seq_dim = draw_history_length * 34
        self.discard_seq_dim = discard_history_length * 39  # 34 + 4 + 1
        self.meta_dim = 31
        self.wait_dim = 38
        self.discard_meta_dim = 170  # 4*34 + 34
        
        self.total_dim = (
            self.basic_dim +
            self.draw_seq_dim +
            self.discard_seq_dim +
            self.meta_dim +
            self.wait_dim +
            self.discard_meta_dim
        )
        
        logger.info(f"Feature encoder initialized with total dimension: {self.total_dim}")
        logger.info(f"  Basic features: {self.basic_dim}")
        logger.info(f"  Draw sequence: {self.draw_seq_dim}")
        logger.info(f"  Discard sequence: {self.discard_seq_dim}")
        logger.info(f"  Meta features: {self.meta_dim}")
        logger.info(f"  Wait features: {self.wait_dim}")
        logger.info(f"  Discard meta: {self.discard_meta_dim}")
    
    def encode_state(self, state: ComprehensiveGameState) -> np.ndarray:
        """Encode a comprehensive game state to fixed-length feature vector.
        
        Args:
            state: ComprehensiveGameState object
            
        Returns:
            Feature vector of shape (total_dim,)
        """
        features = []
        
        # 1. Basic features (340)
        basic_features = self._encode_basic_features(state)
        features.append(basic_features)
        
        # 2. Draw sequence (k * 34)
        draw_seq = self._encode_draw_sequence(state)
        features.append(draw_seq)
        
        # 3. Discard sequence (m * 39)
        discard_seq = self._encode_discard_sequence(state)
        features.append(discard_seq)
        
        # 4. Meta features (31)
        meta_features = self._encode_meta_features(state)
        features.append(meta_features)
        
        # 5. Wait features (38)
        wait_features = self._encode_wait_features(state)
        features.append(wait_features)
        
        # 6. Discard meta features (170)
        discard_meta = self._encode_discard_meta_features(state)
        features.append(discard_meta)
        
        # Concatenate all features
        feature_vector = np.concatenate(features)
        
        # Verify dimension
        assert feature_vector.shape[0] == self.total_dim, \
            f"Feature dimension mismatch: expected {self.total_dim}, got {feature_vector.shape[0]}"
        
        return feature_vector.astype(np.float32)
    
    def _encode_basic_features(self, state: ComprehensiveGameState) -> np.ndarray:
        """Encode basic features (1x34 × 10 = 340).
        
        Features:
        - Own hand (34)
        - Own melds (34)
        - Own discards (34)
        - Opponent 1 discards (34)
        - Opponent 2 discards (34)
        - Opponent 3 discards (34)
        - Opponent 1 melds (34)
        - Opponent 2 melds (34)
        - Opponent 3 melds (34)
        - Dora indicators (34)
        """
        features = []
        
        # Own hand
        hand_encoding = state.hand_types.copy()
        if self.normalize:
            hand_encoding = hand_encoding.astype(np.float32) / 4.0
        features.append(hand_encoding)
        
        # Own melds
        meld_encoding = state.meld_types.copy()
        if self.normalize:
            meld_encoding = meld_encoding.astype(np.float32) / 4.0
        features.append(meld_encoding)
        
        # Own discards
        own_discard_encoding = self._encode_discards_1x34(state.own_discards)
        if self.normalize:
            own_discard_encoding = own_discard_encoding.astype(np.float32) / 4.0
        features.append(own_discard_encoding)
        
        # Opponent discards (3 opponents)
        for opp_discards in state.opponent_discards:
            opp_encoding = self._encode_discards_1x34(opp_discards)
            if self.normalize:
                opp_encoding = opp_encoding.astype(np.float32) / 4.0
            features.append(opp_encoding)
        
        # Opponent melds (3 opponents)
        for opp_melds in state.opponent_melds:
            opp_meld_encoding = self._encode_melds_1x34(opp_melds)
            if self.normalize:
                opp_meld_encoding = opp_meld_encoding.astype(np.float32) / 4.0
            features.append(opp_meld_encoding)
        
        # Dora indicators
        dora_encoding = self._encode_dora_indicators(state.dora_indicators)
        if self.normalize:
            dora_encoding = dora_encoding.astype(np.float32) / 4.0
        features.append(dora_encoding)
        
        return np.concatenate(features)
    
    def _encode_draw_sequence(self, state: ComprehensiveGameState) -> np.ndarray:
        """Encode draw history as sequence (k × 34).
        
        Args:
            state: Game state
            
        Returns:
            Sequence array of shape (k * 34,)
        """
        sequence = np.zeros((self.draw_history_length, 34), dtype=np.float32)
        
        # Get last k draws
        recent_draws = state.draw_history[-self.draw_history_length:]
        
        for i, draw in enumerate(recent_draws):
            if 0 <= draw.tile_type < 34:
                sequence[i, draw.tile_type] = 1.0
        
        return sequence.flatten()
    
    def _encode_discard_sequence(self, state: ComprehensiveGameState) -> np.ndarray:
        """Encode discard history as sequence (m × 39).
        
        Each entry: 34 (tile one-hot) + 4 (player one-hot) + 1 (riichi flag)
        
        Args:
            state: Game state
            
        Returns:
            Sequence array of shape (m * 39,)
        """
        sequence = np.zeros((self.discard_history_length, 39), dtype=np.float32)
        
        # Get last m discards
        recent_discards = state.discard_history[-self.discard_history_length:]
        
        for i, discard in enumerate(recent_discards):
            # Tile one-hot (34)
            if 0 <= discard.tile_type < 34:
                sequence[i, discard.tile_type] = 1.0
            
            # Player one-hot (4)
            if 0 <= discard.player_id < 4:
                sequence[i, 34 + discard.player_id] = 1.0
            
            # Riichi flag (1)
            sequence[i, 38] = 1.0 if discard.is_riichi_discard else 0.0
        
        return sequence.flatten()
    
    def _encode_meta_features(self, state: ComprehensiveGameState) -> np.ndarray:
        """Encode meta features (31).
        
        Features:
        - Round info (8): round_wind, round_index, dealer position, honba, riichi_sticks, turn, remaining_tiles, is_all_last
        - Dealer info (4): is_dealer for each player (one-hot)
        - Riichi declared (4): riichi status for each player
        - Riichi turn (4): normalized turn when riichi was declared
        - Score state (4): normalized scores for each player
        - Score diff (1): normalized difference from 1st place
        - Rank (1): current rank (0-3)
        - Turn info (2): current turn (normalized), remaining draws (normalized)
        - Red dora (3): visible red 5m/5p/5s count
        """
        features = []
        
        # Round info (8)
        round_info = np.array([
            state.round_wind / 1.0,  # 0 or 1 (East/South)
            state.round_index / 3.0,  # 0-3 normalized
            state.dealer / 3.0,  # 0-3 normalized
            state.honba / 5.0,  # Normalized (cap at 5)
            state.riichi_sticks / 4.0,  # Normalized (cap at 4)
            state.turn / 70.0,  # Normalized (typical max ~70 turns)
            max(0, (70 - state.turn)) / 70.0,  # Remaining tiles
            1.0 if state.round_wind == 1 and state.round_index == 3 else 0.0  # Is all last
        ], dtype=np.float32)
        features.append(round_info)
        
        # Dealer info (4) - one-hot
        dealer_info = np.zeros(4, dtype=np.float32)
        dealer_info[state.dealer] = 1.0
        features.append(dealer_info)
        
        # Riichi declared (4)
        riichi_declared = np.array([
            1.0 if info.declared else 0.0 for info in state.riichi_info
        ], dtype=np.float32)
        features.append(riichi_declared)
        
        # Riichi turn (4) - normalized
        riichi_turns = np.array([
            (info.declared_turn / 70.0) if info.declared_turn is not None else 0.0
            for info in state.riichi_info
        ], dtype=np.float32)
        features.append(riichi_turns)
        
        # Score state (4) - normalized
        scores_normalized = np.array([
            score / 100000.0 for score in state.scores
        ], dtype=np.float32)
        features.append(scores_normalized)
        
        # Score diff (1)
        score_diff = np.array([
            state.score_diffs[state.player_id] / 100000.0
        ], dtype=np.float32)
        features.append(score_diff)
        
        # Rank (1)
        rank_normalized = np.array([state.rank / 3.0], dtype=np.float32)
        features.append(rank_normalized)
        
        # Turn info (2)
        turn_info = np.array([
            state.turn / 70.0,
            max(0, 70 - state.turn) / 70.0
        ], dtype=np.float32)
        features.append(turn_info)
        
        # Red dora (3)
        red_dora = np.array([
            state.visible_red_dora.get('5m', 0) / 1.0,
            state.visible_red_dora.get('5p', 0) / 1.0,
            state.visible_red_dora.get('5s', 0) / 1.0
        ], dtype=np.float32)
        features.append(red_dora)
        
        return np.concatenate(features)
    
    def _encode_wait_features(self, state: ComprehensiveGameState) -> np.ndarray:
        """Encode wait and shanten features (38).
        
        Features:
        - Shanten (1): normalized shanten number
        - Is tenpai (1): 0/1 flag
        - Wait count (1): total number of waiting tiles
        - Wait types (1): number of different tile types
        - Wait tiles (34): one-hot for each waiting tile type
        """
        features = []
        
        wait_info = state.wait_info
        
        # Shanten (1) - normalized (-1 to 6 → 0 to 1)
        shanten_normalized = np.array([
            (wait_info.shanten + 1) / 7.0
        ], dtype=np.float32)
        features.append(shanten_normalized)
        
        # Is tenpai (1)
        is_tenpai = np.array([
            1.0 if wait_info.is_tenpai else 0.0
        ], dtype=np.float32)
        features.append(is_tenpai)
        
        # Wait count (1) - normalized
        wait_count_normalized = np.array([
            wait_info.wait_count / 13.0  # Max possible wait
        ], dtype=np.float32)
        features.append(wait_count_normalized)
        
        # Wait types (1) - normalized
        wait_types_normalized = np.array([
            wait_info.wait_types / 13.0  # Max possible types
        ], dtype=np.float32)
        features.append(wait_types_normalized)
        
        # Wait tiles (34) - multi-hot encoding
        wait_tiles = np.zeros(34, dtype=np.float32)
        for tile_type in wait_info.wait_tiles:
            if 0 <= tile_type < 34:
                wait_tiles[tile_type] = 1.0
        features.append(wait_tiles)
        
        return np.concatenate(features)
    
    def _encode_discard_meta_features(self, state: ComprehensiveGameState) -> np.ndarray:
        """Encode discard meta information (170).
        
        Features:
        - Tedashi/tsumogiri ratio per player per tile (4 × 34 = 136)
        - Danger score per tile (34)
        """
        features = []
        
        # Tedashi/tsumogiri ratio (4 × 34)
        for player_id in range(4):
            player_ratio = self._calculate_tedashi_ratio(state, player_id)
            features.append(player_ratio)
        
        # Danger score (34)
        danger_scores = self._calculate_danger_scores(state)
        features.append(danger_scores)
        
        return np.concatenate(features)
    
    def _calculate_tedashi_ratio(self, state: ComprehensiveGameState, 
                                 player_id: int) -> np.ndarray:
        """Calculate tedashi (hand discard) ratio for each tile type.
        
        Args:
            state: Game state
            player_id: Target player
            
        Returns:
            34-dimensional array with tedashi ratio (0=all tsumogiri, 1=all tedashi)
        """
        ratio = np.zeros(34, dtype=np.float32)
        
        # Get discards by this player
        player_discards = [
            d for d in state.discard_history if d.player_id == player_id
        ]
        
        # Count tedashi and total for each tile type
        tedashi_count = np.zeros(34, dtype=np.int32)
        total_count = np.zeros(34, dtype=np.int32)
        
        for discard in player_discards:
            tile_type = discard.tile_type
            if 0 <= tile_type < 34:
                total_count[tile_type] += 1
                if not discard.is_tsumogiri:
                    tedashi_count[tile_type] += 1
        
        # Calculate ratio
        for i in range(34):
            if total_count[i] > 0:
                ratio[i] = tedashi_count[i] / total_count[i]
        
        return ratio
    
    def _calculate_danger_scores(self, state: ComprehensiveGameState) -> np.ndarray:
        """Calculate danger score for each tile type.
        
        Args:
            state: Game state
            
        Returns:
            34-dimensional array with danger scores (0=safe, 1=dangerous)
        """
        danger_scores = np.zeros(34, dtype=np.float32)
        
        # Mark dangerous tiles
        for tile_type in state.dangerous_tiles:
            if 0 <= tile_type < 34:
                danger_scores[tile_type] = 1.0
        
        return danger_scores
    
    def _encode_discards_1x34(self, discards: List[DiscardHistoryEntry]) -> np.ndarray:
        """Encode discards to 1x34 format."""
        encoding = np.zeros(34, dtype=np.int32)
        for discard in discards:
            if 0 <= discard.tile_type < 34:
                encoding[discard.tile_type] += 1
        return encoding
    
    def _encode_melds_1x34(self, melds: List[MeldHistoryEntry]) -> np.ndarray:
        """Encode melds to 1x34 format."""
        encoding = np.zeros(34, dtype=np.int32)
        for meld in melds:
            for tile_type in meld.tile_types:
                if 0 <= tile_type < 34:
                    encoding[tile_type] += 1
        return encoding
    
    def _encode_dora_indicators(self, dora_indicators: List[int]) -> np.ndarray:
        """Encode dora indicators to 1x34 format."""
        encoding = np.zeros(34, dtype=np.int32)
        for dora_id in dora_indicators:
            tile_type = dora_id // 4
            if 0 <= tile_type < 34:
                encoding[tile_type] += 1
        return encoding
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability.
        
        Returns:
            List of feature names
        """
        names = []
        
        # Basic features
        tile_names = self._get_tile_names()
        groups = [
            'own_hand', 'own_melds', 'own_discards',
            'opp1_discards', 'opp2_discards', 'opp3_discards',
            'opp1_melds', 'opp2_melds', 'opp3_melds',
            'dora'
        ]
        for group in groups:
            for tile in tile_names:
                names.append(f'{group}_{tile}')
        
        # Draw sequence
        for i in range(self.draw_history_length):
            for tile in tile_names:
                names.append(f'draw_t-{self.draw_history_length-i}_{tile}')
        
        # Discard sequence
        for i in range(self.discard_history_length):
            for tile in tile_names:
                names.append(f'discard_t-{self.discard_history_length-i}_{tile}')
            for p in range(4):
                names.append(f'discard_t-{self.discard_history_length-i}_p{p}')
            names.append(f'discard_t-{self.discard_history_length-i}_riichi')
        
        # Meta features
        names.extend([
            'round_wind', 'round_index', 'dealer', 'honba', 'riichi_sticks',
            'turn', 'remaining_tiles', 'is_all_last',
            'is_dealer_p0', 'is_dealer_p1', 'is_dealer_p2', 'is_dealer_p3',
            'riichi_p0', 'riichi_p1', 'riichi_p2', 'riichi_p3',
            'riichi_turn_p0', 'riichi_turn_p1', 'riichi_turn_p2', 'riichi_turn_p3',
            'score_p0', 'score_p1', 'score_p2', 'score_p3',
            'score_diff', 'rank', 'turn_normalized', 'remaining_draws',
            'red_5m', 'red_5p', 'red_5s'
        ])
        
        # Wait features
        names.extend([
            'shanten', 'is_tenpai', 'wait_count', 'wait_types'
        ])
        for tile in tile_names:
            names.append(f'wait_{tile}')
        
        # Discard meta
        for p in range(4):
            for tile in tile_names:
                names.append(f'tedashi_ratio_p{p}_{tile}')
        for tile in tile_names:
            names.append(f'danger_{tile}')
        
        return names
    
    def _get_tile_names(self) -> List[str]:
        """Get tile names (0-33)."""
        names = []
        # Man (0-8)
        for i in range(9):
            names.append(f'{i+1}m')
        # Pin (9-17)
        for i in range(9):
            names.append(f'{i+1}p')
        # Sou (18-26)
        for i in range(9):
            names.append(f'{i+1}s')
        # Honors (27-33)
        honor_names = ['E', 'S', 'W', 'N', 'Haku', 'Hatsu', 'Chun']
        names.extend(honor_names)
        return names


class LabelEncoder:
    """ラベルのエンコーダー"""
    
    @staticmethod
    def encode_discard_label(tile_type: int, num_classes: int = 34) -> np.ndarray:
        """Encode discard label as one-hot vector.
        
        Args:
            tile_type: Tile type (0-33)
            num_classes: Number of classes
            
        Returns:
            One-hot vector
        """
        label = np.zeros(num_classes, dtype=np.float32)
        if 0 <= tile_type < num_classes:
            label[tile_type] = 1.0
        return label
    
    @staticmethod
    def encode_discard_label_id(tile_type: int) -> int:
        """Encode discard label as integer ID.
        
        Args:
            tile_type: Tile type (0-33)
            
        Returns:
            Integer label
        """
        return tile_type if 0 <= tile_type < 34 else 0

