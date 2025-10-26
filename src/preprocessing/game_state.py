"""ゲーム状態管理モジュール

局の初めから全員の打牌履歴を保持し、時系列情報を管理します。
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DiscardHistory:
    """打牌履歴"""
    player_id: int
    tile: int  # Tile ID (0-135)
    tile_type: int  # Tile type (0-33)
    turn: int
    is_riichi: bool = False
    is_tsumogiri: bool = False  # ツモ切りかどうか


@dataclass
class RoundGameState:
    """局の全体状態"""
    round_num: int
    dealer: int
    honba: int
    riichi_sticks: int
    dora_indicators: List[int]
    
    # 各プレイヤーの手牌（見えている部分のみ）
    hands: List[List[int]] = field(default_factory=lambda: [[], [], [], []])
    
    # 各プレイヤーの副露
    melds: List[List[Tuple[str, List[int]]]] = field(default_factory=lambda: [[], [], [], []])
    
    # 全員の打牌履歴（時系列順）
    discard_history: List[DiscardHistory] = field(default_factory=list)
    
    # 各プレイヤーの捨て牌（順序維持）
    discards_by_player: List[List[int]] = field(default_factory=lambda: [[], [], [], []])
    
    # 各プレイヤーの点数
    scores: List[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    
    # リーチ宣言状態
    riichi_declared: List[bool] = field(default_factory=lambda: [False, False, False, False])
    
    def add_discard(self, player_id: int, tile: int, turn: int, 
                   is_riichi: bool = False, is_tsumogiri: bool = False):
        """打牌を追加
        
        Args:
            player_id: プレイヤーID (0-3)
            tile: 打牌ID (0-135)
            turn: ターン数
            is_riichi: リーチ宣言の打牌か
            is_tsumogiri: ツモ切りか
        """
        tile_type = tile // 4
        
        discard = DiscardHistory(
            player_id=player_id,
            tile=tile,
            tile_type=tile_type,
            turn=turn,
            is_riichi=is_riichi,
            is_tsumogiri=is_tsumogiri
        )
        
        self.discard_history.append(discard)
        self.discards_by_player[player_id].append(tile)
        
        if is_riichi:
            self.riichi_declared[player_id] = True
    
    def get_recent_discards(self, num_turns: int = 4) -> List[DiscardHistory]:
        """直近N手の打牌を取得
        
        Args:
            num_turns: 取得する手数
            
        Returns:
            直近の打牌リスト
        """
        return self.discard_history[-num_turns:] if len(self.discard_history) > 0 else []
    
    def get_player_discards(self, player_id: int) -> List[int]:
        """特定プレイヤーの全打牌を取得
        
        Args:
            player_id: プレイヤーID
            
        Returns:
            打牌リスト
        """
        return self.discards_by_player[player_id]
    
    def get_all_visible_tiles(self, player_id: int) -> List[int]:
        """特定プレイヤーから見える全ての牌を取得
        
        Args:
            player_id: プレイヤーID
            
        Returns:
            見える牌のリスト
        """
        visible_tiles = []
        
        # 自分の手牌
        visible_tiles.extend(self.hands[player_id])
        
        # 自分の副露
        for meld_type, tiles in self.melds[player_id]:
            visible_tiles.extend(tiles)
        
        # 全員の捨て牌
        for discards in self.discards_by_player:
            visible_tiles.extend(discards)
        
        # 全員の副露
        for pid in range(4):
            if pid != player_id:
                for meld_type, tiles in self.melds[pid]:
                    visible_tiles.extend(tiles)
        
        # ドラ表示牌
        visible_tiles.extend(self.dora_indicators)
        
        return visible_tiles
    
    def encode_discard_history_sequence(self, max_length: int = 64) -> np.ndarray:
        """打牌履歴を時系列エンコーディング
        
        Args:
            max_length: 最大系列長
            
        Returns:
            エンコードされた系列 (max_length, 34+4+1)
            - 34次元: 打牌タイルのone-hot
            - 4次元: プレイヤーID one-hot
            - 1次元: リーチフラグ
        """
        feature_dim = 34 + 4 + 1  # tile + player + riichi
        sequence = np.zeros((max_length, feature_dim), dtype=np.float32)
        
        # 直近max_length分の打牌を取得
        recent_discards = self.discard_history[-max_length:] if len(self.discard_history) > 0 else []
        
        for i, discard in enumerate(recent_discards):
            # 打牌タイル (one-hot)
            sequence[i, discard.tile_type] = 1.0
            
            # プレイヤーID (one-hot)
            sequence[i, 34 + discard.player_id] = 1.0
            
            # リーチフラグ
            sequence[i, 38] = 1.0 if discard.is_riichi else 0.0
        
        return sequence
    
    def encode_discard_history_tiles_only(self, max_length: int = 64) -> np.ndarray:
        """打牌履歴（牌のみ）を時系列エンコーディング
        
        Args:
            max_length: 最大系列長
            
        Returns:
            エンコードされた系列 (max_length, 34)
        """
        sequence = np.zeros((max_length, 34), dtype=np.float32)
        
        recent_discards = self.discard_history[-max_length:] if len(self.discard_history) > 0 else []
        
        for i, discard in enumerate(recent_discards):
            sequence[i, discard.tile_type] = 1.0
        
        return sequence
    
    def get_discard_counts_by_player(self) -> np.ndarray:
        """各プレイヤーの打牌枚数カウント (34種類)
        
        Returns:
            (4, 34) の配列
        """
        counts = np.zeros((4, 34), dtype=np.int32)
        
        for player_id in range(4):
            for tile in self.discards_by_player[player_id]:
                tile_type = tile // 4
                if 0 <= tile_type < 34:
                    counts[player_id, tile_type] += 1
        
        return counts
    
    def get_dangerous_tiles(self, player_id: int) -> List[int]:
        """危険牌を推定（リーチ者の捨て牌の周辺など）
        
        Args:
            player_id: 対象プレイヤー（リーチ者）
            
        Returns:
            危険牌タイプのリスト
        """
        if not self.riichi_declared[player_id]:
            return []
        
        dangerous_tiles = set()
        player_discards = self.discards_by_player[player_id]
        
        for tile in player_discards:
            tile_type = tile // 4
            
            # 同じ牌
            dangerous_tiles.add(tile_type)
            
            # 数牌の場合、前後の牌も危険
            if tile_type < 27:  # 数牌
                suit_start = (tile_type // 9) * 9
                suit_num = tile_type % 9
                
                if suit_num > 0:
                    dangerous_tiles.add(suit_start + suit_num - 1)
                if suit_num < 8:
                    dangerous_tiles.add(suit_start + suit_num + 1)
        
        return list(dangerous_tiles)


class GameStateManager:
    """ゲーム状態の管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.current_round: Optional[RoundGameState] = None
        self.round_history: List[RoundGameState] = []
    
    def start_new_round(self, round_num: int, dealer: int, honba: int = 0,
                       riichi_sticks: int = 0, dora_indicators: List[int] = None,
                       initial_hands: List[List[int]] = None,
                       scores: List[int] = None):
        """新しい局を開始
        
        Args:
            round_num: 局数
            dealer: 親プレイヤーID
            honba: 本場数
            riichi_sticks: 供託棒
            dora_indicators: ドラ表示牌
            initial_hands: 初期手牌
            scores: 点数
        """
        self.current_round = RoundGameState(
            round_num=round_num,
            dealer=dealer,
            honba=honba,
            riichi_sticks=riichi_sticks,
            dora_indicators=dora_indicators or [],
            hands=initial_hands or [[], [], [], []],
            scores=scores or [25000, 25000, 25000, 25000]
        )
    
    def end_current_round(self):
        """現在の局を終了"""
        if self.current_round:
            self.round_history.append(self.current_round)
            self.current_round = None
    
    def add_discard(self, player_id: int, tile: int, turn: int,
                   is_riichi: bool = False, is_tsumogiri: bool = False):
        """打牌を記録"""
        if self.current_round:
            self.current_round.add_discard(player_id, tile, turn, is_riichi, is_tsumogiri)
    
    def get_current_state(self) -> Optional[RoundGameState]:
        """現在の局の状態を取得"""
        return self.current_round
    
    def get_round_history(self) -> List[RoundGameState]:
        """全局の履歴を取得"""
        return self.round_history


