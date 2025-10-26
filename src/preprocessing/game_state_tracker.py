"""Comprehensive Game State Tracker for Mahjong AI.

全ての局面情報を包括的に管理するモジュール。
各打牌タイミングでの完全なゲーム状態を提供。
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np
import logging

from .enhanced_parser import (
    EnhancedGame, EnhancedRoundState, EnhancedGameAction, EnhancedMeldInfo
)

logger = logging.getLogger(__name__)


@dataclass
class DrawHistoryEntry:
    """ツモ履歴のエントリー"""
    player_id: int
    tile: int
    tile_type: int
    turn: int
    is_red_dora: bool = False


@dataclass
class DiscardHistoryEntry:
    """捨て牌履歴のエントリー"""
    player_id: int
    tile: int
    tile_type: int
    turn: int
    is_tsumogiri: bool = False
    is_riichi_discard: bool = False
    is_red_dora: bool = False


@dataclass
class MeldHistoryEntry:
    """副露履歴のエントリー"""
    player_id: int
    meld_type: str  # 'chi', 'pon', 'kan', 'ankan', 'kakan'
    tiles: List[int]
    tile_types: List[int]
    from_who: int
    timing: int


@dataclass
class RiichiInfo:
    """リーチ情報"""
    player_id: int
    declared: bool = False
    declared_turn: Optional[int] = None
    declared_tile: Optional[int] = None


@dataclass
class WaitInfo:
    """待ち情報"""
    is_tenpai: bool = False
    shanten: int = 1
    wait_tiles: List[int] = field(default_factory=list)  # Tile types (0-33)
    wait_count: int = 0  # Total number of waiting tiles
    wait_types: int = 0  # Number of different tile types


@dataclass
class ComprehensiveGameState:
    """包括的なゲーム状態
    
    各打牌タイミングでの完全な情報を保持。
    """
    # ゲーム基本情報
    game_id: str
    round_num: int
    round_wind: int  # 0=East, 1=South
    round_index: int  # 0-3
    dealer: int
    honba: int
    riichi_sticks: int
    turn: int
    
    # プレイヤー情報（観測対象）
    player_id: int
    is_dealer: bool
    
    # 手牌・副露
    hand: List[int]  # Tile IDs (0-135)
    hand_types: np.ndarray  # 1x34 encoding
    melds: List[MeldHistoryEntry]
    meld_types: np.ndarray  # 1x34 encoding
    
    # ツモ履歴
    draw_history: List[DrawHistoryEntry]
    
    # 河（捨て牌）履歴
    discard_history: List[DiscardHistoryEntry]  # 全プレイヤー
    own_discards: List[DiscardHistoryEntry]
    opponent_discards: List[List[DiscardHistoryEntry]]  # 3 opponents
    
    # 副露履歴
    meld_history: List[MeldHistoryEntry]  # 全プレイヤー
    opponent_melds: List[List[MeldHistoryEntry]]  # 3 opponents
    
    # ドラ情報
    dora_indicators: List[int]
    visible_red_dora: Dict[str, int]
    
    # リーチ情報
    riichi_info: List[RiichiInfo]  # 4 players
    
    # 点数情報
    scores: List[int]  # 4 players (in 100s)
    score_diffs: List[int]  # Difference from top score
    rank: int  # Current rank (0=1st, 3=4th)
    
    # 待ち・向聴数情報
    wait_info: WaitInfo
    
    # 危険度情報
    dangerous_tiles: List[int]  # Tile types (0-33) that are dangerous
    
    # ラベル（教師データ）
    label_tile: int  # Actual discarded tile ID
    label_tile_type: int  # Actual discarded tile type (0-33)


class ComprehensiveGameStateTracker:
    """包括的なゲーム状態の追跡管理クラス"""
    
    def __init__(self, enable_shanten_calc: bool = True,
                 enable_danger_estimation: bool = True):
        """Initialize the tracker.
        
        Args:
            enable_shanten_calc: Enable shanten calculation (requires shanten module)
            enable_danger_estimation: Enable danger estimation for riichi
        """
        self.enable_shanten_calc = enable_shanten_calc
        self.enable_danger_estimation = enable_danger_estimation
        
        # Initialize shanten calculator if available
        if enable_shanten_calc:
            try:
                from ..utils.shanten import calculate_shanten
                self.shanten_calculator = calculate_shanten
            except ImportError:
                logger.warning("Shanten calculator not available, disabling shanten calculation")
                self.enable_shanten_calc = False
                self.shanten_calculator = None
        else:
            self.shanten_calculator = None
    
    def extract_all_states(self, game: EnhancedGame, 
                          target_player: Optional[int] = None) -> List[ComprehensiveGameState]:
        """Extract all game states from an EnhancedGame.
        
        Args:
            game: EnhancedGame object
            target_player: Extract states only for this player (None for all players)
            
        Returns:
            List of ComprehensiveGameState objects (one per discard decision)
        """
        all_states = []
        
        for round_idx, round_state in enumerate(game.rounds):
            # Get all actions in this round
            round_actions = [a for a in game.actions if a.round_num == round_idx]
            
            # Extract states for each discard action
            for action_idx, action in enumerate(round_actions):
                if action.action_type != 'discard':
                    continue
                
                player_id = action.player_id
                
                # Filter by target player if specified
                if target_player is not None and player_id != target_player:
                    continue
                
                # Get all actions before this discard
                prior_actions = round_actions[:action_idx]
                
                # Build comprehensive state
                try:
                    state = self._build_state_at_action(
                        game, round_state, round_idx, prior_actions, action
                    )
                    all_states.append(state)
                except Exception as e:
                    logger.warning(f"Failed to build state at round {round_idx}, action {action_idx}: {e}")
        
        return all_states
    
    def _build_state_at_action(self, game: EnhancedGame, round_state: EnhancedRoundState,
                              round_idx: int, prior_actions: List[EnhancedGameAction],
                              discard_action: EnhancedGameAction) -> ComprehensiveGameState:
        """Build complete game state at a specific discard action.
        
        Args:
            game: EnhancedGame object
            round_state: Round state
            round_idx: Round index
            prior_actions: All actions before this discard
            discard_action: The discard action
            
        Returns:
            ComprehensiveGameState object
        """
        player_id = discard_action.player_id
        
        # Reconstruct hand and melds
        hand, melds = self._reconstruct_hand_and_melds(
            round_state, prior_actions, player_id
        )
        
        # Encode hand and melds
        hand_types = self._encode_tiles_1x34(hand)
        meld_types = self._encode_melds_1x34(melds)
        
        # Extract draw history
        draw_history = self._extract_draw_history(prior_actions, player_id)
        
        # Extract discard history
        discard_history = self._extract_discard_history(prior_actions)
        own_discards = [d for d in discard_history if d.player_id == player_id]
        
        # Separate opponent discards (relative to player_id)
        opponent_discards = [[], [], []]
        opponent_ids = [(player_id + i + 1) % 4 for i in range(3)]
        for opp_idx, opp_id in enumerate(opponent_ids):
            opponent_discards[opp_idx] = [d for d in discard_history if d.player_id == opp_id]
        
        # Extract meld history
        meld_history = self._extract_meld_history(prior_actions)
        opponent_melds = [[], [], []]
        for opp_idx, opp_id in enumerate(opponent_ids):
            opponent_melds[opp_idx] = [m for m in meld_history if m.player_id == opp_id]
        
        # Extract riichi info
        riichi_info = self._extract_riichi_info(prior_actions)
        
        # Calculate wait info
        wait_info = self._calculate_wait_info(hand_types)
        
        # Estimate dangerous tiles
        dangerous_tiles = self._estimate_dangerous_tiles(riichi_info, discard_history)
        
        # Calculate scores and ranks
        scores = round_state.scores
        score_diffs = [scores[i] - max(scores) for i in range(4)]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        rank = [i for i, (idx, _) in enumerate(sorted_scores) if idx == player_id][0]
        
        # Build state
        state = ComprehensiveGameState(
            game_id=game.game_id,
            round_num=round_state.round_num,
            round_wind=round_state.round_wind,
            round_index=round_state.round_index,
            dealer=round_state.dealer,
            honba=round_state.honba,
            riichi_sticks=round_state.riichi_sticks,
            turn=discard_action.turn,
            player_id=player_id,
            is_dealer=(player_id == round_state.dealer),
            hand=hand,
            hand_types=hand_types,
            melds=melds,
            meld_types=meld_types,
            draw_history=draw_history,
            discard_history=discard_history,
            own_discards=own_discards,
            opponent_discards=opponent_discards,
            meld_history=meld_history,
            opponent_melds=opponent_melds,
            dora_indicators=round_state.dora_indicators,
            visible_red_dora=round_state.visible_red_dora,
            riichi_info=riichi_info,
            scores=scores,
            score_diffs=score_diffs,
            rank=rank,
            wait_info=wait_info,
            dangerous_tiles=dangerous_tiles,
            label_tile=discard_action.tile,
            label_tile_type=discard_action.tile_type
        )
        
        return state
    
    def _reconstruct_hand_and_melds(self, round_state: EnhancedRoundState,
                                   actions: List[EnhancedGameAction],
                                   player_id: int) -> Tuple[List[int], List[MeldHistoryEntry]]:
        """Reconstruct hand and melds for a player at a specific point.
        
        Args:
            round_state: Round state
            actions: Actions up to this point
            player_id: Player ID
            
        Returns:
            (hand, melds) tuple
        """
        # Start with initial hand
        hand = round_state.initial_hands[player_id].copy()
        melds = []
        
        # Apply all actions
        for action in actions:
            if action.player_id != player_id:
                continue
            
            if action.action_type == 'draw':
                hand.append(action.tile)
            
            elif action.action_type == 'discard':
                if action.tile in hand:
                    hand.remove(action.tile)
            
            elif action.action_type in ['chi', 'pon', 'kan', 'ankan', 'kakan']:
                if action.meld:
                    # Remove tiles from hand for meld
                    meld_tiles = action.meld.tiles
                    tile_types = [t // 4 for t in meld_tiles]
                    
                    # For chi/pon/kan, remove corresponding tiles from hand
                    for tile_type in tile_types:
                        for h_tile in hand[:]:
                            if h_tile // 4 == tile_type:
                                hand.remove(h_tile)
                                break
                    
                    meld_entry = MeldHistoryEntry(
                        player_id=player_id,
                        meld_type=action.meld.meld_type,
                        tiles=meld_tiles,
                        tile_types=tile_types,
                        from_who=action.meld.from_who,
                        timing=action.turn
                    )
                    melds.append(meld_entry)
        
        return hand, melds
    
    def _encode_tiles_1x34(self, tiles: List[int]) -> np.ndarray:
        """Encode tiles to 1x34 format.
        
        Args:
            tiles: List of tile IDs (0-135)
            
        Returns:
            34-dimensional array
        """
        encoding = np.zeros(34, dtype=np.int32)
        for tile in tiles:
            tile_type = tile // 4
            if 0 <= tile_type < 34:
                encoding[tile_type] += 1
        return encoding
    
    def _encode_melds_1x34(self, melds: List[MeldHistoryEntry]) -> np.ndarray:
        """Encode melds to 1x34 format.
        
        Args:
            melds: List of meld history entries
            
        Returns:
            34-dimensional array
        """
        encoding = np.zeros(34, dtype=np.int32)
        for meld in melds:
            for tile_type in meld.tile_types:
                if 0 <= tile_type < 34:
                    encoding[tile_type] += 1
        return encoding
    
    def _extract_draw_history(self, actions: List[EnhancedGameAction],
                             player_id: int) -> List[DrawHistoryEntry]:
        """Extract draw history for a player.
        
        Args:
            actions: All actions
            player_id: Target player
            
        Returns:
            List of DrawHistoryEntry
        """
        draw_history = []
        for action in actions:
            if action.action_type == 'draw' and action.player_id == player_id:
                draw_history.append(DrawHistoryEntry(
                    player_id=action.player_id,
                    tile=action.tile,
                    tile_type=action.tile_type,
                    turn=action.turn,
                    is_red_dora=action.is_red_dora
                ))
        return draw_history
    
    def _extract_discard_history(self, actions: List[EnhancedGameAction]) -> List[DiscardHistoryEntry]:
        """Extract discard history for all players.
        
        Args:
            actions: All actions
            
        Returns:
            List of DiscardHistoryEntry
        """
        discard_history = []
        for action in actions:
            if action.action_type == 'discard':
                discard_history.append(DiscardHistoryEntry(
                    player_id=action.player_id,
                    tile=action.tile,
                    tile_type=action.tile_type,
                    turn=action.turn,
                    is_tsumogiri=action.is_tsumogiri,
                    is_riichi_discard=action.is_riichi_discard,
                    is_red_dora=action.is_red_dora
                ))
        return discard_history
    
    def _extract_meld_history(self, actions: List[EnhancedGameAction]) -> List[MeldHistoryEntry]:
        """Extract meld history for all players.
        
        Args:
            actions: All actions
            
        Returns:
            List of MeldHistoryEntry
        """
        meld_history = []
        for action in actions:
            if action.action_type in ['chi', 'pon', 'kan', 'ankan', 'kakan']:
                if action.meld:
                    meld_history.append(MeldHistoryEntry(
                        player_id=action.player_id,
                        meld_type=action.meld.meld_type,
                        tiles=action.meld.tiles,
                        tile_types=[t // 4 for t in action.meld.tiles],
                        from_who=action.meld.from_who,
                        timing=action.turn
                    ))
        return meld_history
    
    def _extract_riichi_info(self, actions: List[EnhancedGameAction]) -> List[RiichiInfo]:
        """Extract riichi information for all players.
        
        Args:
            actions: All actions
            
        Returns:
            List of 4 RiichiInfo objects
        """
        riichi_info = [RiichiInfo(player_id=i) for i in range(4)]
        
        for action in actions:
            if action.action_type == 'riichi':
                info = riichi_info[action.player_id]
                info.declared = True
                info.declared_turn = action.riichi_turn
                
                # Find the next discard (riichi discard tile)
                for next_action in actions[actions.index(action) + 1:]:
                    if next_action.player_id == action.player_id and next_action.action_type == 'discard':
                        info.declared_tile = next_action.tile_type
                        break
        
        return riichi_info
    
    def _calculate_wait_info(self, hand_types: np.ndarray) -> WaitInfo:
        """Calculate waiting information (tenpai, shanten, etc.)
        
        Args:
            hand_types: 1x34 hand encoding
            
        Returns:
            WaitInfo object
        """
        wait_info = WaitInfo()
        
        if self.enable_shanten_calc and self.shanten_calculator:
            try:
                shanten = self.shanten_calculator(hand_types)
                wait_info.shanten = shanten
                wait_info.is_tenpai = (shanten == 0)
                
                # If tenpai, calculate wait tiles
                if wait_info.is_tenpai:
                    wait_tiles = self._calculate_wait_tiles(hand_types)
                    wait_info.wait_tiles = wait_tiles
                    wait_info.wait_types = len(wait_tiles)
                    
                    # Count total waiting tiles (4 - visible for each type)
                    total_wait = 0
                    for tile_type in wait_tiles:
                        available = 4 - hand_types[tile_type]
                        total_wait += available
                    wait_info.wait_count = total_wait
            
            except Exception as e:
                logger.warning(f"Shanten calculation failed: {e}")
        
        return wait_info
    
    def _calculate_wait_tiles(self, hand_types: np.ndarray) -> List[int]:
        """Calculate which tiles are waited for (simple implementation).
        
        Args:
            hand_types: 1x34 hand encoding
            
        Returns:
            List of tile types (0-33)
        """
        # Simple approach: try adding each tile and check if it completes the hand
        # This is a simplified version; full implementation would need proper win detection
        wait_tiles = []
        
        # For now, return empty list (requires full win detection logic)
        # TODO: Implement complete win detection
        
        return wait_tiles
    
    def _estimate_dangerous_tiles(self, riichi_info: List[RiichiInfo],
                                  discard_history: List[DiscardHistoryEntry]) -> List[int]:
        """Estimate dangerous tiles based on riichi declarations.
        
        Args:
            riichi_info: Riichi information for all players
            discard_history: Discard history
            
        Returns:
            List of tile types (0-33) that are dangerous
        """
        if not self.enable_danger_estimation:
            return []
        
        dangerous_tiles = set()
        
        for info in riichi_info:
            if not info.declared:
                continue
            
            # Get discards by this player before riichi
            player_discards = [
                d.tile_type for d in discard_history
                if d.player_id == info.player_id and d.turn < info.declared_turn
            ]
            
            # Add discarded tiles as safe
            safe_tiles = set(player_discards)
            
            # Tiles NOT discarded are potentially dangerous
            all_tiles = set(range(34))
            potentially_dangerous = all_tiles - safe_tiles
            
            # For number tiles, tiles near discarded tiles are also dangerous
            for tile_type in player_discards:
                if tile_type < 27:  # Number tile
                    suit_start = (tile_type // 9) * 9
                    suit_num = tile_type % 9
                    
                    # Add neighboring tiles as dangerous
                    if suit_num > 0:
                        potentially_dangerous.add(suit_start + suit_num - 1)
                    if suit_num < 8:
                        potentially_dangerous.add(suit_start + suit_num + 1)
                    if suit_num > 1:
                        potentially_dangerous.add(suit_start + suit_num - 2)
                    if suit_num < 7:
                        potentially_dangerous.add(suit_start + suit_num + 2)
            
            dangerous_tiles.update(potentially_dangerous)
        
        return list(dangerous_tiles)

