"""Advanced Game State Management

高度なゲーム状態管理モジュール。ver_3.0.0の優れた実装を参考に、
麻雀AIのための完全なゲーム状態追跡を実装。

Features:
- 手牌の正確な追跡
- 鳴きの正確な処理（チー・ポン・カン）
- リーチ状態の管理
- ドラ表示牌の追跡
- イベント履歴の記録
"""

import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


# Constants
NUM_PLAYERS = 4
NUM_TILE_TYPES = 34  # 0-33 for different tile kinds
MAX_EVENT_HISTORY = 60  # Max sequence length for event history

# Event types
EVENT_TYPES = {
    "INIT": 0, "TSUMO": 1, "DISCARD": 2, "N": 3, "REACH": 4,
    "DORA": 5, "AGARI": 6, "RYUUKYOKU": 7, "PADDING": 8
}

# Naki types
NAKI_TYPES = {"チー": 0, "ポン": 1, "大明槓": 2, "加槓": 3, "暗槓": 4, "不明": -1}


def tile_id_to_index(tile: int) -> int:
    """牌ID（0～135）を牌種インデックス（0～33）に変換"""
    if not isinstance(tile, int) or not (0 <= tile <= 135):
        return -1
    if tile < 108:  # Man, Pin, Sou
        suit_index = tile // 36  # 0: Man, 1: Pin, 2: Sou
        number_index = (tile % 36) // 4  # 0-8 for 1-9
        return suit_index * 9 + number_index
    else:  # Honors
        return 27 + (tile - 108) // 4


def tile_id_to_string(tile: int) -> str:
    """牌ID（0～135）を文字列表現に変換"""
    if not isinstance(tile, int) or not (0 <= tile <= 135):
        return "?"
    
    # Red fives
    if tile == 16: return "0m"
    if tile == 52: return "0p"
    if tile == 88: return "0s"
    
    suits = ["m", "p", "s"]
    honors = ["東", "南", "西", "北", "白", "發", "中"]
    
    if tile < 108:  # Number tiles
        suit_index = tile // 36
        number_index = (tile % 36) // 4
        return f"{number_index + 1}{suits[suit_index]}"
    else:  # Honors
        honor_index = (tile - 108) // 4
        return honors[honor_index] if honor_index < len(honors) else "?"


def decode_naki(m: int) -> Dict[str, Any]:
    """天鳳の副露面子のビットフィールド(m)をデコード
    
    Args:
        m: <N>タグの m 属性の値
        
    Returns:
        Dict with keys: type, tiles, from_who_relative, consumed, raw_value
    """
    result = {
        "type": "不明",
        "tiles": [],
        "from_who_relative": -1,
        "consumed": [],
        "raw_value": m
    }
    
    try:
        from_who_relative = m & 3
        result["from_who_relative"] = from_who_relative
        
        # チー (Chi)
        if m & (1 << 2):
            result["type"] = "チー"
            t = m >> 10
            r = t % 3
            t //= 3
            
            # Base index calculation
            if 0 <= t <= 6:
                base_index = t  # 1m-7m
            elif 7 <= t <= 13:
                base_index = (t - 7) + 9  # 1p-7p
            elif 14 <= t <= 20:
                base_index = (t - 14) + 18  # 1s-7s
            else:
                return result
            
            offsets = [(m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3]
            tiles = []
            for i in range(3):
                tile_kind = base_index + i
                tile_id = tile_kind * 4 + offsets[i]
                tiles.append(tile_id)
            
            result["tiles"] = sorted(tiles)
            result["called_position"] = r
        
        # ポン (Pon)
        elif m & (1 << 3):
            result["type"] = "ポン"
            t = m >> 9
            t //= 3
            
            if not (0 <= t <= 33):
                return result
            
            base_id = t * 4
            unused_offset = (m >> 5) & 3
            
            tiles = []
            for i in range(4):
                if i != unused_offset:
                    tiles.append(base_id + i)
            
            result["tiles"] = sorted(tiles)
        
        # 加槓 (Kakan)
        elif m & (1 << 4):
            result["type"] = "加槓"
            result["from_who_relative"] = -1
            t = m >> 9
            t //= 3
            
            if not (0 <= t <= 33):
                return result
            
            base_id = t * 4
            result["tiles"] = sorted([base_id, base_id + 1, base_id + 2, base_id + 3])
        
        # 暗槓 or 大明槓
        else:
            tile_id_raw = m >> 8
            tile_index = tile_id_raw // 4
            
            if not (0 <= tile_index <= 33):
                return result
            
            base_id = tile_index * 4
            result["tiles"] = sorted([base_id, base_id + 1, base_id + 2, base_id + 3])
            
            if from_who_relative != 3:
                result["type"] = "大明槓"
            else:
                result["type"] = "暗槓"
                result["from_who_relative"] = -1
        
        return result
    
    except Exception as e:
        logger.error(f"decode_naki failed for m={m}: {e}")
        return result


class AdvancedGameState:
    """高度なゲーム状態管理クラス
    
    ver_3.0.0の優れた実装を参考に、完全なゲーム状態追跡を実装。
    手牌、鳴き、打牌、リーチなどを正確に管理。
    """
    
    TSUMO_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
    DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}
    
    def __init__(self):
        """初期化"""
        self.reset_state()
    
    def reset_state(self):
        """全ての内部状態をリセット"""
        self.round_index: int = 0
        self.round_num_wind: int = 0
        self.honba: int = 0
        self.kyotaku: int = 0
        self.dealer: int = -1
        self.initial_scores: List[int] = [25000] * NUM_PLAYERS
        self.dora_indicators: List[int] = []
        self.current_scores: List[int] = [25000] * NUM_PLAYERS
        
        # 各プレイヤーの手牌 (tile IDs)
        self.player_hands: List[List[int]] = [[] for _ in range(NUM_PLAYERS)]
        
        # 各プレイヤーの打牌 [(tile_id, tsumogiri), ...]
        self.player_discards: List[List[Tuple[int, bool]]] = [[] for _ in range(NUM_PLAYERS)]
        
        # 各プレイヤーの副露
        self.player_melds: List[List[Dict]] = [[] for _ in range(NUM_PLAYERS)]
        
        # リーチ状態 (0: not reached, 1: declared but not discarded, 2: reached)
        self.player_reach_status: List[int] = [0] * NUM_PLAYERS
        self.player_reach_junme: List[float] = [-1.0] * NUM_PLAYERS
        self.player_reach_discard_index: List[int] = [-1] * NUM_PLAYERS
        
        # 現在の状態
        self.current_player: int = -1
        self.junme: float = 0.0
        
        # 最後の打牌情報（鳴きで使用）
        self.last_discard_event_player: int = -1
        self.last_discard_event_tile_id: int = -1
        self.last_discard_event_tsumogiri: bool = False
        
        # フラグ
        self.can_ron: bool = False
        self.naki_occurred_in_turn: bool = False
        self.is_rinshan: bool = False
        
        # イベント履歴
        self.event_history: deque = deque(maxlen=MAX_EVENT_HISTORY)
        
        # 山の残り枚数
        self.wall_tile_count: int = 70
    
    def _add_event(self, event_type: str, player: int, tile: int = -1, data: Dict = None):
        """イベント履歴にイベントを追加"""
        if data is None:
            data = {}
        
        event_code = EVENT_TYPES.get(event_type, -1)
        if event_code == -1:
            return
        
        event_info = {
            "type": event_code,
            "player": player,
            "tile_index": tile_id_to_index(tile),
            "junme": int(np.ceil(self.junme)),
            "data": data
        }
        self.event_history.append(event_info)
    
    def _sort_hand(self, player_id: int):
        """指定プレイヤーの手牌をソート"""
        if 0 <= player_id < NUM_PLAYERS:
            self.player_hands[player_id].sort(key=lambda t: (tile_id_to_index(t), t))
    
    def init_round(self, round_data: Dict):
        """新しい局を初期化
        
        Args:
            round_data: parsed XMLから取得したラウンドデータ
        """
        self.reset_state()
        init_info = round_data.get("init", {})
        if not init_info:
            logger.error("No 'init' info found in round_data")
            return
        
        self.round_index = round_data.get("round_index", 0)
        
        # Seed情報の解析
        seed_parts = init_info.get("seed", "0,0,0,0,0,0").split(",")
        try:
            if len(seed_parts) >= 6:
                self.round_num_wind = int(seed_parts[0])
                self.honba = int(seed_parts[1])
                self.kyotaku = int(seed_parts[2])
                dora_indicator_id = int(seed_parts[5])
                if 0 <= dora_indicator_id <= 135:
                    self.dora_indicators = [dora_indicator_id]
                    self._add_event("DORA", player=-1, tile=dora_indicator_id)
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse seed: {e}")
        
        # 親の設定
        self.dealer = int(init_info.get("oya", 0))
        if not (0 <= self.dealer < NUM_PLAYERS):
            self.dealer = 0
        self.current_player = self.dealer
        
        # 点数の設定
        try:
            raw_scores = init_info.get("ten", "250,250,250,250").split(",")
            if len(raw_scores) == 4:
                self.initial_scores = [int(float(s)) * 100 for s in raw_scores]
                self.current_scores = list(self.initial_scores)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse scores: {e}")
        
        # 配牌の設定
        for p in range(NUM_PLAYERS):
            hand_str = init_info.get(f"hai{p}", "")
            self.player_hands[p] = []
            try:
                if hand_str:
                    hand_ids = [int(h) for h in hand_str.split(',') if h.strip()]
                    valid_hand_ids = [tid for tid in hand_ids if 0 <= tid <= 135]
                    self.player_hands[p] = valid_hand_ids
                    self._sort_hand(p)
            except ValueError as e:
                logger.warning(f"Failed to parse initial hand for player {p}: {e}")
        
        # 山の残り枚数を計算
        initial_hand_sum = sum(len(h) for h in self.player_hands)
        self.wall_tile_count = 136 - 14 - initial_hand_sum
        self.junme = 0.0
        
        # INITイベントを追加
        init_data = {"round": self.round_num_wind, "honba": self.honba, "kyotaku": self.kyotaku}
        self._add_event("INIT", player=self.dealer, data=init_data)
    
    def process_tsumo(self, player_id: int, tile_id: int):
        """ツモを処理
        
        Args:
            player_id: プレイヤーID
            tile_id: ツモった牌のID
        """
        if not (0 <= player_id < NUM_PLAYERS):
            logger.error(f"Invalid player_id {player_id}")
            return
        if not (0 <= tile_id <= 135):
            logger.error(f"Invalid tile_id {tile_id}")
            return
        
        self.current_player = player_id
        
        # 順目の更新
        is_first_round = self.junme < 1.0
        is_dealer_turn = player_id == self.dealer
        
        if not self.is_rinshan:
            if is_first_round and is_dealer_turn and self.junme == 0.0:
                self.junme = 0.1
            elif not is_first_round and player_id == 0:
                self.junme = np.floor(self.junme) + 1.0
            elif is_first_round and not is_dealer_turn and self.junme < 1.0:
                if self.junme == 0.1:
                    self.junme = 1.0
        
        rinshan_draw = self.is_rinshan
        if rinshan_draw:
            self.is_rinshan = False
        else:
            if self.wall_tile_count > 0:
                self.wall_tile_count -= 1
        
        self.naki_occurred_in_turn = False
        self.player_hands[player_id].append(tile_id)
        self._sort_hand(player_id)
        
        tsumo_data = {"rinshan": rinshan_draw}
        self._add_event("TSUMO", player=player_id, tile=tile_id, data=tsumo_data)
        self.can_ron = False
    
    def process_discard(self, player_id: int, tile_id: int, tsumogiri: bool):
        """打牌を処理
        
        Args:
            player_id: プレイヤーID
            tile_id: 捨てた牌のID
            tsumogiri: ツモ切りかどうか
        """
        reach_declared = self.player_reach_status[player_id] == 1
        
        if not (0 <= player_id < NUM_PLAYERS):
            logger.error(f"Invalid player_id {player_id}")
            return
        if not (0 <= tile_id <= 135):
            logger.error(f"Invalid tile_id {tile_id}")
            return
        
        # 手牌から削除
        if tile_id in self.player_hands[player_id]:
            self.player_hands[player_id].remove(tile_id)
            self._sort_hand(player_id)
        else:
            # 同じ種類の牌を探して削除
            discard_tile_index = tile_id_to_index(tile_id)
            found_similar = False
            
            if discard_tile_index != -1:
                for hand_tile_id in list(self.player_hands[player_id]):
                    if tile_id_to_index(hand_tile_id) == discard_tile_index:
                        self.player_hands[player_id].remove(hand_tile_id)
                        self._sort_hand(player_id)
                        found_similar = True
                        logger.info(f"P{player_id}: Used similar tile {hand_tile_id} for discard {tile_id}")
                        break
            
            if not found_similar:
                logger.warning(f"P{player_id} discarding {tile_id_to_string(tile_id)} not found in hand")
        
        # 打牌を記録
        self.player_discards[player_id].append((tile_id, tsumogiri))
        
        discard_data = {"tsumogiri": int(tsumogiri)}
        self._add_event("DISCARD", player=player_id, tile=tile_id, data=discard_data)
        
        # 最後の打牌情報を更新
        self.last_discard_event_player = player_id
        self.last_discard_event_tile_id = tile_id
        self.last_discard_event_tsumogiri = tsumogiri
        self.can_ron = True
        
        # リーチ確定処理
        if reach_declared:
            self.player_reach_status[player_id] = 2
            self.player_reach_discard_index[player_id] = len(self.player_discards[player_id]) - 1
            self.player_reach_junme[player_id] = self.junme
            if self.current_scores[player_id] >= 1000:
                self.kyotaku += 1
                self.current_scores[player_id] -= 1000
            
            reach_data = {"step": 2, "junme": int(np.ceil(self.junme))}
            self._add_event("REACH", player=player_id, data=reach_data)
    
    def process_naki(self, naki_player_id: int, meld_code: int):
        """鳴きを処理
        
        Args:
            naki_player_id: 鳴いたプレイヤーID
            meld_code: 鳴きコード
        """
        if not (0 <= naki_player_id < NUM_PLAYERS):
            logger.error(f"Invalid naki_player_id {naki_player_id}")
            return
        
        naki_info = decode_naki(meld_code)
        naki_type = naki_info.get("type", "不明")
        decoded_tiles = naki_info.get("tiles", [])
        from_who_relative = naki_info.get("from_who_relative", -1)
        
        if naki_type == "不明":
            logger.warning(f"decode_naki failed for m={meld_code}")
            return
        
        from_who_player_abs = -1
        called_tile_id = -1
        tiles_to_remove = []
        actual_meld_tiles = []
        
        # 鳴きの種類に応じて処理
        if naki_type in ["チー", "ポン", "大明槓"]:
            called_tile_id = self.last_discard_event_tile_id
            discarder_player_id = self.last_discard_event_player
            
            if discarder_player_id == -1 or called_tile_id == -1:
                logger.warning(f"Naki {naki_type} without preceding discard")
                return
            
            from_who_player_abs = discarder_player_id
            called_tile_kind = tile_id_to_index(called_tile_id)
            
            if called_tile_kind == -1:
                logger.error(f"Invalid called_tile_id {called_tile_id}")
                return
            
            # チーの処理
            if naki_type == "チー":
                is_from_kamicha = (discarder_player_id - naki_player_id + NUM_PLAYERS) % NUM_PLAYERS == 3
                if not is_from_kamicha:
                    logger.warning(f"Chi from non-kamicha player")
                    return
                
                # 順子を探す
                possible_seqs = []
                if called_tile_kind % 9 <= 6:
                    possible_seqs.append([called_tile_kind, called_tile_kind + 1, called_tile_kind + 2])
                if called_tile_kind % 9 >= 1 and called_tile_kind % 9 <= 7:
                    possible_seqs.append([called_tile_kind - 1, called_tile_kind, called_tile_kind + 1])
                if called_tile_kind % 9 >= 2:
                    possible_seqs.append([called_tile_kind - 2, called_tile_kind - 1, called_tile_kind])
                
                found_seq = False
                player_hand_kinds = defaultdict(list)
                for tid in self.player_hands[naki_player_id]:
                    player_hand_kinds[tile_id_to_index(tid)].append(tid)
                
                for seq_kinds in possible_seqs:
                    if not all(k // 9 == called_tile_kind // 9 for k in seq_kinds):
                        continue
                    needed_kinds = [k for k in seq_kinds if k != called_tile_kind]
                    if len(needed_kinds) == 2 and player_hand_kinds[needed_kinds[0]] and player_hand_kinds[needed_kinds[1]]:
                        tile1_id = player_hand_kinds[needed_kinds[0]].pop(0)
                        tile2_id = player_hand_kinds[needed_kinds[1]].pop(0)
                        tiles_to_remove = [tile1_id, tile2_id]
                        actual_meld_tiles = sorted([called_tile_id] + tiles_to_remove)
                        found_seq = True
                        break
                
                if not found_seq:
                    logger.warning(f"Chi: No valid sequence found")
                    return
            
            # ポン・大明槓の処理
            elif naki_type in ["ポン", "大明槓"]:
                needed_count = 2 if naki_type == "ポン" else 3
                matching_tiles = [tid for tid in self.player_hands[naki_player_id] 
                                if tile_id_to_index(tid) == called_tile_kind]
                if len(matching_tiles) < needed_count:
                    logger.warning(f"{naki_type}: Need {needed_count}, have {len(matching_tiles)}")
                    return
                
                tiles_to_remove = matching_tiles[:needed_count]
                actual_meld_tiles = sorted([called_tile_id] + tiles_to_remove)
                
                if naki_type == "大明槓":
                    self.is_rinshan = True
        
        # 加槓の処理
        elif naki_type == "加槓":
            t = meld_code >> 9
            t //= 3
            kakan_tile_kind = t if 0 <= t <= 33 else -1
            
            if kakan_tile_kind == -1:
                logger.error(f"Kakan: Invalid tile kind")
                return
            
            possible_ids = [tid for tid in self.player_hands[naki_player_id] 
                          if tile_id_to_index(tid) == kakan_tile_kind]
            if not possible_ids:
                logger.error(f"Kakan: Tile not found in hand")
                return
            
            tiles_to_remove = [possible_ids[0]]
            from_who_player_abs = naki_player_id
            called_tile_id = -1
            self.is_rinshan = True
        
        # 暗槓の処理
        elif naki_type == "暗槓":
            tile_id_raw = meld_code >> 8
            ankan_tile_kind = tile_id_raw // 4
            
            if not (0 <= ankan_tile_kind <= 33):
                logger.error(f"Ankan: Invalid tile kind")
                return
            
            matching_tiles = [tid for tid in self.player_hands[naki_player_id] 
                            if tile_id_to_index(tid) == ankan_tile_kind]
            if len(matching_tiles) < 4:
                logger.error(f"Ankan: Need 4, have {len(matching_tiles)}")
                return
            
            tiles_to_remove = matching_tiles[:4]
            actual_meld_tiles = sorted(tiles_to_remove)
            from_who_player_abs = naki_player_id
            called_tile_id = -1
            self.is_rinshan = True
        
        # 手牌から牌を削除
        removed_count = 0
        for tile_to_remove in tiles_to_remove:
            if tile_to_remove in self.player_hands[naki_player_id]:
                self.player_hands[naki_player_id].remove(tile_to_remove)
                removed_count += 1
            else:
                logger.error(f"Naki: Could not find tile {tile_id_to_string(tile_to_remove)} to remove")
                return
        
        if removed_count == len(tiles_to_remove):
            self._sort_hand(naki_player_id)
            
            # 加槓の場合は既存のポンを更新
            if naki_type == "加槓":
                updated = False
                pon_index = tile_id_to_index(tiles_to_remove[0])
                for i, existing_meld in enumerate(self.player_melds[naki_player_id]):
                    if (existing_meld['type'] == "ポン" and 
                        tile_id_to_index(existing_meld['tiles'][0]) == pon_index):
                        actual_meld_tiles = sorted(existing_meld['tiles'] + tiles_to_remove)
                        self.player_melds[naki_player_id][i]['type'] = "加槓"
                        self.player_melds[naki_player_id][i]['tiles'] = actual_meld_tiles
                        self.player_melds[naki_player_id][i]['jun'] = self.junme
                        updated = True
                        break
                
                if not updated:
                    logger.error("Kakan: Corresponding Pon not found")
            else:
                # 新しい面子を追加
                new_meld = {
                    'type': naki_type,
                    'tiles': actual_meld_tiles,
                    'from_who': from_who_player_abs,
                    'called_tile': called_tile_id,
                    'm': meld_code,
                    'jun': self.junme
                }
                self.player_melds[naki_player_id].append(new_meld)
            
            # 状態を更新
            self.current_player = naki_player_id
            self.naki_occurred_in_turn = True
            self.can_ron = False
            self.last_discard_event_player = -1
            self.last_discard_event_tile_id = -1
            self.last_discard_event_tsumogiri = False
            
            # イベントを追加
            naki_event_data = {
                "naki_type": NAKI_TYPES.get(naki_type, -1),
                "from_who": from_who_player_abs
            }
            event_tile = called_tile_id if called_tile_id != -1 else actual_meld_tiles[0]
            self._add_event("N", player=naki_player_id, tile=event_tile, data=naki_event_data)
    
    def process_reach(self, player_id: int, step: int):
        """リーチを処理
        
        Args:
            player_id: プレイヤーID
            step: リーチステップ (1: 宣言, 2: 確定)
        """
        if not (0 <= player_id < NUM_PLAYERS):
            logger.error(f"Invalid player_id {player_id}")
            return
        
        if step == 1:
            if self.player_reach_status[player_id] != 0:
                return
            if self.current_scores[player_id] < 1000:
                return
            
            self.player_reach_status[player_id] = 1
            reach_data = {"step": 1}
            self._add_event("REACH", player=player_id, data=reach_data)
    
    def process_dora(self, tile_id: int):
        """ドラ表示牌を追加
        
        Args:
            tile_id: ドラ表示牌のID
        """
        if not (0 <= tile_id <= 135):
            logger.error(f"Invalid dora tile_id {tile_id}")
            return
        
        self.dora_indicators.append(tile_id)
        self._add_event("DORA", player=-1, tile=tile_id)
    
    def process_event(self, event_xml: Dict):
        """XMLイベントを解析して適切な処理メソッドを呼び出す
        
        Args:
            event_xml: {"tag": str, "attrib": dict} 形式のイベント
        """
        tag = event_xml.get("tag", "")
        attrib = event_xml.get("attrib", {})
        
        # ツモイベント
        for t_tag, p_id in self.TSUMO_TAGS.items():
            if tag.startswith(t_tag) and len(tag) > 1 and tag[1:].isdigit():
                try:
                    tsumo_pai_id = int(tag[1:])
                    self.process_tsumo(p_id, tsumo_pai_id)
                    return
                except (ValueError, IndexError):
                    continue
        
        # 捨て牌イベント
        for d_tag, p_id in self.DISCARD_TAGS.items():
            if tag.startswith(d_tag) and len(tag) > 1 and tag[1:].isdigit():
                try:
                    discard_pai_id = int(tag[1:])
                    tsumogiri = tag[0].islower()
                    self.process_discard(p_id, discard_pai_id, tsumogiri)
                    return
                except (ValueError, IndexError):
                    continue
        
        # 鳴きイベント
        if tag == "N":
            try:
                naki_player_id = int(attrib.get("who", -1))
                meld_code = int(attrib.get("m", "0"))
                if naki_player_id != -1:
                    self.process_naki(naki_player_id, meld_code)
                    return
            except (ValueError, KeyError):
                logger.warning(f"Failed to process N event: {attrib}")
        
        # リーチイベント
        if tag == "REACH":
            try:
                reach_player_id = int(attrib.get("who", -1))
                step = int(attrib.get("step", 0))
                if reach_player_id != -1:
                    self.process_reach(reach_player_id, step)
                    return
            except (ValueError, KeyError):
                logger.warning(f"Failed to process REACH event: {attrib}")
        
        # ドラ表示イベント
        if tag == "DORA":
            try:
                hai = int(attrib.get("hai", -1))
                if hai != -1:
                    self.process_dora(hai)
                    return
            except (ValueError, KeyError):
                logger.warning(f"Failed to process DORA event: {attrib}")
    
    def get_hand_indices(self, player_id: int) -> List[int]:
        """指定プレイヤーの手牌を牌種インデックスのリストで取得
        
        Args:
            player_id: プレイヤーID
            
        Returns:
            牌種インデックスのリスト
        """
        if 0 <= player_id < NUM_PLAYERS:
            return [tile_id_to_index(t) for t in self.player_hands[player_id] 
                   if tile_id_to_index(t) != -1]
        return []
    
    def get_visible_tiles(self, player_id: int) -> List[int]:
        """指定プレイヤーから見える全ての牌を取得
        
        Args:
            player_id: プレイヤーID
            
        Returns:
            見える牌のリスト (tile IDs)
        """
        visible = []
        
        # 自分の手牌
        visible.extend(self.player_hands[player_id])
        
        # 全員の打牌
        for p in range(NUM_PLAYERS):
            for tile_id, _ in self.player_discards[p]:
                visible.append(tile_id)
        
        # 全員の副露
        for p in range(NUM_PLAYERS):
            for meld in self.player_melds[p]:
                visible.extend(meld.get("tiles", []))
        
        # ドラ表示牌
        visible.extend(self.dora_indicators)
        
        return visible
    
    def get_state_dict(self) -> Dict[str, Any]:
        """現在の状態を辞書形式で取得
        
        Returns:
            状態の辞書
        """
        return {
            "round_index": self.round_index,
            "round_num_wind": self.round_num_wind,
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "dealer": self.dealer,
            "current_scores": list(self.current_scores),
            "player_hands": [list(hand) for hand in self.player_hands],
            "player_discards": [list(discards) for discards in self.player_discards],
            "player_melds": [list(melds) for melds in self.player_melds],
            "player_reach_status": list(self.player_reach_status),
            "dora_indicators": list(self.dora_indicators),
            "current_player": self.current_player,
            "junme": self.junme,
            "wall_tile_count": self.wall_tile_count
        }

