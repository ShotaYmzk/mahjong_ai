"""Enhanced XML Parser for Mahjong game logs.

拡張機能:
- 手出し/ツモ切り判定
- 赤ドラ検出（5m=16/52, 5p=52/88, 5s=88/124）
- リーチ宣言巡目の記録
- ドラ増加（カン）の追跡
- タイムスタンプ管理
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMeldInfo:
    """拡張された副露情報"""
    meld_type: str  # 'chi', 'pon', 'kan', 'ankan', 'kakan'
    tiles: List[int]  # Tile IDs in the meld
    from_who: int  # Which player the tile was taken from (-1 for self)
    called_tile: int  # The specific tile that was called
    timing: int  # Turn number when the meld was made


@dataclass
class EnhancedGameAction:
    """拡張されたゲームアクション"""
    player_id: int
    action_type: str  # 'draw', 'discard', 'chi', 'pon', 'kan', 'riichi', 'tsumo', 'ron', 'dora'
    tile: Optional[int]  # Tile ID (for draw/discard)
    tile_type: Optional[int]  # Tile type (0-33)
    meld: Optional[EnhancedMeldInfo]  # Meld information
    turn: int
    round_num: int
    
    # 拡張フィールド
    is_tsumogiri: bool = False  # ツモ切りかどうか
    is_riichi_discard: bool = False  # リーチ宣言牌か
    is_red_dora: bool = False  # 赤ドラか
    riichi_turn: Optional[int] = None  # リーチ宣言した巡目


@dataclass
class EnhancedRoundState:
    """拡張された局の状態"""
    round_num: int  # Round number
    round_wind: int  # 0=East, 1=South
    round_index: int  # 0-3 for E1-E4, S1-S4
    honba: int  # Number of repeat counters
    riichi_sticks: int  # Number of riichi sticks on table
    dora_indicators: List[int]  # Dora indicator tiles (can increase with kan)
    ura_dora_indicators: List[int]  # Ura dora indicators (revealed at end)
    dealer: int  # Dealer (oya) player ID
    initial_hands: List[List[int]]  # Initial 13-tile hands for all 4 players
    scores: List[int]  # Scores for all 4 players (in 100s)
    
    # 赤ドラ情報
    visible_red_dora: Dict[str, int] = field(default_factory=dict)  # {'5m': 1, '5p': 0, '5s': 1}


@dataclass
class EnhancedGame:
    """拡張されたゲーム記録"""
    game_id: str
    game_type: str  # '四鳳東喰', '四鳳南喰', etc.
    player_names: List[str]
    player_ranks: List[str]  # Dan ranks
    player_ratings: List[float]  # R ratings
    rounds: List[EnhancedRoundState]
    actions: List[EnhancedGameAction]
    final_scores: List[int]
    final_ranks: List[int]  # Final placement (0-3)


class EnhancedXMLParser:
    """Enhanced parser for Tenhou Mahjong XML game logs."""
    
    # 赤ドラのtile ID（天鳳形式: 0-135）
    RED_DORA_IDS = {
        16: '5m',   # Red 5 man
        52: '5p',   # Red 5 pin
        88: '5s',   # Red 5 sou
    }
    
    def __init__(self):
        """Initialize the enhanced parser."""
        self.current_game_type = None
    
    @staticmethod
    def tile_to_type(tile_id: int) -> int:
        """Convert tile ID (0-135) to tile type (0-33)."""
        return tile_id // 4
    
    @staticmethod
    def is_red_dora(tile_id: int) -> bool:
        """Check if a tile is a red dora."""
        return tile_id in EnhancedXMLParser.RED_DORA_IDS
    
    @staticmethod
    def get_red_dora_type(tile_id: int) -> Optional[str]:
        """Get red dora type ('5m', '5p', '5s') if applicable."""
        return EnhancedXMLParser.RED_DORA_IDS.get(tile_id, None)
    
    @staticmethod
    def decode_meld(meld_code: int) -> EnhancedMeldInfo:
        """Decode meld information from Tenhou's m attribute.
        
        Args:
            meld_code: Integer meld code from XML
            
        Returns:
            EnhancedMeldInfo object with correctly decoded tiles
        """
        from_who = meld_code & 3
        
        # Chi (sequence) - bit 2 is set
        if meld_code & (1 << 2):
            meld_type = 'chi'
            t = meld_code >> 10
            r = t % 3  # Which tile was called (0-2)
            t //= 3
            
            # Determine base tile index (0-33 system)
            if 0 <= t <= 6:
                base_index = t  # 1m-7m
            elif 7 <= t <= 13:
                base_index = (t - 7) + 9  # 1p-7p
            elif 14 <= t <= 20:
                base_index = (t - 14) + 18  # 1s-7s
            else:
                base_index = 0
            
            # Get the three tile IDs in the sequence
            offsets = [(meld_code >> 3) & 3, (meld_code >> 5) & 3, (meld_code >> 7) & 3]
            tiles = []
            for i in range(3):
                tile_kind = base_index + i
                tile_id = tile_kind * 4 + offsets[i]
                tiles.append(tile_id)
            
            called_tile = tiles[r]
        
        # Pon (triplet) - bit 3 is set
        elif meld_code & (1 << 3):
            meld_type = 'pon'
            t = meld_code >> 9
            r = t % 3  # Which position was called
            t //= 3
            
            base_id = t * 4
            unused_offset = (meld_code >> 5) & 3
            
            # Select 3 out of 4 tiles (excluding unused)
            tiles = []
            for i in range(4):
                if i != unused_offset:
                    tiles.append(base_id + i)
            
            called_tile = tiles[r] if r < len(tiles) else tiles[0]
        
        # Kakan (added kan) - bit 4 is set
        elif meld_code & (1 << 4):
            meld_type = 'kakan'
            t = meld_code >> 9
            t //= 3
            
            base_id = t * 4
            tiles = [base_id, base_id + 1, base_id + 2, base_id + 3]
            from_who = -1  # Self
            called_tile = tiles[0]
        
        # Ankan or Daiminkan
        else:
            tile_id_raw = meld_code >> 8
            tile_index = tile_id_raw // 4
            
            base_id = tile_index * 4
            tiles = [base_id, base_id + 1, base_id + 2, base_id + 3]
            
            # If from_who is 3, it's ankan (concealed kan)
            if from_who == 3:
                meld_type = 'ankan'
                from_who = -1  # Self
            else:
                meld_type = 'kan'
            
            called_tile = tiles[0]
        
        return EnhancedMeldInfo(
            meld_type=meld_type,
            tiles=tiles,
            from_who=from_who,
            called_tile=called_tile,
            timing=0  # Will be set later
        )
    
    def parse_file(self, filepath: str) -> Optional[EnhancedGame]:
        """Parse a single XML file and return an EnhancedGame object.
        
        Args:
            filepath: Path to XML file
            
        Returns:
            EnhancedGame object or None if parsing fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract game ID from filename
            import os
            game_id = os.path.basename(filepath).replace('.xml', '')
            
            # Parse game type
            game_type = self._parse_game_type(content)
            
            # Parse player info
            player_names, player_ranks, player_ratings = self._parse_player_info(content)
            
            # Parse rounds and actions
            rounds, actions = self._parse_rounds_and_actions_enhanced(content)
            
            # Parse final scores and ranks
            final_scores, final_ranks = self._parse_final_scores_and_ranks(content)
            
            return EnhancedGame(
                game_id=game_id,
                game_type=game_type,
                player_names=player_names,
                player_ranks=player_ranks,
                player_ratings=player_ratings,
                rounds=rounds,
                actions=actions,
                final_scores=final_scores,
                final_ranks=final_ranks
            )
        
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return None
    
    def _parse_game_type(self, content: str) -> str:
        """Parse game type from GO tag."""
        go_match = re.search(r'<GO type="(\d+)"', content)
        if go_match:
            type_code = int(go_match.group(1))
            # Decode game type (simplified)
            # 天鳳のtype codeは複雑だが、基本的な判定のみ実装
            if type_code & 0x10:
                return "三麻"
            else:
                return "四麻"
        return "不明"
    
    def _parse_player_info(self, content: str) -> Tuple[List[str], List[str], List[float]]:
        """Parse player names, ranks, and ratings from UN tag."""
        un_match = re.search(
            r'<UN n0="([^"]*)" n1="([^"]*)" n2="([^"]*)" n3="([^"]*)"'
            r'(?: dan="([^"]*)")?(?: rate="([^"]*)")?(?: sx="([^"]*)")?',
            content
        )
        
        if not un_match:
            return (
                ["Player0", "Player1", "Player2", "Player3"],
                ["0", "0", "0", "0"],
                [1500.0, 1500.0, 1500.0, 1500.0]
            )
        
        # Decode URL-encoded names
        from urllib.parse import unquote
        names = [unquote(un_match.group(i + 1)) for i in range(4)]
        
        # Parse ranks (dan)
        ranks_str = un_match.group(5) if len(un_match.groups()) >= 5 and un_match.group(5) else "0,0,0,0"
        ranks = ranks_str.split(',') if ',' in ranks_str else ["0", "0", "0", "0"]
        
        # Parse ratings
        ratings_str = un_match.group(6) if len(un_match.groups()) >= 6 and un_match.group(6) else "1500,1500,1500,1500"
        try:
            ratings = [float(r) for r in ratings_str.split(',')] if ',' in ratings_str else [1500.0] * 4
        except:
            ratings = [1500.0, 1500.0, 1500.0, 1500.0]
        
        return names, ranks, ratings
    
    def _parse_rounds_and_actions_enhanced(self, content: str) -> Tuple[List[EnhancedRoundState], List[EnhancedGameAction]]:
        """Parse all rounds and actions with enhanced information."""
        rounds = []
        actions = []
        
        # Split content by INIT tags to separate rounds
        round_splits = re.split(r'<INIT ', content)
        
        for round_idx, round_content in enumerate(round_splits[1:]):  # Skip first split (header)
            round_content = '<INIT ' + round_content
            
            # Parse INIT tag
            init_match = re.search(
                r'<INIT seed="([^"]+)" ten="([^"]+)" oya="(\d)" hai0="([^"]+)" hai1="([^"]+)" hai2="([^"]+)" hai3="([^"]+)"',
                round_content
            )
            
            if not init_match:
                continue
            
            # Parse seed (round info)
            seed_parts = init_match.group(1).split(',')
            round_num = int(seed_parts[0])
            honba = int(seed_parts[1])
            riichi_sticks = int(seed_parts[2])
            dora_indicator = int(seed_parts[5])
            
            # Determine round wind and index
            round_wind = 0 if round_num < 4 else 1  # 0=East, 1=South
            round_index = round_num % 4
            
            # Parse scores
            scores = [int(s) for s in init_match.group(2).split(',')]
            
            # Parse dealer
            dealer = int(init_match.group(3))
            
            # Parse initial hands
            initial_hands = []
            for i in range(4):
                hand_str = init_match.group(4 + i)
                hand = [int(t) for t in hand_str.split(',')]
                initial_hands.append(hand)
            
            # Count red dora in initial hands
            visible_red_dora = {'5m': 0, '5p': 0, '5s': 0}
            for hand in initial_hands:
                for tile in hand:
                    red_type = self.get_red_dora_type(tile)
                    if red_type:
                        visible_red_dora[red_type] += 1
            
            round_state = EnhancedRoundState(
                round_num=round_num,
                round_wind=round_wind,
                round_index=round_index,
                honba=honba,
                riichi_sticks=riichi_sticks,
                dora_indicators=[dora_indicator],
                ura_dora_indicators=[],
                dealer=dealer,
                initial_hands=initial_hands,
                scores=scores,
                visible_red_dora=visible_red_dora
            )
            
            # Parse actions in this round with enhanced information
            round_actions = self._parse_round_actions_enhanced(
                round_content, round_idx, round_state
            )
            actions.extend(round_actions)
            
            # Parse additional dora (from DORA tags)
            for dora_match in re.finditer(r'<DORA hai="(\d+)"', round_content):
                new_dora = int(dora_match.group(1))
                round_state.dora_indicators.append(new_dora)
            
            rounds.append(round_state)
        
        return rounds, actions
    
    def _parse_round_actions_enhanced(self, round_content: str, round_idx: int, 
                                     round_state: EnhancedRoundState) -> List[EnhancedGameAction]:
        """Parse actions in a round with enhanced information."""
        actions = []
        turn = 0
        
        # Track last draw for each player to detect tsumogiri
        last_draw = {0: None, 1: None, 2: None, 3: None}
        
        # Track riichi status
        riichi_status = {0: False, 1: False, 2: False, 3: False}
        riichi_turns = {0: None, 1: None, 2: None, 3: None}
        
        # Parse all tags in order
        all_tags = re.findall(r'<([A-Z]+)([^>]*)/?>', round_content)
        
        for tag_name, tag_attrs in all_tags:
            # Draw actions: T/U/V/W
            if tag_name in 'TUVW':
                player_id = 'TUVW'.index(tag_name)
                tile_match = re.search(r'(\d+)', tag_attrs)
                if tile_match:
                    tile = int(tile_match.group(1))
                    tile_type = self.tile_to_type(tile)
                    is_red = self.is_red_dora(tile)
                    
                    last_draw[player_id] = tile
                    
                    actions.append(EnhancedGameAction(
                        player_id=player_id,
                        action_type='draw',
                        tile=tile,
                        tile_type=tile_type,
                        meld=None,
                        turn=turn,
                        round_num=round_idx,
                        is_red_dora=is_red
                    ))
            
            # Discard actions: D/E/F/G
            elif tag_name in 'DEFG':
                player_id = 'DEFG'.index(tag_name)
                tile_match = re.search(r'(\d+)', tag_attrs)
                if tile_match:
                    tile = int(tile_match.group(1))
                    tile_type = self.tile_to_type(tile)
                    is_red = self.is_red_dora(tile)
                    
                    # Check if this is tsumogiri (discarding the just-drawn tile)
                    is_tsumogiri = (last_draw[player_id] == tile)
                    
                    # Check if this is a riichi discard
                    is_riichi_discard = riichi_status[player_id] and (riichi_turns[player_id] == turn)
                    
                    actions.append(EnhancedGameAction(
                        player_id=player_id,
                        action_type='discard',
                        tile=tile,
                        tile_type=tile_type,
                        meld=None,
                        turn=turn,
                        round_num=round_idx,
                        is_tsumogiri=is_tsumogiri,
                        is_riichi_discard=is_riichi_discard,
                        is_red_dora=is_red,
                        riichi_turn=riichi_turns[player_id] if riichi_status[player_id] else None
                    ))
                    
                    # Clear last draw after discard
                    last_draw[player_id] = None
                    turn += 1
            
            # Call actions: N
            elif tag_name == 'N':
                who_match = re.search(r'who="(\d)"', tag_attrs)
                m_match = re.search(r'm="(\d+)"', tag_attrs)
                if who_match and m_match:
                    player_id = int(who_match.group(1))
                    meld_code = int(m_match.group(1))
                    meld_info = self.decode_meld(meld_code)
                    meld_info.timing = turn
                    
                    actions.append(EnhancedGameAction(
                        player_id=player_id,
                        action_type=meld_info.meld_type,
                        tile=None,
                        tile_type=None,
                        meld=meld_info,
                        turn=turn,
                        round_num=round_idx
                    ))
            
            # Riichi actions
            elif tag_name == 'REACH':
                who_match = re.search(r'who="(\d)"', tag_attrs)
                step_match = re.search(r'step="(\d)"', tag_attrs)
                if who_match and step_match:
                    player_id = int(who_match.group(1))
                    step = int(step_match.group(1))
                    
                    if step == 1:  # Riichi declaration
                        riichi_status[player_id] = True
                        riichi_turns[player_id] = turn
                        
                        actions.append(EnhancedGameAction(
                            player_id=player_id,
                            action_type='riichi',
                            tile=None,
                            tile_type=None,
                            meld=None,
                            turn=turn,
                            round_num=round_idx,
                            riichi_turn=turn
                        ))
        
        return actions
    
    def _parse_final_scores_and_ranks(self, content: str) -> Tuple[List[int], List[int]]:
        """Parse final scores and ranks from owari attribute."""
        # Look for owari attribute in AGARI or RYUUKYOKU tag
        owari_match = re.search(r'owari="([^"]+)"', content)
        if owari_match:
            owari_parts = owari_match.group(1).split(',')
            # owari format: score0, delta0, score1, delta1, ...
            scores = [int(owari_parts[i]) for i in range(0, 8, 2)]
            
            # Calculate ranks (1st place = 0, 4th place = 3)
            score_with_idx = [(score, idx) for idx, score in enumerate(scores)]
            score_with_idx.sort(key=lambda x: x[0], reverse=True)
            ranks = [0, 0, 0, 0]
            for rank, (score, idx) in enumerate(score_with_idx):
                ranks[idx] = rank
            
            return scores, ranks
        
        # Fallback: try to get from last sc (score change) attribute
        sc_matches = list(re.finditer(r'sc="([^"]+)"', content))
        if sc_matches:
            last_sc = sc_matches[-1].group(1).split(',')
            scores = [int(last_sc[i]) for i in range(0, 8, 2)]
            
            score_with_idx = [(score, idx) for idx, score in enumerate(scores)]
            score_with_idx.sort(key=lambda x: x[0], reverse=True)
            ranks = [0, 0, 0, 0]
            for rank, (score, idx) in enumerate(score_with_idx):
                ranks[idx] = rank
            
            return scores, ranks
        
        return [250, 250, 250, 250], [0, 0, 0, 0]  # Default
    
    def parse_directory(self, dirpath: str, max_files: Optional[int] = None, 
                       show_progress: bool = False) -> List[EnhancedGame]:
        """Parse all XML files in a directory.
        
        Args:
            dirpath: Path to directory containing XML files
            max_files: Maximum number of files to parse (None for all)
            show_progress: Show progress bar
            
        Returns:
            List of EnhancedGame objects
        """
        import os
        import glob
        from pathlib import Path
        
        xml_files = glob.glob(os.path.join(dirpath, '*.xml'))
        if max_files:
            xml_files = xml_files[:max_files]
        
        games = []
        total = len(xml_files)
        
        if show_progress:
            try:
                from tqdm import tqdm
                xml_files = tqdm(xml_files, desc="Parsing XML files")
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")
        
        for filepath in xml_files:
            game = self.parse_file(filepath)
            if game:
                games.append(game)
        
        logger.info(f"Parsed {len(games)} games from {dirpath}")
        return games
    
    def compute_statistics(self, games: List[EnhancedGame]) -> Dict[str, any]:
        """Compute statistics across multiple games.
        
        Args:
            games: List of EnhancedGame objects
            
        Returns:
            Dictionary of statistics
        """
        if not games:
            return {}
        
        total_actions = sum(len(game.actions) for game in games)
        total_rounds = sum(len(game.rounds) for game in games)
        
        call_counts = sum(
            sum(1 for action in game.actions if action.action_type in ['chi', 'pon', 'kan', 'ankan', 'kakan'])
            for game in games
        )
        
        riichi_counts = sum(
            sum(1 for action in game.actions if action.action_type == 'riichi')
            for game in games
        )
        
        discard_counts = sum(
            sum(1 for action in game.actions if action.action_type == 'discard')
            for game in games
        )
        
        tsumogiri_counts = sum(
            sum(1 for action in game.actions if action.action_type == 'discard' and action.is_tsumogiri)
            for game in games
        )
        
        red_dora_counts = sum(
            sum(1 for action in game.actions if action.is_red_dora)
            for game in games
        )
        
        return {
            'total_games': len(games),
            'total_rounds': total_rounds,
            'total_actions': total_actions,
            'total_discards': discard_counts,
            'avg_rounds_per_game': total_rounds / len(games),
            'avg_actions_per_game': total_actions / len(games),
            'call_rate': call_counts / discard_counts if discard_counts > 0 else 0,
            'riichi_rate': riichi_counts / total_rounds if total_rounds > 0 else 0,
            'tsumogiri_rate': tsumogiri_counts / discard_counts if discard_counts > 0 else 0,
            'red_dora_appearance': red_dora_counts / total_actions if total_actions > 0 else 0,
        }

