"""XML Parser for Mahjong game logs.

Parses XML game logs from Tenhou and extracts:
- Game metadata (players, scores)
- Action sequences (draws, discards, calls)
- Game state at each decision point
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MeldInfo:
    """Represents a meld (chi/pon/kan)."""
    meld_type: str  # 'chi', 'pon', 'kan'
    tiles: List[int]  # Tile IDs in the meld
    from_who: int  # Which player the tile was taken from


@dataclass
class GameAction:
    """Represents a single action in a Mahjong game."""
    player_id: int
    action_type: str  # 'draw', 'discard', 'chi', 'pon', 'kan', 'riichi', 'tsumo', 'ron'
    tile: Optional[int]  # Tile ID (for draw/discard)
    meld: Optional[MeldInfo]  # Meld information (for calls)
    turn: int
    round_num: int


@dataclass
class RoundState:
    """State of a round at initialization."""
    round_num: int  # Round number (0=East 1, 1=East 2, etc.)
    honba: int  # Number of repeat counters
    riichi_sticks: int  # Number of riichi sticks on table
    dora_indicator: int  # Dora indicator tile
    dealer: int  # Dealer (oya) player ID
    initial_hands: List[List[int]]  # Initial 13-tile hands for all 4 players
    scores: List[int]  # Scores for all 4 players (in 100s)


@dataclass
class Game:
    """Complete game record."""
    game_id: str
    player_names: List[str]
    rounds: List[RoundState]
    actions: List[GameAction]
    final_scores: List[int]


class TenhouXMLParser:
    """Parser for Tenhou Mahjong XML game logs."""
    
    def __init__(self):
        """Initialize the XML parser."""
        pass
    
    @staticmethod
    def tile_to_type(tile_id: int) -> int:
        """Convert tile ID (0-135) to tile type (0-33).
        
        Tenhou uses 136 tiles (34 types × 4 copies).
        - 0-35: Man (characters) 1m-9m (4 copies each)
        - 36-71: Pin (circles) 1p-9p (4 copies each)
        - 72-107: Sou (bamboo) 1s-9s (4 copies each)
        - 108-135: Honors E,S,W,N,Haku,Hatsu,Chun (4 copies each)
        
        Args:
            tile_id: Tile ID from 0-135
            
        Returns:
            Tile type from 0-33
        """
        return tile_id // 4
    
    @staticmethod
    def decode_meld(meld_code: int) -> MeldInfo:
        """Decode meld information from Tenhou's m attribute.
        
        Args:
            meld_code: Integer meld code from XML
            
        Returns:
            MeldInfo object
        """
        # This is a simplified version - full decoding is complex
        # Reference: http://tenhou.net/img/tehai.js
        
        if meld_code & 0x4:
            # Chi (sequence)
            meld_type = 'chi'
            t0 = (meld_code >> 3) & 0x3
            t1 = (meld_code >> 5) & 0x3
            t2 = (meld_code >> 7) & 0x3
            base_tile = (meld_code >> 10) & 0x7f
            tiles = [base_tile + t0, base_tile + t1, base_tile + t2]
            from_who = meld_code & 0x3
        elif meld_code & 0x18:
            # Pon (triplet)
            meld_type = 'pon'
            base_tile = (meld_code >> 9) & 0x7f
            unused = (meld_code >> 5) & 0x3
            tiles = [base_tile + i for i in range(3) if i != unused]
            from_who = meld_code & 0x3
        else:
            # Kan (quad)
            meld_type = 'kan'
            base_tile = (meld_code >> 8) & 0x7f
            tiles = [base_tile + i for i in range(4)]
            from_who = meld_code & 0x3
        
        return MeldInfo(meld_type=meld_type, tiles=tiles, from_who=from_who)
    
    def parse_file(self, filepath: str) -> Optional[Game]:
        """Parse a single XML file and return a Game object.
        
        Args:
            filepath: Path to XML file
            
        Returns:
            Game object or None if parsing fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract game ID from filename
            import os
            game_id = os.path.basename(filepath).replace('.xml', '')
            
            # Parse player names from UN tag
            player_names = self._parse_player_names(content)
            
            # Parse rounds and actions
            rounds, actions = self._parse_rounds_and_actions(content)
            
            # Parse final scores from owari attribute in last AGARI/RYUUKYOKU
            final_scores = self._parse_final_scores(content)
            
            return Game(
                game_id=game_id,
                player_names=player_names,
                rounds=rounds,
                actions=actions,
                final_scores=final_scores
            )
        
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return None
    
    def _parse_player_names(self, content: str) -> List[str]:
        """Parse player names from UN tag."""
        un_match = re.search(r'<UN n0="([^"]*)" n1="([^"]*)" n2="([^"]*)" n3="([^"]*)"', content)
        if un_match:
            # Decode URL-encoded names
            from urllib.parse import unquote
            return [unquote(un_match.group(i + 1)) for i in range(4)]
        return ["Player0", "Player1", "Player2", "Player3"]
    
    def _parse_rounds_and_actions(self, content: str) -> Tuple[List[RoundState], List[GameAction]]:
        """Parse all rounds and actions from the XML content."""
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
            
            round_state = RoundState(
                round_num=round_num,
                honba=honba,
                riichi_sticks=riichi_sticks,
                dora_indicator=dora_indicator,
                dealer=dealer,
                initial_hands=initial_hands,
                scores=scores
            )
            rounds.append(round_state)
            
            # Parse actions in this round
            turn = 0
            
            # Draw actions: T/U/V/W followed by tile ID
            for match in re.finditer(r'<([TUVW])(\d+)/>', round_content):
                player_id = 'TUVW'.index(match.group(1))
                tile = int(match.group(2))
                actions.append(GameAction(
                    player_id=player_id,
                    action_type='draw',
                    tile=tile,
                    meld=None,
                    turn=turn,
                    round_num=round_idx
                ))
            
            # Discard actions: D/E/F/G followed by tile ID
            for match in re.finditer(r'<([DEFG])(\d+)/>', round_content):
                player_id = 'DEFG'.index(match.group(1))
                tile = int(match.group(2))
                actions.append(GameAction(
                    player_id=player_id,
                    action_type='discard',
                    tile=tile,
                    meld=None,
                    turn=turn,
                    round_num=round_idx
                ))
                turn += 1
            
            # Call actions: N tag
            for match in re.finditer(r'<N who="(\d)" m="(\d+)"', round_content):
                player_id = int(match.group(1))
                meld_code = int(match.group(2))
                meld_info = self.decode_meld(meld_code)
                actions.append(GameAction(
                    player_id=player_id,
                    action_type=meld_info.meld_type,
                    tile=None,
                    meld=meld_info,
                    turn=turn,
                    round_num=round_idx
                ))
            
            # Riichi actions
            for match in re.finditer(r'<REACH who="(\d)" step="1"', round_content):
                player_id = int(match.group(1))
                actions.append(GameAction(
                    player_id=player_id,
                    action_type='riichi',
                    tile=None,
                    meld=None,
                    turn=turn,
                    round_num=round_idx
                ))
        
        return rounds, actions
    
    def _parse_final_scores(self, content: str) -> List[int]:
        """Parse final scores from owari attribute."""
        # Look for owari attribute in AGARI or RYUUKYOKU tag
        owari_match = re.search(r'owari="([^"]+)"', content)
        if owari_match:
            scores_str = owari_match.group(1).split(',')
            # owari format: score0, delta0, score1, delta1, ...
            # We want just the scores
            return [int(scores_str[i]) for i in range(0, 8, 2)]
        
        # Fallback: try to get from last sc (score change) attribute
        sc_matches = list(re.finditer(r'sc="([^"]+)"', content))
        if sc_matches:
            last_sc = sc_matches[-1].group(1).split(',')
            return [int(last_sc[i]) for i in range(0, 8, 2)]
        
        return [250, 250, 250, 250]  # Default starting scores
    
    def parse_directory(self, dirpath: str, max_files: Optional[int] = None) -> List[Game]:
        """Parse all XML files in a directory.
        
        Args:
            dirpath: Path to directory containing XML files
            max_files: Maximum number of files to parse (None for all)
            
        Returns:
            List of Game objects
        """
        import os
        import glob
        
        xml_files = glob.glob(os.path.join(dirpath, '*.xml'))
        if max_files:
            xml_files = xml_files[:max_files]
        
        games = []
        for filepath in xml_files:
            game = self.parse_file(filepath)
            if game:
                games.append(game)
        
        logger.info(f"Parsed {len(games)} games from {dirpath}")
        return games
    
    def compute_statistics(self, games: List[Game]) -> Dict[str, float]:
        """Compute statistics across multiple games.
        
        Args:
            games: List of Game objects
            
        Returns:
            Dictionary of statistics
        """
        if not games:
            return {}
        
        total_actions = sum(len(game.actions) for game in games)
        total_rounds = sum(len(game.rounds) for game in games)
        
        call_counts = sum(
            sum(1 for action in game.actions if action.action_type in ['chi', 'pon', 'kan'])
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
        
        return {
            'total_games': len(games),
            'total_rounds': total_rounds,
            'total_actions': total_actions,
            'avg_rounds_per_game': total_rounds / len(games),
            'avg_actions_per_game': total_actions / len(games),
            'call_rate': call_counts / discard_counts if discard_counts > 0 else 0,
            'riichi_rate': riichi_counts / total_rounds if total_rounds > 0 else 0,
        }
