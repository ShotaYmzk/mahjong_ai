"""Dataset builder for Mahjong AI training.

Builds PyTorch TensorDatasets from parsed XML game logs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import List, Dict, Tuple, Optional
import logging
import json
from pathlib import Path

from .parse_xml import TenhouXMLParser, Game, GameAction, RoundState
from .feature_encoder import TileEncoder, AdvancedFeatureExtractor

logger = logging.getLogger(__name__)


class MahjongGameStateExtractor:
    """Extracts game states and actions from parsed games."""
    
    def __init__(self):
        """Initialize the extractor."""
        self.parser = TenhouXMLParser()
        self.encoder = TileEncoder()
    
    def reconstruct_hand_state(self, round_state: RoundState, 
                               actions_so_far: List[GameAction],
                               player_id: int) -> Tuple[List[int], List[Tuple[str, List[int]]]]:
        """Reconstruct hand and melds for a player at a specific point in time.
        
        Args:
            round_state: Initial round state
            actions_so_far: All actions up to this point in the round
            player_id: Player whose state to reconstruct
            
        Returns:
            (hand_tiles, melds) tuple
        """
        # Start with initial hand
        hand = round_state.initial_hands[player_id].copy()
        melds = []
        
        # Apply all actions
        for action in actions_so_far:
            if action.player_id != player_id:
                continue
            
            if action.action_type == 'draw':
                hand.append(action.tile)
            
            elif action.action_type == 'discard':
                if action.tile in hand:
                    hand.remove(action.tile)
            
            elif action.action_type in ['chi', 'pon', 'kan']:
                # Remove tiles from hand for meld
                if action.meld:
                    meld_tiles = action.meld.tiles
                    for tile in meld_tiles:
                        tile_type = self.encoder.tile_id_to_type(tile)
                        # Remove corresponding tile from hand
                        for h_tile in hand[:]:
                            if self.encoder.tile_id_to_type(h_tile) == tile_type:
                                hand.remove(h_tile)
                                break
                    melds.append((action.action_type, meld_tiles))
        
        return hand, melds
    
    def extract_decision_points(self, game: Game) -> List[Dict]:
        """Extract all decision points from a game.
        
        A decision point is any time a player must choose to discard a tile.
        
        Args:
            game: Parsed game object
            
        Returns:
            List of decision point dictionaries
        """
        decision_points = []
        
        for round_idx, round_state in enumerate(game.rounds):
            # Get all actions in this round
            round_actions = [a for a in game.actions if a.round_num == round_idx]
            
            # Track actions for each player
            for action_idx, action in enumerate(round_actions):
                # We want decision points right before discards
                if action.action_type != 'discard':
                    continue
                
                player_id = action.player_id
                
                # Get all actions before this discard
                prior_actions = round_actions[:action_idx]
                
                # Reconstruct game state
                hand, melds = self.reconstruct_hand_state(round_state, prior_actions, player_id)
                
                # Collect discards for all players
                all_discards = [[] for _ in range(4)]
                for prior_action in prior_actions:
                    if prior_action.action_type == 'discard':
                        all_discards[prior_action.player_id].append(prior_action.tile)
                
                # Collect melds for other players
                opponent_melds = [[] for _ in range(3)]
                opponent_idx = 0
                for pid in range(4):
                    if pid == player_id:
                        continue
                    _, opp_melds = self.reconstruct_hand_state(round_state, prior_actions, pid)
                    opponent_melds[opponent_idx] = opp_melds
                    opponent_idx += 1
                
                # Create decision point
                decision_point = {
                    'game_id': game.game_id,
                    'round_num': round_idx,
                    'player_id': player_id,
                    'hand': hand,
                    'melds': melds,
                    'discards': all_discards,
                    'opponent_melds': opponent_melds,
                    'dora_indicator': [round_state.dora_indicator],
                    'dealer': round_state.dealer,
                    'is_dealer': (player_id == round_state.dealer),
                    'score': round_state.scores[player_id],
                    'turn': action_idx,
                    'label_tile': action.tile,  # The tile that was actually discarded
                }
                
                decision_points.append(decision_point)
        
        return decision_points


class MahjongDatasetBuilder:
    """Builds PyTorch datasets from game logs."""
    
    def __init__(self, data_dir: str, output_dir: str):
        """Initialize the dataset builder.
        
        Args:
            data_dir: Directory containing XML game logs
            output_dir: Directory to save processed datasets
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = TenhouXMLParser()
        self.encoder = TileEncoder()
        self.extractor = MahjongGameStateExtractor()
        self.feature_extractor = AdvancedFeatureExtractor()
    
    def build_dataset(self, max_games: Optional[int] = None) -> Dict[str, any]:
        """Build dataset from XML files.
        
        Args:
            max_games: Maximum number of games to process
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info(f"Parsing games from {self.data_dir}")
        
        # Parse all games
        games = self.parser.parse_directory(str(self.data_dir), max_files=max_games)
        
        if not games:
            raise ValueError(f"No games found in {self.data_dir}")
        
        logger.info(f"Parsed {len(games)} games")
        
        # Extract decision points
        all_decision_points = []
        for game in games:
            try:
                decision_points = self.extractor.extract_decision_points(game)
                all_decision_points.extend(decision_points)
            except Exception as e:
                logger.warning(f"Failed to extract decision points from game {game.game_id}: {e}")
        
        logger.info(f"Extracted {len(all_decision_points)} decision points")
        
        # Convert to features and labels
        features_list = []
        labels_list = []
        
        for dp in all_decision_points:
            try:
                # Encode features
                own_hand = dp['hand']
                own_melds = dp['melds']
                all_discards = dp['discards']
                opponent_melds = dp['opponent_melds']
                dora_indicators = dp['dora_indicator']
                
                feature_vector = self.encoder.encode_game_state(
                    own_hand, own_melds, all_discards, opponent_melds, dora_indicators
                )
                
                # Label is the tile that was discarded (0-33)
                label = self.encoder.tile_id_to_type(dp['label_tile'])
                
                features_list.append(feature_vector)
                labels_list.append(label)
            
            except Exception as e:
                logger.warning(f"Failed to encode decision point: {e}")
        
        # Convert to numpy arrays
        X = np.stack(features_list).astype(np.float32)
        y = np.array(labels_list, dtype=np.int64)
        
        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        # Save to disk
        np.save(self.output_dir / 'X.npy', X)
        np.save(self.output_dir / 'y.npy', y)
        
        # Compute and save statistics
        statistics = self.parser.compute_statistics(games)
        statistics.update({
            'num_decision_points': len(all_decision_points),
            'feature_dim': X.shape[1],
            'num_classes': 34,
        })
        
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"Dataset saved to {self.output_dir}")
        
        return statistics
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load preprocessed dataset from disk.
        
        Returns:
            (X, y) tuple of numpy arrays
        """
        X = np.load(self.output_dir / 'X.npy')
        y = np.load(self.output_dir / 'y.npy')
        
        logger.info(f"Loaded dataset: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def create_torch_dataset(self) -> TensorDataset:
        """Create PyTorch TensorDataset.
        
        Returns:
            TensorDataset
        """
        X, y = self.load_dataset()
        
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        return TensorDataset(X_tensor, y_tensor)
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, random_seed: int = 42) -> Dict[str, TensorDataset]:
        """Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' TensorDatasets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        X, y = self.load_dataset()
        
        # Shuffle data
        np.random.seed(random_seed)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Calculate split points
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split data
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Save splits
        split_dir = self.output_dir / 'split'
        split_dir.mkdir(exist_ok=True)
        
        np.save(split_dir / 'X_train.npy', X_train)
        np.save(split_dir / 'y_train.npy', y_train)
        np.save(split_dir / 'X_val.npy', X_val)
        np.save(split_dir / 'y_val.npy', y_val)
        np.save(split_dir / 'X_test.npy', X_test)
        np.save(split_dir / 'y_test.npy', y_test)
        
        logger.info(f"Split dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Create TensorDatasets
        datasets = {
            'train': TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            'val': TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            'test': TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        }
        
        return datasets
    
    def create_dataloaders(self, batch_size: int = 32, num_workers: int = 4) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders for train/val/test.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        datasets = self.split_dataset()
        
        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers),
            'val': DataLoader(datasets['val'], batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers),
            'test': DataLoader(datasets['test'], batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)
        }
        
        return dataloaders


class SequenceDatasetBuilder:
    """Builds sequence datasets for transformer models."""
    
    def __init__(self, data_dir: str, output_dir: str, max_sequence_length: int = 128):
        """Initialize the sequence dataset builder.
        
        Args:
            data_dir: Directory containing XML game logs
            output_dir: Directory to save processed datasets
            max_sequence_length: Maximum sequence length
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_sequence_length = max_sequence_length
        
        self.parser = TenhouXMLParser()
        self.encoder = TileEncoder()
        self.extractor = MahjongGameStateExtractor()
    
    def build_sequence_dataset(self, max_games: Optional[int] = None) -> Dict[str, any]:
        """Build sequence dataset where each sample is a complete round.
        
        Args:
            max_games: Maximum number of games to process
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info(f"Building sequence dataset from {self.data_dir}")
        
        # Parse games
        games = self.parser.parse_directory(str(self.data_dir), max_files=max_games)
        
        sequences = []
        
        for game in games:
            for round_idx, round_state in enumerate(game.rounds):
                # Get all actions in this round for each player
                for player_id in range(4):
                    round_actions = [a for a in game.actions 
                                   if a.round_num == round_idx and a.player_id == player_id]
                    
                    if len(round_actions) == 0:
                        continue
                    
                    # Build sequence of states and actions
                    sequence_features = []
                    sequence_labels = []
                    
                    for action_idx, action in enumerate(round_actions):
                        if action.action_type != 'discard':
                            continue
                        
                        prior_actions = [a for a in game.actions 
                                       if a.round_num == round_idx][:action_idx]
                        
                        hand, melds = self.extractor.reconstruct_hand_state(
                            round_state, prior_actions, player_id
                        )
                        
                        # Simplified feature: just hand encoding
                        features = self.encoder.encode_hand(hand)
                        label = self.encoder.tile_id_to_type(action.tile)
                        
                        sequence_features.append(features)
                        sequence_labels.append(label)
                    
                    if len(sequence_features) > 0:
                        sequences.append({
                            'features': np.array(sequence_features),
                            'labels': np.array(sequence_labels)
                        })
        
        logger.info(f"Built {len(sequences)} sequences")
        
        # Pad sequences
        X_padded = []
        y_padded = []
        masks = []
        
        for seq in sequences:
            features = seq['features']
            labels = seq['labels']
            seq_len = len(features)
            
            if seq_len > self.max_sequence_length:
                features = features[:self.max_sequence_length]
                labels = labels[:self.max_sequence_length]
                seq_len = self.max_sequence_length
            
            # Pad
            pad_len = self.max_sequence_length - seq_len
            features_padded = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            labels_padded = np.pad(labels, (0, pad_len), mode='constant', constant_values=-1)
            mask = np.array([1] * seq_len + [0] * pad_len)
            
            X_padded.append(features_padded)
            y_padded.append(labels_padded)
            masks.append(mask)
        
        X = np.stack(X_padded).astype(np.float32)
        y = np.stack(y_padded).astype(np.int64)
        masks = np.stack(masks).astype(np.bool_)
        
        # Save
        np.save(self.output_dir / 'X_seq.npy', X)
        np.save(self.output_dir / 'y_seq.npy', y)
        np.save(self.output_dir / 'masks_seq.npy', masks)
        
        logger.info(f"Sequence dataset saved: X={X.shape}, y={y.shape}, masks={masks.shape}")
        
        return {
            'num_sequences': len(sequences),
            'max_sequence_length': self.max_sequence_length,
            'feature_dim': X.shape[2]
        }


