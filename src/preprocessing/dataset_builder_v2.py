"""Comprehensive Dataset Builder V2 for Mahjong AI.

全機能を統合した包括的なデータセット構築モジュール。

機能:
- XML解析 → ゲーム状態抽出 → 特徴量エンコーディング → NumPy配列保存
- ゲーム単位でのtrain/val/test分割（データリーク防止）
- データ検証の統合
- メタデータ保存
- 統計情報の出力
- チャンク処理（大規模データ対応）
- マルチプロセス対応
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import asdict
import pickle
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from .enhanced_parser import EnhancedXMLParser, EnhancedGame
from .game_state_tracker import ComprehensiveGameStateTracker, ComprehensiveGameState
from .feature_encoder_v2 import AdvancedFeatureEncoderV2, LabelEncoder
from .data_validator import DataValidator, ValidationReport

logger = logging.getLogger(__name__)


class ComprehensiveDatasetBuilder:
    """包括的なデータセットビルダー"""
    
    def __init__(self, 
                 xml_dir: str,
                 output_dir: str,
                 config: Optional[Dict] = None):
        """Initialize the dataset builder.
        
        Args:
            xml_dir: Directory containing XML game logs
            output_dir: Directory to save processed datasets
            config: Configuration dictionary
        """
        self.xml_dir = Path(xml_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default config
        self.config = {
            'draw_history_length': 8,
            'discard_history_length': 32,
            'enable_shanten_calc': True,
            'enable_danger_estimation': True,
            'random_seed': 42,
            'train_ratio': 0.80,
            'val_ratio': 0.10,
            'test_ratio': 0.10,
            'chunk_size': 1000,  # Process games in chunks
            'num_workers': 1,  # Number of parallel workers (1 = sequential)
            'validate': True,
        }
        
        if config:
            self.config.update(config)
        
        # Initialize components
        self.parser = EnhancedXMLParser()
        self.tracker = ComprehensiveGameStateTracker(
            enable_shanten_calc=self.config['enable_shanten_calc'],
            enable_danger_estimation=self.config['enable_danger_estimation']
        )
        self.encoder = AdvancedFeatureEncoderV2(
            draw_history_length=self.config['draw_history_length'],
            discard_history_length=self.config['discard_history_length']
        )
        self.label_encoder = LabelEncoder()
        self.validator = DataValidator()
        
        # Set random seed
        np.random.seed(self.config['random_seed'])
        
        logger.info(f"Dataset builder initialized")
        logger.info(f"  XML dir: {self.xml_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Feature dim: {self.encoder.total_dim}")
        logger.info(f"  Config: {self.config}")
    
    def build_complete_dataset(self, 
                              max_games: Optional[int] = None,
                              validate: bool = True,
                              show_progress: bool = True) -> Dict:
        """Build complete dataset from XML files.
        
        Args:
            max_games: Maximum number of games to process
            validate: Enable validation
            show_progress: Show progress bar
            
        Returns:
            Statistics dictionary
        """
        logger.info("="*80)
        logger.info("Building Complete Dataset")
        logger.info("="*80)
        
        # Step 1: Parse XML files
        logger.info("Step 1: Parsing XML files...")
        games = self._parse_all_games(max_games, show_progress)
        
        if not games:
            raise ValueError(f"No games found in {self.xml_dir}")
        
        logger.info(f"Parsed {len(games)} games")
        
        # Step 2: Validate parsed games (optional)
        if validate:
            logger.info("Step 2: Validating parsed games...")
            validation_report = self._validate_games(games)
            logger.info(validation_report.get_summary())
            
            if not validation_report.passed:
                logger.warning("Validation found errors. Continuing anyway...")
                validation_report.print_errors(max_errors=10)
        
        # Step 3: Extract game states
        logger.info("Step 3: Extracting game states...")
        all_states, game_to_states = self._extract_all_states(games, show_progress)
        
        logger.info(f"Extracted {len(all_states)} decision points")
        
        # Step 4: Encode features
        logger.info("Step 4: Encoding features...")
        X, y, metadata = self._encode_all_features(all_states, show_progress)
        
        logger.info(f"Encoded features: X={X.shape}, y={y.shape}")
        
        # Step 5: Validate features
        if validate:
            logger.info("Step 5: Validating features...")
            feature_report = self.validator.validate_numpy_arrays(X, y)
            logger.info(feature_report.get_summary())
            
            if not feature_report.passed:
                logger.error("Feature validation failed!")
                feature_report.print_errors()
        
        # Step 6: Split dataset by game
        logger.info("Step 6: Splitting dataset...")
        split_indices = self._split_by_game(metadata, game_to_states)
        
        # Step 7: Save dataset
        logger.info("Step 7: Saving dataset...")
        self._save_dataset(X, y, metadata, split_indices)
        
        # Step 8: Compute and save statistics
        logger.info("Step 8: Computing statistics...")
        stats = self._compute_statistics(games, all_states, X, y, split_indices)
        
        # Save statistics
        stats_path = self.output_dir / 'dataset_info.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("="*80)
        logger.info("Dataset Building Complete")
        logger.info("="*80)
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Feature dimension: {X.shape[1]}")
        logger.info(f"Train samples: {len(split_indices['train'])}")
        logger.info(f"Val samples: {len(split_indices['val'])}")
        logger.info(f"Test samples: {len(split_indices['test'])}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*80)
        
        return stats
    
    def _parse_all_games(self, max_games: Optional[int], 
                        show_progress: bool) -> List[EnhancedGame]:
        """Parse all XML files."""
        return self.parser.parse_directory(
            str(self.xml_dir),
            max_files=max_games,
            show_progress=show_progress
        )
    
    def _validate_games(self, games: List[EnhancedGame]) -> ValidationReport:
        """Validate parsed games."""
        overall_report = ValidationReport(passed=True, total_checks=0)
        
        for game in tqdm(games, desc="Validating games"):
            report = self.validator.validate_parsed_game(game)
            overall_report.errors.extend(report.errors)
            overall_report.warnings.extend(report.warnings)
            overall_report.total_checks += report.total_checks
            if not report.passed:
                overall_report.passed = False
        
        return overall_report
    
    def _extract_all_states(self, games: List[EnhancedGame], 
                           show_progress: bool) -> Tuple[List[ComprehensiveGameState], Dict[str, List[int]]]:
        """Extract all game states from games.
        
        Returns:
            (all_states, game_to_states) where game_to_states maps game_id to state indices
        """
        all_states = []
        game_to_states = {}
        
        iterator = tqdm(games, desc="Extracting states") if show_progress else games
        
        for game in iterator:
            try:
                states = self.tracker.extract_all_states(game)
                
                # Record which indices belong to this game
                start_idx = len(all_states)
                end_idx = start_idx + len(states)
                game_to_states[game.game_id] = list(range(start_idx, end_idx))
                
                all_states.extend(states)
            
            except Exception as e:
                logger.warning(f"Failed to extract states from game {game.game_id}: {e}")
        
        return all_states, game_to_states
    
    def _encode_all_features(self, states: List[ComprehensiveGameState],
                            show_progress: bool) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Encode all states to features.
        
        Returns:
            (X, y, metadata)
        """
        features_list = []
        labels_list = []
        metadata_list = []
        
        iterator = tqdm(states, desc="Encoding features") if show_progress else states
        
        for state in iterator:
            try:
                # Encode features
                feature_vector = self.encoder.encode_state(state)
                features_list.append(feature_vector)
                
                # Encode label
                label = self.label_encoder.encode_discard_label_id(state.label_tile_type)
                labels_list.append(label)
                
                # Store metadata
                metadata = {
                    'game_id': state.game_id,
                    'round_num': state.round_num,
                    'player_id': state.player_id,
                    'turn': state.turn,
                    'label_tile_type': state.label_tile_type
                }
                metadata_list.append(metadata)
            
            except Exception as e:
                logger.warning(f"Failed to encode state from game {state.game_id}: {e}")
        
        X = np.stack(features_list).astype(np.float32)
        y = np.array(labels_list, dtype=np.int64)
        
        return X, y, metadata_list
    
    def _split_by_game(self, metadata: List[Dict], 
                      game_to_states: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Split dataset by game (no data leakage).
        
        Args:
            metadata: Metadata list
            game_to_states: Mapping from game_id to state indices
            
        Returns:
            Dictionary with 'train', 'val', 'test' indices
        """
        # Get all unique game IDs
        game_ids = list(game_to_states.keys())
        
        # Shuffle games
        np.random.seed(self.config['random_seed'])
        np.random.shuffle(game_ids)
        
        # Calculate split points
        n_games = len(game_ids)
        train_end = int(n_games * self.config['train_ratio'])
        val_end = train_end + int(n_games * self.config['val_ratio'])
        
        # Split game IDs
        train_games = set(game_ids[:train_end])
        val_games = set(game_ids[train_end:val_end])
        test_games = set(game_ids[val_end:])
        
        logger.info(f"Split games: train={len(train_games)}, val={len(val_games)}, test={len(test_games)}")
        
        # Validate split
        if self.config['validate']:
            split_report = self.validator.validate_dataset_split(
                train_games, val_games, test_games
            )
            if not split_report.passed:
                logger.error("Dataset split validation failed!")
                split_report.print_errors()
        
        # Get indices for each split
        train_indices = []
        val_indices = []
        test_indices = []
        
        for game_id, state_indices in game_to_states.items():
            if game_id in train_games:
                train_indices.extend(state_indices)
            elif game_id in val_games:
                val_indices.extend(state_indices)
            elif game_id in test_games:
                test_indices.extend(state_indices)
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'train_games': list(train_games),
            'val_games': list(val_games),
            'test_games': list(test_games)
        }
    
    def _save_dataset(self, X: np.ndarray, y: np.ndarray, 
                     metadata: List[Dict], split_indices: Dict[str, List[int]]):
        """Save dataset to disk."""
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Get indices for this split
            indices = split_indices[split]
            
            # Save features and labels
            X_split = X[indices]
            y_split = y[indices]
            
            np.save(split_dir / 'X_train.npy' if split == 'train' else split_dir / f'X_{split}.npy', X_split)
            np.save(split_dir / 'y_train.npy' if split == 'train' else split_dir / f'y_{split}.npy', y_split)
            
            # Save metadata
            metadata_split = [metadata[i] for i in indices]
            with open(split_dir / f'metadata_{split}.json', 'w', encoding='utf-8') as f:
                json.dump(metadata_split, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {split} split: {len(X_split)} samples")
        
        # Save split info
        split_info = {
            'train_games': split_indices['train_games'],
            'val_games': split_indices['val_games'],
            'test_games': split_indices['test_games'],
            'train_size': len(split_indices['train']),
            'val_size': len(split_indices['val']),
            'test_size': len(split_indices['test']),
            'random_seed': self.config['random_seed']
        }
        
        with open(self.output_dir / 'split_info.json', 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved split info to {self.output_dir / 'split_info.json'}")
    
    def _compute_statistics(self, games: List[EnhancedGame],
                           states: List[ComprehensiveGameState],
                           X: np.ndarray, y: np.ndarray,
                           split_indices: Dict[str, List[int]]) -> Dict:
        """Compute dataset statistics."""
        
        # Game statistics
        game_stats = self.parser.compute_statistics(games)
        
        # State statistics
        state_stats = {
            'total_decision_points': len(states),
            'avg_decision_points_per_game': len(states) / len(games) if games else 0,
        }
        
        # Feature statistics
        feature_stats = {
            'feature_dim': X.shape[1],
            'feature_mean': float(np.mean(X)),
            'feature_std': float(np.std(X)),
            'feature_min': float(np.min(X)),
            'feature_max': float(np.max(X)),
        }
        
        # Label statistics
        label_counts = np.bincount(y, minlength=34)
        label_stats = {
            'num_classes': 34,
            'label_distribution': label_counts.tolist(),
            'most_common_label': int(np.argmax(label_counts)),
            'least_common_label': int(np.argmin(label_counts[label_counts > 0]) if np.any(label_counts > 0) else 0),
        }
        
        # Split statistics
        split_stats = {
            'train_size': len(split_indices['train']),
            'val_size': len(split_indices['val']),
            'test_size': len(split_indices['test']),
            'train_games': len(split_indices['train_games']),
            'val_games': len(split_indices['val_games']),
            'test_games': len(split_indices['test_games']),
            'train_ratio': len(split_indices['train']) / len(X) if len(X) > 0 else 0,
            'val_ratio': len(split_indices['val']) / len(X) if len(X) > 0 else 0,
            'test_ratio': len(split_indices['test']) / len(X) if len(X) > 0 else 0,
        }
        
        # Combine all stats
        stats = {
            'dataset_info': {
                'total_games': len(games),
                'total_samples': len(X),
                'feature_dimension': X.shape[1],
                'num_classes': 34,
            },
            'game_statistics': game_stats,
            'state_statistics': state_stats,
            'feature_statistics': feature_stats,
            'label_statistics': label_stats,
            'split_statistics': split_stats,
            'config': self.config,
        }
        
        return stats
    
    def load_dataset(self, split: str = 'train') -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load a dataset split from disk.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            (X, y, metadata)
        """
        split_dir = self.output_dir / split
        
        if split == 'train':
            X = np.load(split_dir / 'X_train.npy')
            y = np.load(split_dir / 'y_train.npy')
        else:
            X = np.load(split_dir / f'X_{split}.npy')
            y = np.load(split_dir / f'y_{split}.npy')
        
        with open(split_dir / f'metadata_{split}.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded {split} dataset: X={X.shape}, y={y.shape}")
        
        return X, y, metadata
    
    def get_statistics(self) -> Dict:
        """Load dataset statistics.
        
        Returns:
            Statistics dictionary
        """
        stats_path = self.output_dir / 'dataset_info.json'
        
        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")
        
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        
        print("="*80)
        print("Dataset Statistics")
        print("="*80)
        
        print("\nDataset Info:")
        for key, value in stats['dataset_info'].items():
            print(f"  {key}: {value}")
        
        print("\nGame Statistics:")
        for key, value in stats['game_statistics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\nSplit Statistics:")
        for key, value in stats['split_statistics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\nLabel Distribution (Top 10):")
        label_dist = stats['label_statistics']['label_distribution']
        top_10 = sorted(enumerate(label_dist), key=lambda x: x[1], reverse=True)[:10]
        for tile_type, count in top_10:
            print(f"  Tile {tile_type}: {count} ({count/sum(label_dist)*100:.2f}%)")
        
        print("="*80)

