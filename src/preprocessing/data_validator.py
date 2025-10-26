"""Data Validation Module for Mahjong AI Pipeline.

データ変換パイプラインの各ステップでデータ整合性を検証。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
from pathlib import Path

from .enhanced_parser import EnhancedGame, EnhancedRoundState, EnhancedGameAction
from .game_state_tracker import ComprehensiveGameState

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """検証エラー情報"""
    error_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    game_id: Optional[str] = None
    round_num: Optional[int] = None
    context: Optional[Dict] = None


@dataclass
class ValidationReport:
    """検証レポート"""
    passed: bool
    total_checks: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    infos: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, error: ValidationError):
        """Add an error to the report."""
        if error.severity == 'error':
            self.errors.append(error)
            self.passed = False
        elif error.severity == 'warning':
            self.warnings.append(error)
        else:
            self.infos.append(error)
    
    def get_summary(self) -> str:
        """Get summary string."""
        return (
            f"Validation Report:\n"
            f"  Passed: {self.passed}\n"
            f"  Total Checks: {self.total_checks}\n"
            f"  Errors: {len(self.errors)}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f"  Infos: {len(self.infos)}"
        )
    
    def print_errors(self, max_errors: int = 10):
        """Print first N errors."""
        if not self.errors:
            logger.info("No errors found!")
            return
        
        logger.error(f"Found {len(self.errors)} errors:")
        for i, error in enumerate(self.errors[:max_errors]):
            logger.error(f"  {i+1}. [{error.error_type}] {error.message}")
            if error.game_id:
                logger.error(f"      Game: {error.game_id}, Round: {error.round_num}")


class DataValidator:
    """データ検証クラス"""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize the validator.
        
        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
    
    def validate_parsed_game(self, game: EnhancedGame) -> ValidationReport:
        """Validate a parsed game.
        
        Args:
            game: EnhancedGame object
            
        Returns:
            ValidationReport
        """
        report = ValidationReport(passed=True, total_checks=0)
        
        # Check 1: Player count
        report.total_checks += 1
        if len(game.player_names) != 4:
            report.add_error(ValidationError(
                error_type='player_count',
                severity='error',
                message=f'Expected 4 players, got {len(game.player_names)}',
                game_id=game.game_id
            ))
        
        # Check 2: Round count
        report.total_checks += 1
        if len(game.rounds) == 0:
            report.add_error(ValidationError(
                error_type='round_count',
                severity='error',
                message='No rounds found in game',
                game_id=game.game_id
            ))
        
        # Check 3: Score validity
        report.total_checks += 1
        if len(game.final_scores) != 4:
            report.add_error(ValidationError(
                error_type='score_count',
                severity='error',
                message=f'Expected 4 final scores, got {len(game.final_scores)}',
                game_id=game.game_id
            ))
        
        # Check 4: Each round
        for round_idx, round_state in enumerate(game.rounds):
            round_report = self._validate_round_state(round_state, game.game_id, round_idx)
            report.errors.extend(round_report.errors)
            report.warnings.extend(round_report.warnings)
            report.infos.extend(round_report.infos)
            report.total_checks += round_report.total_checks
            if not round_report.passed:
                report.passed = False
        
        # Check 5: Action consistency
        action_report = self._validate_actions(game.actions, game.game_id)
        report.errors.extend(action_report.errors)
        report.warnings.extend(action_report.warnings)
        report.total_checks += action_report.total_checks
        if not action_report.passed:
            report.passed = False
        
        return report
    
    def _validate_round_state(self, round_state: EnhancedRoundState,
                             game_id: str, round_idx: int) -> ValidationReport:
        """Validate a round state."""
        report = ValidationReport(passed=True, total_checks=0)
        
        # Check 1: Initial hands count
        report.total_checks += 1
        if len(round_state.initial_hands) != 4:
            report.add_error(ValidationError(
                error_type='hand_count',
                severity='error',
                message=f'Expected 4 initial hands, got {len(round_state.initial_hands)}',
                game_id=game_id,
                round_num=round_idx
            ))
        
        # Check 2: Hand size (13 tiles each)
        for player_id, hand in enumerate(round_state.initial_hands):
            report.total_checks += 1
            if len(hand) != 13:
                report.add_error(ValidationError(
                    error_type='hand_size',
                    severity='error',
                    message=f'Player {player_id} has {len(hand)} tiles, expected 13',
                    game_id=game_id,
                    round_num=round_idx
                ))
        
        # Check 3: Tile ID range
        for player_id, hand in enumerate(round_state.initial_hands):
            for tile in hand:
                report.total_checks += 1
                if not (0 <= tile <= 135):
                    report.add_error(ValidationError(
                        error_type='tile_range',
                        severity='error',
                        message=f'Invalid tile ID {tile} (must be 0-135)',
                        game_id=game_id,
                        round_num=round_idx
                    ))
        
        # Check 4: Tile count (no more than 4 of each type)
        report.total_checks += 1
        tile_counts = np.zeros(34, dtype=np.int32)
        for hand in round_state.initial_hands:
            for tile in hand:
                tile_type = tile // 4
                if 0 <= tile_type < 34:
                    tile_counts[tile_type] += 1
        
        if np.any(tile_counts > 4):
            invalid_types = np.where(tile_counts > 4)[0]
            report.add_error(ValidationError(
                error_type='tile_count',
                severity='error',
                message=f'Tile types {invalid_types.tolist()} appear more than 4 times in initial hands',
                game_id=game_id,
                round_num=round_idx
            ))
        
        # Check 5: Dealer validity
        report.total_checks += 1
        if not (0 <= round_state.dealer < 4):
            report.add_error(ValidationError(
                error_type='dealer_invalid',
                severity='error',
                message=f'Invalid dealer {round_state.dealer} (must be 0-3)',
                game_id=game_id,
                round_num=round_idx
            ))
        
        # Check 6: Dora indicator range
        for dora in round_state.dora_indicators:
            report.total_checks += 1
            if not (0 <= dora <= 135):
                report.add_error(ValidationError(
                    error_type='dora_range',
                    severity='error',
                    message=f'Invalid dora indicator {dora} (must be 0-135)',
                    game_id=game_id,
                    round_num=round_idx
                ))
        
        # Check 7: Score validity
        report.total_checks += 1
        if len(round_state.scores) != 4:
            report.add_error(ValidationError(
                error_type='score_count',
                severity='error',
                message=f'Expected 4 scores, got {len(round_state.scores)}',
                game_id=game_id,
                round_num=round_idx
            ))
        
        for score in round_state.scores:
            report.total_checks += 1
            if score < 0 or score > 1000:  # Scores in 100s
                report.add_error(ValidationError(
                    error_type='score_range',
                    severity='warning',
                    message=f'Unusual score {score * 100} (in 100s: {score})',
                    game_id=game_id,
                    round_num=round_idx
                ))
        
        return report
    
    def _validate_actions(self, actions: List[EnhancedGameAction],
                         game_id: str) -> ValidationReport:
        """Validate action sequence."""
        report = ValidationReport(passed=True, total_checks=0)
        
        if len(actions) == 0:
            report.add_error(ValidationError(
                error_type='no_actions',
                severity='error',
                message='No actions found',
                game_id=game_id
            ))
            return report
        
        # Check 1: Player ID validity
        for action in actions:
            report.total_checks += 1
            if not (0 <= action.player_id < 4):
                report.add_error(ValidationError(
                    error_type='player_id_invalid',
                    severity='error',
                    message=f'Invalid player_id {action.player_id} (must be 0-3)',
                    game_id=game_id
                ))
        
        # Check 2: Tile ID range (for draw/discard)
        for action in actions:
            if action.tile is not None:
                report.total_checks += 1
                if not (0 <= action.tile <= 135):
                    report.add_error(ValidationError(
                        error_type='action_tile_range',
                        severity='error',
                        message=f'Invalid tile {action.tile} in {action.action_type} action',
                        game_id=game_id
                    ))
        
        # Check 3: Action type validity
        valid_action_types = {
            'draw', 'discard', 'chi', 'pon', 'kan', 'ankan', 'kakan', 
            'riichi', 'tsumo', 'ron', 'dora'
        }
        for action in actions:
            report.total_checks += 1
            if action.action_type not in valid_action_types:
                report.add_error(ValidationError(
                    error_type='action_type_invalid',
                    severity='warning',
                    message=f'Unknown action type: {action.action_type}',
                    game_id=game_id
                ))
        
        return report
    
    def validate_game_state(self, state: ComprehensiveGameState) -> ValidationReport:
        """Validate a comprehensive game state.
        
        Args:
            state: ComprehensiveGameState object
            
        Returns:
            ValidationReport
        """
        report = ValidationReport(passed=True, total_checks=0)
        
        # Check 1: Hand encoding validity
        report.total_checks += 1
        if state.hand_types.shape != (34,):
            report.add_error(ValidationError(
                error_type='hand_shape',
                severity='error',
                message=f'Hand encoding shape {state.hand_types.shape} != (34,)',
                game_id=state.game_id,
                round_num=state.round_num
            ))
        
        # Check 2: Hand tile count
        report.total_checks += 1
        hand_count = state.hand_types.sum()
        if not (0 <= hand_count <= 14):
            report.add_error(ValidationError(
                error_type='hand_count',
                severity='error',
                message=f'Hand has {hand_count} tiles (expected 13-14)',
                game_id=state.game_id,
                round_num=state.round_num
            ))
        
        # Check 3: No tile type exceeds 4
        report.total_checks += 1
        if np.any(state.hand_types > 4):
            report.add_error(ValidationError(
                error_type='hand_tile_excess',
                severity='error',
                message='Hand contains more than 4 of a tile type',
                game_id=state.game_id,
                round_num=state.round_num
            ))
        
        # Check 4: Label validity
        report.total_checks += 1
        if not (0 <= state.label_tile_type < 34):
            report.add_error(ValidationError(
                error_type='label_range',
                severity='error',
                message=f'Label tile type {state.label_tile_type} out of range (0-33)',
                game_id=state.game_id,
                round_num=state.round_num
            ))
        
        # Check 5: Player ID validity
        report.total_checks += 1
        if not (0 <= state.player_id < 4):
            report.add_error(ValidationError(
                error_type='player_id',
                severity='error',
                message=f'Invalid player_id {state.player_id}',
                game_id=state.game_id,
                round_num=state.round_num
            ))
        
        # Check 6: Scores count
        report.total_checks += 1
        if len(state.scores) != 4:
            report.add_error(ValidationError(
                error_type='scores_count',
                severity='error',
                message=f'Expected 4 scores, got {len(state.scores)}',
                game_id=state.game_id,
                round_num=state.round_num
            ))
        
        return report
    
    def validate_feature_vector(self, features: np.ndarray, 
                               expected_dim: int) -> ValidationReport:
        """Validate a feature vector.
        
        Args:
            features: Feature vector
            expected_dim: Expected dimension
            
        Returns:
            ValidationReport
        """
        report = ValidationReport(passed=True, total_checks=0)
        
        # Check 1: Shape
        report.total_checks += 1
        if features.shape != (expected_dim,):
            report.add_error(ValidationError(
                error_type='feature_shape',
                severity='error',
                message=f'Feature shape {features.shape} != ({expected_dim},)'
            ))
        
        # Check 2: No NaN
        report.total_checks += 1
        if np.any(np.isnan(features)):
            nan_count = np.sum(np.isnan(features))
            report.add_error(ValidationError(
                error_type='feature_nan',
                severity='error',
                message=f'Feature contains {nan_count} NaN values'
            ))
        
        # Check 3: No Inf
        report.total_checks += 1
        if np.any(np.isinf(features)):
            inf_count = np.sum(np.isinf(features))
            report.add_error(ValidationError(
                error_type='feature_inf',
                severity='error',
                message=f'Feature contains {inf_count} Inf values'
            ))
        
        # Check 4: Range check (assuming normalized features are in [0, 1])
        report.total_checks += 1
        if np.any(features < -1.0) or np.any(features > 2.0):
            out_of_range = np.sum((features < -1.0) | (features > 2.0))
            report.add_error(ValidationError(
                error_type='feature_range',
                severity='warning',
                message=f'{out_of_range} features out of expected range [-1, 2]'
            ))
        
        return report
    
    def validate_dataset_split(self, train_games: Set[str], val_games: Set[str],
                              test_games: Set[str]) -> ValidationReport:
        """Validate dataset split (no data leakage).
        
        Args:
            train_games: Set of game IDs in training set
            val_games: Set of game IDs in validation set
            test_games: Set of game IDs in test set
            
        Returns:
            ValidationReport
        """
        report = ValidationReport(passed=True, total_checks=0)
        
        # Check 1: No overlap between train and val
        report.total_checks += 1
        train_val_overlap = train_games & val_games
        if train_val_overlap:
            report.add_error(ValidationError(
                error_type='data_leak_train_val',
                severity='error',
                message=f'Train and val sets overlap: {len(train_val_overlap)} games',
                context={'overlap_games': list(train_val_overlap)[:10]}
            ))
        
        # Check 2: No overlap between train and test
        report.total_checks += 1
        train_test_overlap = train_games & test_games
        if train_test_overlap:
            report.add_error(ValidationError(
                error_type='data_leak_train_test',
                severity='error',
                message=f'Train and test sets overlap: {len(train_test_overlap)} games',
                context={'overlap_games': list(train_test_overlap)[:10]}
            ))
        
        # Check 3: No overlap between val and test
        report.total_checks += 1
        val_test_overlap = val_games & test_games
        if val_test_overlap:
            report.add_error(ValidationError(
                error_type='data_leak_val_test',
                severity='error',
                message=f'Val and test sets overlap: {len(val_test_overlap)} games',
                context={'overlap_games': list(val_test_overlap)[:10]}
            ))
        
        # Check 4: All sets are non-empty
        report.total_checks += 1
        if len(train_games) == 0:
            report.add_error(ValidationError(
                error_type='empty_train',
                severity='error',
                message='Training set is empty'
            ))
        
        report.total_checks += 1
        if len(val_games) == 0:
            report.add_error(ValidationError(
                error_type='empty_val',
                severity='error',
                message='Validation set is empty'
            ))
        
        report.total_checks += 1
        if len(test_games) == 0:
            report.add_error(ValidationError(
                error_type='empty_test',
                severity='error',
                message='Test set is empty'
            ))
        
        # Check 5: Split ratios are reasonable
        total = len(train_games) + len(val_games) + len(test_games)
        if total > 0:
            train_ratio = len(train_games) / total
            val_ratio = len(val_games) / total
            test_ratio = len(test_games) / total
            
            report.total_checks += 1
            if train_ratio < 0.5 or train_ratio > 0.95:
                report.add_error(ValidationError(
                    error_type='train_ratio',
                    severity='warning',
                    message=f'Train ratio {train_ratio:.2%} is unusual (expected 50-95%)'
                ))
            
            report.total_checks += 1
            if val_ratio < 0.05 or val_ratio > 0.3:
                report.add_error(ValidationError(
                    error_type='val_ratio',
                    severity='warning',
                    message=f'Val ratio {val_ratio:.2%} is unusual (expected 5-30%)'
                ))
            
            report.total_checks += 1
            if test_ratio < 0.05 or test_ratio > 0.3:
                report.add_error(ValidationError(
                    error_type='test_ratio',
                    severity='warning',
                    message=f'Test ratio {test_ratio:.2%} is unusual (expected 5-30%)'
                ))
        
        return report
    
    def validate_numpy_arrays(self, X: np.ndarray, y: np.ndarray) -> ValidationReport:
        """Validate numpy arrays for dataset.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            ValidationReport
        """
        report = ValidationReport(passed=True, total_checks=0)
        
        # Check 1: Shape consistency
        report.total_checks += 1
        if X.shape[0] != y.shape[0]:
            report.add_error(ValidationError(
                error_type='array_size_mismatch',
                severity='error',
                message=f'X has {X.shape[0]} samples, y has {y.shape[0]} samples'
            ))
        
        # Check 2: No NaN in X
        report.total_checks += 1
        if np.any(np.isnan(X)):
            nan_samples = np.sum(np.any(np.isnan(X), axis=1))
            report.add_error(ValidationError(
                error_type='X_contains_nan',
                severity='error',
                message=f'{nan_samples} samples in X contain NaN'
            ))
        
        # Check 3: No Inf in X
        report.total_checks += 1
        if np.any(np.isinf(X)):
            inf_samples = np.sum(np.any(np.isinf(X), axis=1))
            report.add_error(ValidationError(
                error_type='X_contains_inf',
                severity='error',
                message=f'{inf_samples} samples in X contain Inf'
            ))
        
        # Check 4: Label range
        report.total_checks += 1
        if np.any(y < 0) or np.any(y >= 34):
            invalid_labels = np.sum((y < 0) | (y >= 34))
            report.add_error(ValidationError(
                error_type='label_out_of_range',
                severity='error',
                message=f'{invalid_labels} labels out of range [0, 33]'
            ))
        
        # Check 5: Label distribution
        report.total_checks += 1
        unique_labels = np.unique(y)
        if len(unique_labels) < 34:
            missing_labels = set(range(34)) - set(unique_labels.tolist())
            report.add_error(ValidationError(
                error_type='incomplete_label_coverage',
                severity='warning',
                message=f'{len(missing_labels)} tile types never appear as labels',
                context={'missing_labels': sorted(missing_labels)}
            ))
        
        return report

