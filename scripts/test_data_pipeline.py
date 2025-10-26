#!/usr/bin/env python3
"""
データ変換パイプラインのテストスクリプト

サンプルXMLファイルを使用して、全てのコンポーネントが正常に動作することを確認します。
"""

import sys
from pathlib import Path

# srcをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    EnhancedXMLParser,
    ComprehensiveGameStateTracker,
    AdvancedFeatureEncoderV2,
    DataValidator,
    ComprehensiveDatasetBuilder
)
import numpy as np


def test_enhanced_parser():
    """拡張XMLパーサーのテスト"""
    print("\n" + "="*80)
    print("Test 1: Enhanced XML Parser")
    print("="*80)
    
    parser = EnhancedXMLParser()
    
    # サンプルXMLファイルをパース
    sample_xml = Path(__file__).parent.parent / "2009080100gm-00e1-0000-63d644dd.xml"
    
    if not sample_xml.exists():
        print(f"❌ Sample XML not found: {sample_xml}")
        return False
    
    game = parser.parse_file(str(sample_xml))
    
    if game is None:
        print("❌ Failed to parse XML")
        return False
    
    print(f"✅ Game ID: {game.game_id}")
    print(f"✅ Players: {', '.join(game.player_names)}")
    print(f"✅ Rounds: {len(game.rounds)}")
    print(f"✅ Actions: {len(game.actions)}")
    print(f"✅ Final Scores: {game.final_scores}")
    
    # 統計情報
    stats = parser.compute_statistics([game])
    print(f"✅ Statistics:")
    print(f"   - Call rate: {stats['call_rate']:.2%}")
    print(f"   - Riichi rate: {stats['riichi_rate']:.2%}")
    print(f"   - Tsumogiri rate: {stats['tsumogiri_rate']:.2%}")
    
    return True


def test_game_state_tracker():
    """ゲーム状態追跡のテスト"""
    print("\n" + "="*80)
    print("Test 2: Comprehensive Game State Tracker")
    print("="*80)
    
    parser = EnhancedXMLParser()
    tracker = ComprehensiveGameStateTracker(
        enable_shanten_calc=False,  # Disable for testing
        enable_danger_estimation=True
    )
    
    sample_xml = Path(__file__).parent.parent / "2009080100gm-00e1-0000-63d644dd.xml"
    game = parser.parse_file(str(sample_xml))
    
    if game is None:
        print("❌ Failed to parse XML")
        return False
    
    # 状態抽出
    states = tracker.extract_all_states(game)
    
    if len(states) == 0:
        print("❌ No states extracted")
        return False
    
    print(f"✅ Extracted {len(states)} decision points")
    
    # 最初の状態を確認
    state = states[0]
    print(f"✅ Sample state:")
    print(f"   - Game ID: {state.game_id}")
    print(f"   - Round: {state.round_num}")
    print(f"   - Player: {state.player_id}")
    print(f"   - Hand size: {len(state.hand)}")
    print(f"   - Hand encoding: {state.hand_types.shape}")
    print(f"   - Discards: {len(state.discard_history)}")
    print(f"   - Label: {state.label_tile_type}")
    
    return True


def test_feature_encoder():
    """特徴量エンコーダーのテスト"""
    print("\n" + "="*80)
    print("Test 3: Advanced Feature Encoder V2")
    print("="*80)
    
    parser = EnhancedXMLParser()
    tracker = ComprehensiveGameStateTracker(enable_shanten_calc=False)
    encoder = AdvancedFeatureEncoderV2(
        draw_history_length=8,
        discard_history_length=32
    )
    
    sample_xml = Path(__file__).parent.parent / "2009080100gm-00e1-0000-63d644dd.xml"
    game = parser.parse_file(str(sample_xml))
    
    if game is None:
        print("❌ Failed to parse XML")
        return False
    
    states = tracker.extract_all_states(game)
    
    if len(states) == 0:
        print("❌ No states extracted")
        return False
    
    # 特徴量エンコード
    features = encoder.encode_state(states[0])
    
    print(f"✅ Feature dimension: {encoder.total_dim}")
    print(f"✅ Feature vector shape: {features.shape}")
    print(f"✅ Feature vector dtype: {features.dtype}")
    print(f"✅ Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"✅ Feature mean: {features.mean():.4f}")
    print(f"✅ Feature std: {features.std():.4f}")
    
    # NaN/Infチェック
    has_nan = np.any(np.isnan(features))
    has_inf = np.any(np.isinf(features))
    
    if has_nan or has_inf:
        print(f"❌ Invalid values: NaN={has_nan}, Inf={has_inf}")
        return False
    
    print("✅ No NaN/Inf values")
    
    return True


def test_data_validator():
    """データ検証のテスト"""
    print("\n" + "="*80)
    print("Test 4: Data Validator")
    print("="*80)
    
    parser = EnhancedXMLParser()
    validator = DataValidator()
    
    sample_xml = Path(__file__).parent.parent / "2009080100gm-00e1-0000-63d644dd.xml"
    game = parser.parse_file(str(sample_xml))
    
    if game is None:
        print("❌ Failed to parse XML")
        return False
    
    # ゲーム検証
    report = validator.validate_parsed_game(game)
    
    print(f"✅ Validation passed: {report.passed}")
    print(f"✅ Total checks: {report.total_checks}")
    print(f"✅ Errors: {len(report.errors)}")
    print(f"✅ Warnings: {len(report.warnings)}")
    
    if report.errors:
        print("❌ Validation errors found:")
        for error in report.errors[:5]:
            print(f"   - {error.message}")
        return False
    
    return True


def test_dataset_builder_small():
    """データセットビルダーのテスト（小規模）"""
    print("\n" + "="*80)
    print("Test 5: Comprehensive Dataset Builder (Small)")
    print("="*80)
    
    # テスト用の出力ディレクトリ
    output_dir = Path(__file__).parent.parent / "data" / "test_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # XMLディレクトリ（サンプルXMLがあるディレクトリ）
    xml_dir = Path(__file__).parent.parent
    
    config = {
        'draw_history_length': 4,  # 小さめに設定
        'discard_history_length': 16,
        'enable_shanten_calc': False,
        'enable_danger_estimation': True,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'random_seed': 42,
    }
    
    builder = ComprehensiveDatasetBuilder(
        xml_dir=str(xml_dir),
        output_dir=str(output_dir),
        config=config
    )
    
    try:
        # 1ファイルのみで構築
        stats = builder.build_complete_dataset(
            max_games=1,
            validate=True,
            show_progress=False
        )
        
        print(f"✅ Dataset built successfully")
        print(f"✅ Total samples: {stats['dataset_info']['total_samples']}")
        print(f"✅ Feature dimension: {stats['dataset_info']['feature_dimension']}")
        print(f"✅ Train samples: {stats['split_statistics']['train_size']}")
        print(f"✅ Val samples: {stats['split_statistics']['val_size']}")
        print(f"✅ Test samples: {stats['split_statistics']['test_size']}")
        
        # データ読み込み確認
        X_train, y_train, metadata = builder.load_dataset('train')
        print(f"✅ Loaded train data: X={X_train.shape}, y={y_train.shape}")
        
        # クリーンアップ
        import shutil
        shutil.rmtree(output_dir)
        print(f"✅ Cleaned up test directory")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全てのテストを実行"""
    print("="*80)
    print("Data Conversion Pipeline Test Suite")
    print("="*80)
    
    tests = [
        ("Enhanced XML Parser", test_enhanced_parser),
        ("Game State Tracker", test_game_state_tracker),
        ("Feature Encoder", test_feature_encoder),
        ("Data Validator", test_data_validator),
        ("Dataset Builder", test_dataset_builder_small),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # サマリー
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("="*80)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

