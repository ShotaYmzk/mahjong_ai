#!/usr/bin/env python3
"""
包括的データセット構築スクリプト

天鳳XMLログから教師あり学習用の固定長ベクトルデータセットを構築します。

使用例:
    # 基本的な使用（デフォルト設定）
    python scripts/build_comprehensive_dataset.py
    
    # カスタム設定
    python scripts/build_comprehensive_dataset.py \
        --xml-dir data/xml_logs \
        --output-dir data/processed_v2 \
        --max-games 10000 \
        --draw-history 8 \
        --discard-history 32
    
    # テスト実行（最初の100ゲームのみ）
    python scripts/build_comprehensive_dataset.py --max-games 100
"""

import sys
import argparse
import logging
from pathlib import Path

# srcをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import ComprehensiveDatasetBuilder


def setup_logging(log_file: Path):
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='天鳳XMLログから包括的データセットを構築',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # 基本的な使用
  python scripts/build_comprehensive_dataset.py
  
  # カスタムディレクトリ
  python scripts/build_comprehensive_dataset.py \\
      --xml-dir data/xml_logs \\
      --output-dir data/processed_v2
  
  # テスト実行（最初の100ゲーム）
  python scripts/build_comprehensive_dataset.py --max-games 100
  
  # 高度な設定
  python scripts/build_comprehensive_dataset.py \\
      --draw-history 16 \\
      --discard-history 64 \\
      --no-shanten \\
      --train-ratio 0.8 \\
      --val-ratio 0.1 \\
      --test-ratio 0.1

特徴量次元数:
  基本特徴量: 340次元
  ツモ履歴: draw_history × 34次元
  捨て牌履歴: discard_history × 39次元
  メタ特徴量: 31次元
  待ち・向聴数: 38次元
  捨て牌メタ: 170次元
  
  デフォルト (draw=8, discard=32): 2,099次元
        '''
    )
    
    # ディレクトリ設定
    parser.add_argument(
        '--xml-dir',
        type=str,
        default='/home/ubuntu/Documents/mahjong_ai/data/xml_logs',
        help='XMLファイルのディレクトリ'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/ubuntu/Documents/mahjong_ai/data/processed_v2',
        help='処理済みデータの出力ディレクトリ'
    )
    
    # データ設定
    parser.add_argument(
        '--max-games',
        type=int,
        default=None,
        help='処理する最大ゲーム数（None=全て）'
    )
    
    # 特徴量設定
    parser.add_argument(
        '--draw-history',
        type=int,
        default=8,
        help='ツモ履歴の長さ (k)'
    )
    
    parser.add_argument(
        '--discard-history',
        type=int,
        default=32,
        help='捨て牌履歴の長さ (m)'
    )
    
    parser.add_argument(
        '--no-shanten',
        action='store_true',
        help='向聴数計算を無効化'
    )
    
    parser.add_argument(
        '--no-danger',
        action='store_true',
        help='危険度推定を無効化'
    )
    
    # 分割設定
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.80,
        help='訓練データの割合'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.10,
        help='検証データの割合'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.10,
        help='テストデータの割合'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='乱数シード'
    )
    
    # 処理設定
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='データ検証を無効化'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='プログレスバーを非表示'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='dataset_building.log',
        help='ログファイルのパス'
    )
    
    args = parser.parse_args()
    
    # ログ設定
    log_file = Path(args.log_file)
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("包括的データセット構築スクリプト")
    logger.info("="*80)
    logger.info(f"XMLディレクトリ: {args.xml_dir}")
    logger.info(f"出力ディレクトリ: {args.output_dir}")
    if args.max_games:
        logger.info(f"最大ゲーム数: {args.max_games}")
    logger.info(f"ツモ履歴長: {args.draw_history}")
    logger.info(f"捨て牌履歴長: {args.discard_history}")
    logger.info(f"分割比率: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    logger.info("="*80)
    
    # 設定の検証
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        logger.error("エラー: train_ratio + val_ratio + test_ratio は 1.0 である必要があります")
        return 1
    
    # XMLディレクトリの確認
    xml_dir = Path(args.xml_dir)
    if not xml_dir.exists():
        logger.error(f"エラー: XMLディレクトリが存在しません: {xml_dir}")
        return 1
    
    # XMLファイルの確認
    xml_files = list(xml_dir.glob('*.xml'))
    if not xml_files:
        logger.error(f"エラー: XMLファイルが見つかりません: {xml_dir}")
        return 1
    
    logger.info(f"✅ XMLファイルを発見: {len(xml_files)} ファイル")
    
    # 設定を構築
    config = {
        'draw_history_length': args.draw_history,
        'discard_history_length': args.discard_history,
        'enable_shanten_calc': not args.no_shanten,
        'enable_danger_estimation': not args.no_danger,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'random_seed': args.random_seed,
        'validate': not args.no_validate,
    }
    
    # 特徴量次元数を計算して表示
    feature_dim = (
        340 +  # 基本特徴量
        args.draw_history * 34 +  # ツモ履歴
        args.discard_history * 39 +  # 捨て牌履歴
        31 +  # メタ特徴量
        38 +  # 待ち・向聴数
        170  # 捨て牌メタ
    )
    logger.info(f"特徴量次元数: {feature_dim}")
    logger.info(f"  基本特徴量: 340")
    logger.info(f"  ツモ履歴: {args.draw_history * 34} ({args.draw_history} × 34)")
    logger.info(f"  捨て牌履歴: {args.discard_history * 39} ({args.discard_history} × 39)")
    logger.info(f"  メタ特徴量: 31")
    logger.info(f"  待ち・向聴数: 38")
    logger.info(f"  捨て牌メタ: 170")
    logger.info("="*80)
    
    try:
        # データセットビルダーの初期化
        builder = ComprehensiveDatasetBuilder(
            xml_dir=args.xml_dir,
            output_dir=args.output_dir,
            config=config
        )
        
        # データセット構築
        logger.info("データセット構築を開始します...")
        stats = builder.build_complete_dataset(
            max_games=args.max_games,
            validate=not args.no_validate,
            show_progress=not args.no_progress
        )
        
        # 統計情報を表示
        logger.info("\n" + "="*80)
        logger.info("最終統計情報")
        logger.info("="*80)
        logger.info(f"総ゲーム数: {stats['dataset_info']['total_games']}")
        logger.info(f"総サンプル数: {stats['dataset_info']['total_samples']}")
        logger.info(f"特徴量次元数: {stats['dataset_info']['feature_dimension']}")
        logger.info("")
        logger.info(f"訓練データ: {stats['split_statistics']['train_size']} サンプル "
                   f"({stats['split_statistics']['train_ratio']:.1%})")
        logger.info(f"検証データ: {stats['split_statistics']['val_size']} サンプル "
                   f"({stats['split_statistics']['val_ratio']:.1%})")
        logger.info(f"テストデータ: {stats['split_statistics']['test_size']} サンプル "
                   f"({stats['split_statistics']['test_ratio']:.1%})")
        logger.info("")
        logger.info(f"訓練ゲーム数: {stats['split_statistics']['train_games']}")
        logger.info(f"検証ゲーム数: {stats['split_statistics']['val_games']}")
        logger.info(f"テストゲーム数: {stats['split_statistics']['test_games']}")
        logger.info("="*80)
        
        # 出力先を表示
        logger.info("\n✅ データセット構築完了!")
        logger.info(f"出力ディレクトリ: {args.output_dir}")
        logger.info(f"ログファイル: {log_file}")
        logger.info("\n次のステップ:")
        logger.info("  1. データセットの確認:")
        logger.info(f"     python -c \"from src.preprocessing import ComprehensiveDatasetBuilder; "
                   f"builder = ComprehensiveDatasetBuilder('', '{args.output_dir}'); "
                   f"builder.print_statistics()\"")
        logger.info("  2. 学習の開始:")
        logger.info("     python src/training/train_supervised.py")
        logger.info("")
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  ユーザーによって中断されました")
        return 130
    
    except Exception as e:
        logger.error(f"❌ エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

