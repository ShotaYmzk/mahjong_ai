#!/usr/bin/env python3
"""天鳳データ収集スクリプト

このスクリプトは天鳳のHTMLログからXMLデータを収集します。
dataset_prepareの優れた実装を参考にした完全なデータ収集パイプライン。

使用方法:
    # 基本的な使用
    python scripts/collect_tenhou_data.py
    
    # 年とゲームタイプを指定
    python scripts/collect_tenhou_data.py --years 2024 2023 --game-type 四鳳
    
    # 最大ファイル数を制限（テスト用）
    python scripts/collect_tenhou_data.py --max-files 10
"""

import sys
import argparse
import logging
from pathlib import Path

# srcをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_collection import TenhouDataFetcher


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
        description='天鳳のデータを収集します',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # 2024年のデータを収集
  python scripts/collect_tenhou_data.py --years 2024
  
  # 2023-2024年の四鳳のデータを収集
  python scripts/collect_tenhou_data.py --years 2024 2023 --game-type 四鳳
  
  # テスト実行（最初の10ファイルのみ）
  python scripts/collect_tenhou_data.py --max-files 10
        '''
    )
    
    parser.add_argument(
        '--html-dir',
        type=str,
        default='/home/ubuntu/Documents/tenhou_dataset',
        help='展開済みHTMLファイルのベースディレクトリ'
    )
    
    parser.add_argument(
        '--xml-dir',
        type=str,
        default='/home/ubuntu/Documents/mahjong_ai/data/xml_logs',
        help='XMLファイルの保存先ディレクトリ'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2024, 2023],
        help='処理する年（複数指定可能）'
    )
    
    parser.add_argument(
        '--game-type',
        type=str,
        default='四鳳',
        choices=['四鳳', '三鳳', '四般', '三般'],
        help='収集するゲームタイプ'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='処理する最大HTMLファイル数（テスト用）'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='無効なXMLファイルを削除する'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='tenhou_data_collection.log',
        help='ログファイルのパス'
    )
    
    args = parser.parse_args()
    
    # ログ設定
    log_file = Path(args.log_file)
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("天鳳データ収集スクリプト")
    logger.info("=" * 80)
    logger.info(f"HTMLディレクトリ: {args.html_dir}")
    logger.info(f"XMLディレクトリ: {args.xml_dir}")
    logger.info(f"対象年: {args.years}")
    logger.info(f"ゲームタイプ: {args.game_type}")
    if args.max_files:
        logger.info(f"最大ファイル数: {args.max_files}")
    logger.info("=" * 80)
    
    # HTMLディレクトリの存在確認
    html_dir = Path(args.html_dir)
    if not html_dir.exists():
        logger.error(f"HTMLディレクトリが存在しません: {html_dir}")
        logger.error("展開済みのHTMLファイルを配置してください")
        return 1
    
    # データフェッチャーの初期化
    fetcher = TenhouDataFetcher(
        html_base_dir=args.html_dir,
        xml_save_dir=args.xml_dir
    )
    
    # データ収集
    try:
        # 年の範囲を作成
        years = sorted(args.years, reverse=True)
        year_range = range(max(years), min(years) - 1, -1)
        
        logger.info(f"データ収集を開始します...")
        xml_files = fetcher.fetch_from_html_directory(
            year_range=year_range,
            game_type=args.game_type,
            max_files=args.max_files
        )
        
        logger.info(f"✅ データ収集完了: {len(xml_files)} ファイル")
        
        # クリーンアップ（オプション）
        if args.clean:
            logger.info("無効なファイルのクリーンアップを実行します...")
            removed = fetcher.clean_invalid_files()
            logger.info(f"✅ クリーンアップ完了: {removed} ファイル削除")
        
        # 最終統計
        logger.info("\n" + "=" * 80)
        logger.info("最終統計")
        logger.info("=" * 80)
        logger.info(f"総試行数: {fetcher.stats['total_attempted']}")
        logger.info(f"成功: {fetcher.stats['successful']}")
        logger.info(f"既存スキップ: {fetcher.stats['skipped_existing']}")
        logger.info(f"失敗: {fetcher.stats['failed']}")
        logger.info(f"最終ファイル数: {len(xml_files)}")
        logger.info("=" * 80)
        
        # サンプルファイルの情報を表示
        if xml_files:
            sample_file = xml_files[0]
            logger.info(f"\nサンプルファイル: {sample_file.name}")
            player_names = fetcher.get_player_names_from_xml(sample_file)
            logger.info(f"プレイヤー: {', '.join(player_names)}")
        
        logger.info(f"\nログファイル: {log_file}")
        logger.info(f"XMLファイル保存先: {args.xml_dir}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  ユーザーによって中断されました")
        return 130
    
    except Exception as e:
        logger.error(f"❌ エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

