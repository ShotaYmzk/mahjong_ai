"""Tenhou Data Fetcher

天鳳のデータを取得するための統合モジュール。
dataset_prepareの優れた実装を参考に、完全なデータ収集パイプラインを実装。
"""

import re
import os
import glob
import requests
import html
import logging
from pathlib import Path
from time import sleep
from typing import List, Optional, Dict
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class TenhouDataFetcher:
    """天鳳のデータを取得・管理するクラス"""
    
    def __init__(self, 
                 html_base_dir: str = "/home/ubuntu/Documents/tenhou_dataset",
                 xml_save_dir: str = "/home/ubuntu/Documents/mahjong_ai/data/xml_logs"):
        """初期化
        
        Args:
            html_base_dir: 展開済みHTMLファイルの親ディレクトリ
            xml_save_dir: XMLファイルの保存先ディレクトリ
        """
        self.html_base_dir = Path(html_base_dir)
        self.xml_save_dir = Path(xml_save_dir)
        self.xml_save_dir.mkdir(parents=True, exist_ok=True)
        
        # リクエストヘッダー（天鳳サーバーへの配慮）
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # ダウンロード統計
        self.stats = {
            "total_attempted": 0,
            "successful": 0,
            "skipped_existing": 0,
            "failed": 0
        }
    
    def download_xml_from_url(self, url: str, retry: int = 3) -> Optional[Path]:
        """指定されたURLからXMLをダウンロード
        
        Args:
            url: 天鳳のログURL
            retry: リトライ回数
            
        Returns:
            保存されたXMLファイルのパス（失敗時はNone）
        """
        # ログIDを抽出
        log_id_match = re.search(r'\?(\d+gm-[\w-]+)', url)
        if not log_id_match:
            logger.error(f"URLからログIDを抽出できません: {url}")
            return None
        
        log_id = log_id_match.group(1)
        file_path = self.xml_save_dir / f"{log_id}.xml"
        
        # 既に存在する場合はスキップ
        if file_path.exists() and file_path.stat().st_size > 0:
            logger.info(f"既に存在: {file_path.name}")
            self.stats["skipped_existing"] += 1
            return file_path
        
        # ダウンロード試行
        for attempt in range(retry):
            try:
                self.stats["total_attempted"] += 1
                logger.info(f"リクエスト開始 ({attempt + 1}/{retry}): {url}")
                
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                # 文字コードを自動判定
                response.encoding = response.apparent_encoding
                html_text = response.text
                
                # <mjloggm ...> から </mjloggm> までを抽出
                mjlog_match = re.search(
                    r'(<mjloggm[^>]*>.*?</mjloggm>)', 
                    html_text, 
                    re.DOTALL | re.IGNORECASE
                )
                
                if not mjlog_match:
                    logger.error(f"<mjloggm>タグが見つかりません: {url}")
                    logger.debug(f"HTML内容抜粋: {html_text[:500]}")
                    self.stats["failed"] += 1
                    return None
                
                # mjloggmブロックを抽出
                raw_log = mjlog_match.group(1).strip()
                logger.debug(f"抽出したログの長さ: {len(raw_log)} 文字")
                
                if not raw_log:
                    logger.warning(f"ログ内容が空です: {url}")
                    self.stats["failed"] += 1
                    return None
                
                # HTMLエンティティをデコード (&lt; &gt; など)
                decoded_log = html.unescape(raw_log)
                
                # UTF-8でファイルに書き込み
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(decoded_log)
                
                # 保存確認
                if file_path.exists() and file_path.stat().st_size > 0:
                    logger.info(f"✅ ダウンロード成功: {file_path.name} ({file_path.stat().st_size} bytes)")
                    self.stats["successful"] += 1
                    return file_path
                else:
                    logger.error(f"ファイルが生成されなかったか空です: {file_path}")
                    self.stats["failed"] += 1
                    return None
                
            except requests.exceptions.RequestException as e:
                logger.error(f"リクエスト失敗 ({attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.stats["failed"] += 1
                    return None
            
            except Exception as e:
                logger.error(f"予期せぬエラー: {e}", exc_info=True)
                self.stats["failed"] += 1
                return None
            
            finally:
                # サーバー負荷軽減のため待機
                sleep(0.5)
        
        return None
    
    def extract_urls_from_html(self, html_file: Path, 
                               game_type: str = "四鳳") -> List[str]:
        """HTMLファイルからゲームURLを抽出
        
        Args:
            html_file: HTMLファイルのパス
            game_type: 抽出するゲームタイプ（デフォルト: 四鳳）
            
        Returns:
            URLのリスト
        """
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"ファイル読み込みエラー {html_file}: {e}")
            return []
        
        urls = []
        pattern = re.compile(r'href=["\'](http://tenhou\.net/0/\?log=([^"\']+))["\']')
        
        for line in lines:
            # 指定されたゲームタイプを含む行のみ処理
            if game_type not in line:
                continue
            
            match = pattern.search(line)
            if match:
                log_param = match.group(2)
                # ログ表示用URLに変換
                url = f'http://tenhou.net/0/log/?{log_param}'
                urls.append(url)
        
        return urls
    
    def fetch_from_html_directory(self, 
                                   year_range: range = range(2024, 2022, -1),
                                   game_type: str = "四鳳",
                                   max_files: Optional[int] = None) -> List[Path]:
        """HTMLディレクトリからXMLファイルを一括取得
        
        Args:
            year_range: 処理する年の範囲
            game_type: 取得するゲームタイプ
            max_files: 最大処理ファイル数（テスト用）
            
        Returns:
            ダウンロードされたXMLファイルのパスリスト
        """
        logger.info("=" * 80)
        logger.info("天鳳データ収集開始")
        logger.info(f"HTMLベースディレクトリ: {self.html_base_dir}")
        logger.info(f"XML保存ディレクトリ: {self.xml_save_dir}")
        logger.info(f"処理対象年: {list(year_range)}")
        logger.info(f"ゲームタイプ: {game_type}")
        logger.info("=" * 80)
        
        # HTMLファイルを収集
        html_files = []
        for year in year_range:
            year_dir = self.html_base_dir / str(year)
            if year_dir.exists():
                found = list(year_dir.glob("scc*.html"))
                logger.info(f"年 {year}: {len(found)} 個のHTMLファイルを発見")
                html_files.extend(found)
            else:
                logger.warning(f"ディレクトリが見つかりません: {year_dir}")
        
        logger.info(f"発見したHTMLファイル総数: {len(html_files)}")
        
        if not html_files:
            logger.error("HTMLファイルが見つかりません")
            return []
        
        # 最大数の制限
        if max_files:
            html_files = html_files[:max_files]
            logger.info(f"処理を {max_files} ファイルに制限")
        
        xml_files = []
        
        # 各HTMLファイルを処理
        for i, html_file in enumerate(html_files):
            logger.info(f"\n--- HTML処理中 ({i+1}/{len(html_files)}): {html_file.name} ---")
            
            # URLを抽出
            urls = self.extract_urls_from_html(html_file, game_type=game_type)
            logger.info(f"抽出された{game_type}URL数: {len(urls)}")
            
            # 各URLをダウンロード
            for url in urls:
                xml_file = self.download_xml_from_url(url)
                if xml_file:
                    xml_files.append(xml_file)
        
        # 統計を表示
        logger.info("\n" + "=" * 80)
        logger.info("データ収集完了")
        logger.info("=" * 80)
        logger.info(f"総試行数: {self.stats['total_attempted']}")
        logger.info(f"成功: {self.stats['successful']}")
        logger.info(f"既存スキップ: {self.stats['skipped_existing']}")
        logger.info(f"失敗: {self.stats['failed']}")
        logger.info(f"最終的に取得されたXMLファイル数: {len(xml_files)}")
        logger.info("=" * 80)
        
        return xml_files
    
    def download_from_log_id(self, log_id: str) -> Optional[Path]:
        """ログIDから直接XMLをダウンロード
        
        Args:
            log_id: 天鳳のログID (例: "2024010100gm-00a9-0000-12345678")
            
        Returns:
            保存されたXMLファイルのパス（失敗時はNone）
        """
        url = f"http://tenhou.net/0/log/?{log_id}"
        return self.download_xml_from_url(url)
    
    def get_player_names_from_xml(self, xml_file: Path) -> List[str]:
        """XMLファイルからプレイヤー名を取得
        
        Args:
            xml_file: XMLファイルのパス
            
        Returns:
            プレイヤー名のリスト
        """
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # <UN>タグからプレイヤー名を抽出
            un_match = re.search(
                r'<UN n0="([^"]*)" n1="([^"]*)" n2="([^"]*)" n3="([^"]*)"',
                content
            )
            
            if un_match:
                return [unquote(un_match.group(i + 1)) for i in range(4)]
            else:
                return ["Player0", "Player1", "Player2", "Player3"]
        
        except Exception as e:
            logger.error(f"プレイヤー名の取得に失敗: {e}")
            return ["Player0", "Player1", "Player2", "Player3"]
    
    def validate_xml_file(self, xml_file: Path) -> bool:
        """XMLファイルの妥当性を検証
        
        Args:
            xml_file: XMLファイルのパス
            
        Returns:
            妥当性（True/False）
        """
        if not xml_file.exists():
            return False
        
        if xml_file.stat().st_size == 0:
            return False
        
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 基本的なタグの存在確認
            required_tags = ['<mjloggm', '<INIT', '</mjloggm>']
            return all(tag in content for tag in required_tags)
        
        except Exception as e:
            logger.error(f"ファイル検証エラー: {e}")
            return False
    
    def clean_invalid_files(self) -> int:
        """無効なXMLファイルを削除
        
        Returns:
            削除されたファイル数
        """
        removed = 0
        for xml_file in self.xml_save_dir.glob("*.xml"):
            if not self.validate_xml_file(xml_file):
                logger.warning(f"無効なファイルを削除: {xml_file.name}")
                xml_file.unlink()
                removed += 1
        
        logger.info(f"削除された無効なファイル数: {removed}")
        return removed


def main():
    """テスト用のメイン関数"""
    import sys
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('tenhou_fetcher.log', encoding='utf-8')
        ]
    )
    
    # フェッチャーの初期化
    fetcher = TenhouDataFetcher()
    
    # テスト: 特定のログIDをダウンロード
    test_log_id = "2024010100gm-00a9-0000-12345678"
    logger.info(f"テスト: ログID {test_log_id} をダウンロード")
    xml_file = fetcher.download_from_log_id(test_log_id)
    
    if xml_file:
        logger.info(f"✅ ダウンロード成功: {xml_file}")
        
        # プレイヤー名を取得
        player_names = fetcher.get_player_names_from_xml(xml_file)
        logger.info(f"プレイヤー: {', '.join(player_names)}")
        
        # 妥当性を検証
        is_valid = fetcher.validate_xml_file(xml_file)
        logger.info(f"妥当性: {'✅ 有効' if is_valid else '❌ 無効'}")
    else:
        logger.error("❌ ダウンロード失敗")
    
    # 無効なファイルをクリーンアップ
    fetcher.clean_invalid_files()


if __name__ == "__main__":
    main()

