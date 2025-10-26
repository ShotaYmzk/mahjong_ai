"""XML Downloader

天鳳のXMLログをダウンロードするモジュール
"""

import requests
import html as html_module
import re
import logging
from pathlib import Path
from time import sleep
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class XMLDownloader:
    """XMLログをダウンロードするクラス"""
    
    def __init__(self, save_dir: Path, 
                 max_workers: int = 5,
                 request_delay: float = 0.5):
        """初期化
        
        Args:
            save_dir: XMLファイルの保存先ディレクトリ
            max_workers: 並列ダウンロード数
            request_delay: リクエスト間の遅延（秒）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.request_delay = request_delay
        
        # リクエストヘッダー
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }
        
        # セッションを使用（Keep-Aliveで効率化）
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def download_single(self, url: str, timeout: int = 30) -> Optional[Path]:
        """単一のXMLをダウンロード
        
        Args:
            url: ダウンロードURL
            timeout: タイムアウト（秒）
            
        Returns:
            保存されたファイルのパス（失敗時はNone）
        """
        # ログIDを抽出
        log_id_match = re.search(r'\?(\d+gm-[\w-]+)', url)
        if not log_id_match:
            logger.error(f"ログIDを抽出できません: {url}")
            return None
        
        log_id = log_id_match.group(1)
        file_path = self.save_dir / f"{log_id}.xml"
        
        # 既に存在する場合はスキップ
        if file_path.exists() and file_path.stat().st_size > 100:
            logger.debug(f"スキップ（既存）: {log_id}")
            return file_path
        
        try:
            # リクエストを送信
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            # 文字コードを推定
            response.encoding = response.apparent_encoding
            html_text = response.text
            
            # mjloggmタグを抽出
            mjlog_match = re.search(
                r'(<mjloggm[^>]*>.*?</mjloggm>)',
                html_text,
                re.DOTALL | re.IGNORECASE
            )
            
            if not mjlog_match:
                logger.error(f"mjloggmタグが見つかりません: {url}")
                return None
            
            # HTMLエンティティをデコード
            raw_log = mjlog_match.group(1).strip()
            decoded_log = html_module.unescape(raw_log)
            
            # ファイルに保存
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(decoded_log)
            
            logger.info(f"✅ ダウンロード成功: {log_id}")
            
            # サーバー負荷軽減のため待機
            sleep(self.request_delay)
            
            return file_path
        
        except requests.exceptions.RequestException as e:
            logger.error(f"リクエストエラー {log_id}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"予期せぬエラー {log_id}: {e}")
            return None
    
    def download_batch(self, urls: List[str], 
                      show_progress: bool = True) -> List[Path]:
        """複数のXMLを並列ダウンロード
        
        Args:
            urls: ダウンロードURLのリスト
            show_progress: 進捗表示をするか
            
        Returns:
            保存されたファイルのパスリスト
        """
        downloaded_files = []
        total = len(urls)
        
        logger.info(f"バッチダウンロード開始: {total} 件")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 全てのダウンロードタスクを投入
            future_to_url = {
                executor.submit(self.download_single, url): url 
                for url in urls
            }
            
            # 完了したタスクから処理
            for i, future in enumerate(as_completed(future_to_url), 1):
                url = future_to_url[future]
                
                try:
                    result = future.result()
                    if result:
                        downloaded_files.append(result)
                    
                    if show_progress and i % 10 == 0:
                        logger.info(f"進捗: {i}/{total} ({i/total*100:.1f}%)")
                
                except Exception as e:
                    logger.error(f"ダウンロード失敗 {url}: {e}")
        
        logger.info(f"バッチダウンロード完了: {len(downloaded_files)}/{total} 件成功")
        return downloaded_files
    
    def download_from_log_ids(self, log_ids: List[str]) -> List[Path]:
        """ログIDのリストからダウンロード
        
        Args:
            log_ids: ログIDのリスト
            
        Returns:
            保存されたファイルのパスリスト
        """
        urls = [f"http://tenhou.net/0/log/?{log_id}" for log_id in log_ids]
        return self.download_batch(urls)
    
    def verify_downloaded_files(self) -> tuple[int, int]:
        """ダウンロードしたファイルを検証
        
        Returns:
            (有効なファイル数, 無効なファイル数)
        """
        valid = 0
        invalid = 0
        
        for xml_file in self.save_dir.glob("*.xml"):
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 基本的なチェック
                if '<mjloggm' in content and '</mjloggm>' in content:
                    valid += 1
                else:
                    invalid += 1
                    logger.warning(f"無効なファイル: {xml_file.name}")
            
            except Exception as e:
                invalid += 1
                logger.error(f"ファイル検証エラー {xml_file.name}: {e}")
        
        logger.info(f"検証結果: 有効={valid}, 無効={invalid}")
        return valid, invalid
    
    def cleanup_invalid_files(self) -> int:
        """無効なファイルを削除
        
        Returns:
            削除されたファイル数
        """
        removed = 0
        
        for xml_file in self.save_dir.glob("*.xml"):
            try:
                # 空ファイルや小さすぎるファイルを削除
                if xml_file.stat().st_size < 100:
                    xml_file.unlink()
                    removed += 1
                    logger.info(f"削除（小さすぎる）: {xml_file.name}")
                    continue
                
                # 内容をチェック
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if '<mjloggm' not in content or '</mjloggm>' not in content:
                    xml_file.unlink()
                    removed += 1
                    logger.info(f"削除（無効な内容）: {xml_file.name}")
            
            except Exception as e:
                logger.error(f"削除処理エラー {xml_file.name}: {e}")
        
        logger.info(f"クリーンアップ完了: {removed} 件削除")
        return removed
    
    def __del__(self):
        """デストラクタ: セッションをクローズ"""
        if hasattr(self, 'session'):
            self.session.close()


def main():
    """テスト用のメイン関数"""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # テスト用のダウンローダー
    downloader = XMLDownloader(
        save_dir=Path("/home/ubuntu/Documents/mahjong_ai/data/xml_logs_test"),
        max_workers=3,
        request_delay=1.0
    )
    
    # テスト用のURL（実際のログIDに置き換えてください）
    test_urls = [
        "http://tenhou.net/0/log/?2024010100gm-00a9-0000-12345678",
        # 追加のテストURL...
    ]
    
    logger.info("テストダウンロード開始")
    downloaded = downloader.download_batch(test_urls, show_progress=True)
    logger.info(f"ダウンロード完了: {len(downloaded)} 件")
    
    # 検証
    valid, invalid = downloader.verify_downloaded_files()
    logger.info(f"検証: 有効={valid}, 無効={invalid}")
    
    # クリーンアップ
    if invalid > 0:
        removed = downloader.cleanup_invalid_files()
        logger.info(f"クリーンアップ: {removed} 件削除")


if __name__ == "__main__":
    main()

