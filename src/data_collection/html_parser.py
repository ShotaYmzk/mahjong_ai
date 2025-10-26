"""HTML Log Parser

天鳳のHTMLログからURLを抽出・変換するモジュール
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class HTMLLogParser:
    """天鳳のHTMLログを解析するクラス"""
    
    # ゲームタイプの定義
    GAME_TYPES = {
        "四鳳": "4-player phoenix",
        "三鳳": "3-player phoenix",
        "四般": "4-player general",
        "三般": "3-player general",
    }
    
    def __init__(self):
        """初期化"""
        self.url_pattern = re.compile(
            r'href=["\'](http://tenhou\.net/0/\?log=([^"\']+))["\']'
        )
    
    def parse_html_file(self, html_file: Path, 
                       game_type: Optional[str] = None) -> List[Dict[str, str]]:
        """HTMLファイルを解析してログ情報を抽出
        
        Args:
            html_file: HTMLファイルのパス
            game_type: フィルタするゲームタイプ（Noneの場合は全て）
            
        Returns:
            ログ情報のリスト
            [{
                "log_id": str,
                "original_url": str,
                "display_url": str,
                "game_type": str,
                "line_num": int
            }, ...]
        """
        if not html_file.exists():
            logger.error(f"HTMLファイルが存在しません: {html_file}")
            return []
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"ファイル読み込みエラー {html_file}: {e}")
            return []
        
        logs = []
        
        for line_num, line in enumerate(lines, 1):
            # ゲームタイプのフィルタリング
            if game_type and game_type not in line:
                continue
            
            # URLを抽出
            match = self.url_pattern.search(line)
            if not match:
                continue
            
            original_url = match.group(1)
            log_param = match.group(2)
            
            # ログIDを抽出
            log_id_match = re.search(r'(\d+gm-[\w-]+)', log_param)
            if not log_id_match:
                logger.warning(f"ログIDを抽出できません (行 {line_num}): {log_param}")
                continue
            
            log_id = log_id_match.group(1)
            
            # 表示用URLに変換
            display_url = f'http://tenhou.net/0/log/?{log_param}'
            
            # ゲームタイプを判定
            detected_game_type = "不明"
            for gt in self.GAME_TYPES:
                if gt in line:
                    detected_game_type = gt
                    break
            
            logs.append({
                "log_id": log_id,
                "original_url": original_url,
                "display_url": display_url,
                "game_type": detected_game_type,
                "line_num": line_num
            })
        
        logger.info(f"{html_file.name}: {len(logs)} 件のログを抽出")
        return logs
    
    def convert_url_format(self, input_file: Path, output_file: Path,
                          game_type: str = "四鳳") -> int:
        """HTMLファイルのURL形式を変換
        
        Args:
            input_file: 入力HTMLファイル
            output_file: 出力HTMLファイル
            game_type: フィルタするゲームタイプ
            
        Returns:
            変換された行数
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            return 0
        
        filtered_lines = []
        
        for line in lines:
            # パイプ区切りで分割
            parts = line.split('|')
            if len(parts) < 3:
                continue
            
            # 3つ目の項目でゲームタイプをチェック
            third_field = parts[2].strip()
            if not third_field.startswith(game_type):
                continue
            
            # URL形式を変換
            # http://tenhou.net/0/?log=XXX → http://tenhou.net/0/log/?XXX
            new_line = self.url_pattern.sub(
                r'href="http://tenhou.net/0/log/?\2"',
                line
            )
            filtered_lines.append(new_line)
        
        # 出力ファイルに書き込み
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)
            logger.info(f"変換完了: {len(filtered_lines)} 行を {output_file} に保存")
        except Exception as e:
            logger.error(f"ファイル書き込みエラー: {e}")
            return 0
        
        return len(filtered_lines)
    
    def extract_all_urls(self, html_file: Path) -> List[str]:
        """HTMLファイルから全てのURLを抽出（簡易版）
        
        Args:
            html_file: HTMLファイルのパス
            
        Returns:
            URLのリスト
        """
        logs = self.parse_html_file(html_file)
        return [log["display_url"] for log in logs]
    
    def filter_by_game_type(self, logs: List[Dict[str, str]], 
                           game_type: str) -> List[Dict[str, str]]:
        """ゲームタイプでフィルタリング
        
        Args:
            logs: ログ情報のリスト
            game_type: フィルタするゲームタイプ
            
        Returns:
            フィルタされたログ情報のリスト
        """
        return [log for log in logs if log["game_type"] == game_type]
    
    def get_statistics(self, html_file: Path) -> Dict[str, int]:
        """HTMLファイルの統計情報を取得
        
        Args:
            html_file: HTMLファイルのパス
            
        Returns:
            統計情報の辞書
        """
        logs = self.parse_html_file(html_file)
        
        stats = {
            "total": len(logs),
        }
        
        # ゲームタイプ別の集計
        for game_type in self.GAME_TYPES:
            stats[game_type] = len([
                log for log in logs if log["game_type"] == game_type
            ])
        
        return stats


def main():
    """テスト用のメイン関数"""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = HTMLLogParser()
    
    # テスト用のサンプルHTMLファイル（存在する場合）
    test_file = Path("/home/ubuntu/Documents/tenhou_dataset/2024/scc20240101.html")
    
    if test_file.exists():
        logger.info(f"テストファイル: {test_file}")
        
        # 統計情報を取得
        stats = parser.get_statistics(test_file)
        logger.info(f"統計情報: {stats}")
        
        # 四鳳のログを抽出
        logs = parser.parse_html_file(test_file, game_type="四鳳")
        logger.info(f"四鳳のログ数: {len(logs)}")
        
        if logs:
            logger.info(f"最初のログ: {logs[0]}")
    else:
        logger.warning(f"テストファイルが存在しません: {test_file}")
        logger.info("HTMLLogParserのテストをスキップします")


if __name__ == "__main__":
    main()

