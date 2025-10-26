#!/usr/bin/env python3
"""
麻雀ゲームログの全体統計分析スクリプト

鳳凰卓全体の統計を計算します：
- 対局数、局数、平均局数
- 和了数、和了率、流局数、流局率
- 放銃数、放銃率、立直数、立直率
- 副露数、副露率、総打点、平均打点
- 親の連チャン率、親の平均打点、子の平均打点
- 席順別の平均順位
"""

import os
import re
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse
import json


class OverallMahjongStatistics:
    """麻雀の全体統計情報を計算するクラス"""
    
    def __init__(self):
        # 全体統計
        self.total_games = 0
        self.total_rounds = 0
        self.total_agari = 0
        self.total_ryuukyoku = 0
        self.total_houjuu = 0  # 放銃数（ロン和了数）
        self.total_riichi = 0
        self.total_calls = 0  # 副露数（鳴いた回数）
        self.total_points = 0  # 総打点
        
        # 親関連の統計
        self.oya_consecutive_wins = 0  # 親の連荘回数
        self.oya_rounds = 0  # 親の総局数
        self.oya_agari = 0  # 親の和了回数
        self.oya_points = []  # 親の和了時の打点リスト
        self.ko_points = []  # 子の和了時の打点リスト
        
        # 席順別の統計（初期席順による平均順位）
        self.seat_rankings = defaultdict(list)  # seat_id -> [rankings]
        
        # デバッグ用
        self.error_count = 0
    
    def parse_xml_file(self, filepath: str):
        """XMLファイルを解析して統計情報を更新"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # ゲームカウント
            self.total_games += 1
            
            # 席順別の順位を記録
            self._analyze_seat_rankings(content)
            
            # ラウンドごとの統計を収集
            self._analyze_rounds(content)
            
        except Exception as e:
            self.error_count += 1
            if self.error_count <= 10:  # 最初の10個のエラーのみ表示
                print(f"Error parsing {os.path.basename(filepath)}: {e}")
    
    def _analyze_seat_rankings(self, content: str):
        """席順別の順位を分析"""
        # owari属性から最終スコアを取得
        owari_match = re.search(r'owari="([^"]+)"', content)
        if owari_match:
            scores_str = owari_match.group(1).split(',')
            # owari format: score0, delta0, score1, delta1, ...
            scores = [(int(scores_str[i]), i // 2) for i in range(0, 8, 2)]
            # スコア順でソート（降順）
            scores.sort(reverse=True)
            
            # 順位を割り当て（1位から4位）
            for rank, (score, seat_id) in enumerate(scores, 1):
                self.seat_rankings[seat_id].append(rank)
    
    def _analyze_rounds(self, content: str):
        """ラウンドごとの統計を分析"""
        # ラウンドごとに分割
        round_splits = re.split(r'<INIT ', content)
        
        for round_idx, round_content in enumerate(round_splits[1:]):
            round_content = '<INIT ' + round_content
            
            # ラウンドカウント
            self.total_rounds += 1
            
            # INIT情報を取得（親の情報など）
            init_match = re.search(
                r'<INIT seed="([^"]+)" ten="([^"]+)" oya="(\d)"',
                round_content
            )
            
            if not init_match:
                continue
            
            oya = int(init_match.group(3))  # 親のプレイヤーID
            self.oya_rounds += 1
            
            # リーチ数をカウント
            riichi_count = len(re.findall(r'<REACH who="\d" step="1"', round_content))
            self.total_riichi += riichi_count
            
            # 副露数をカウント（N タグの数）
            call_count = len(re.findall(r'<N who="\d"', round_content))
            self.total_calls += call_count
            
            # 和了情報を解析
            agari_found = False
            for match in re.finditer(r'<AGARI\s+([^>]+)>', round_content):
                agari_found = True
                agari_attrs = match.group(1)
                
                # 各属性を抽出
                who_match = re.search(r'who="(\d)"', agari_attrs)
                from_who_match = re.search(r'fromWho="(\d)"', agari_attrs)
                ten_match = re.search(r'ten="([^"]+)"', agari_attrs)
                
                if not (who_match and from_who_match and ten_match):
                    continue
                
                winner = int(who_match.group(1))
                from_who = int(from_who_match.group(1))
                ten_parts = ten_match.group(1).split(',')
                
                # 和了数をカウント
                self.total_agari += 1
                
                # 点数を記録（2番目の値が実際の点数）
                if len(ten_parts) >= 2:
                    points = int(ten_parts[1])
                    self.total_points += points
                    
                    # 親か子かで分類
                    if winner == oya:
                        self.oya_agari += 1
                        self.oya_points.append(points)
                    else:
                        self.ko_points.append(points)
                
                # ロン和了の場合、放銃数をカウント
                if winner != from_who:
                    self.total_houjuu += 1
                
                # 親の連荘判定（親が和了した場合）
                if winner == oya:
                    self.oya_consecutive_wins += 1
            
            # 流局情報を解析
            if not agari_found:
                ryuukyoku_match = re.search(r'<RYUUKYOKU[^>]*', round_content)
                if ryuukyoku_match:
                    self.total_ryuukyoku += 1
    
    def calculate_statistics(self) -> Dict[str, any]:
        """統計指標を計算"""
        results = {}
        
        # 基礎情報
        results['対局数'] = self.total_games
        results['局数'] = self.total_rounds
        results['平均局数'] = self.total_rounds / self.total_games if self.total_games > 0 else 0
        
        # プレイヤー×局数（4人麻雀）
        player_rounds = self.total_rounds * 4
        
        # 和了関連（各プレイヤーが和了する確率）
        results['和了数'] = self.total_agari
        results['和了率'] = self.total_agari / player_rounds if player_rounds > 0 else 0
        
        # 流局関連
        results['流局数'] = self.total_ryuukyoku
        results['流局率'] = self.total_ryuukyoku / self.total_rounds if self.total_rounds > 0 else 0
        
        # 放銃関連（各プレイヤーが放銃する確率）
        results['放銃数'] = self.total_houjuu
        results['放銃率'] = self.total_houjuu / player_rounds if player_rounds > 0 else 0
        
        # 立直関連（各プレイヤーが立直する確率）
        results['立直数'] = self.total_riichi
        results['立直率'] = self.total_riichi / player_rounds if player_rounds > 0 else 0
        
        # 副露関連（各プレイヤーが副露する確率）
        results['副露数'] = self.total_calls
        results['副露率'] = self.total_calls / player_rounds if player_rounds > 0 else 0
        
        # 打点関連
        results['総打点'] = self.total_points
        results['平均打点'] = self.total_points / self.total_agari if self.total_agari > 0 else 0
        
        # 親関連
        results['親の連荘率'] = self.oya_consecutive_wins / self.oya_rounds if self.oya_rounds > 0 else 0
        results['親の和了数'] = self.oya_agari
        results['親の平均打点'] = sum(self.oya_points) / len(self.oya_points) if self.oya_points else 0
        results['子の平均打点'] = sum(self.ko_points) / len(self.ko_points) if self.ko_points else 0
        
        # 席順別の平均順位
        results['席順別平均順位'] = {}
        seat_names = ['東家', '南家', '西家', '北家']
        for seat_id in range(4):
            if self.seat_rankings[seat_id]:
                avg_rank = sum(self.seat_rankings[seat_id]) / len(self.seat_rankings[seat_id])
                results['席順別平均順位'][seat_names[seat_id]] = avg_rank
        
        return results
    
    def print_statistics(self, results: Dict[str, any]):
        """統計結果を表示"""
        print("=" * 100)
        print("麻雀全体統計分析（鳳凰卓）")
        print("=" * 100)
        
        print("\n【基礎情報】")
        print(f"  対局数: {results['対局数']:,}")
        print(f"  局数: {results['局数']:,}")
        print(f"  平均局数: {results['平均局数']:.2f}")
        
        print("\n【和了】")
        print(f"  和了数: {results['和了数']:,}")
        print(f"  和了率: {results['和了率']:.3f}")
        
        print("\n【流局】")
        print(f"  流局数: {results['流局数']:,}")
        print(f"  流局率: {results['流局率']:.3f}")
        
        print("\n【放銃】")
        print(f"  放銃数: {results['放銃数']:,}")
        print(f"  放銃率: {results['放銃率']:.3f}")
        
        print("\n【立直】")
        print(f"  立直数: {results['立直数']:,}")
        print(f"  立直率: {results['立直率']:.3f}")
        
        print("\n【副露】")
        print(f"  副露数: {results['副露数']:,}")
        print(f"  副露率: {results['副露率']:.3f}")
        
        print("\n【打点】")
        print(f"  総打点: {results['総打点']:,}")
        print(f"  平均打点: {results['平均打点']:.0f}")
        
        print("\n【親関連】")
        print(f"  親の連荘率: {results['親の連荘率']:.3f}")
        print(f"  親の和了数: {results['親の和了数']:,}")
        print(f"  親の平均打点: {results['親の平均打点']:.0f}")
        print(f"  子の平均打点: {results['子の平均打点']:.0f}")
        
        print("\n【席順別平均順位】")
        for seat_name, avg_rank in results['席順別平均順位'].items():
            print(f"  {seat_name}: {avg_rank:.3f}")
        
        if self.error_count > 0:
            print(f"\n※ 解析エラー: {self.error_count}ファイル")
    
    def export_to_json(self, results: Dict[str, any], output_path: str):
        """統計結果をJSON形式で出力"""
        # JSON用にデータを整形
        export_data = {
            '基礎情報': {
                '対局数': results['対局数'],
                '局数': results['局数'],
                '平均局数': round(results['平均局数'], 2)
            },
            '和了': {
                '和了数': results['和了数'],
                '和了率': round(results['和了率'], 3)
            },
            '流局': {
                '流局数': results['流局数'],
                '流局率': round(results['流局率'], 3)
            },
            '放銃': {
                '放銃数': results['放銃数'],
                '放銃率': round(results['放銃率'], 3)
            },
            '立直': {
                '立直数': results['立直数'],
                '立直率': round(results['立直率'], 3)
            },
            '副露': {
                '副露数': results['副露数'],
                '副露率': round(results['副露率'], 3)
            },
            '打点': {
                '総打点': results['総打点'],
                '平均打点': round(results['平均打点'], 0)
            },
            '親関連': {
                '親の連荘率': round(results['親の連荘率'], 3),
                '親の和了数': results['親の和了数'],
                '親の平均打点': round(results['親の平均打点'], 0),
                '子の平均打点': round(results['子の平均打点'], 0)
            },
            '席順別平均順位': {
                seat: round(rank, 3)
                for seat, rank in results['席順別平均順位'].items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nJSON形式で出力しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='麻雀ゲームログの全体統計分析')
    parser.add_argument('directory', help='XMLファイルが格納されているディレクトリ')
    parser.add_argument('--max-files', type=int, default=None, help='解析する最大ファイル数')
    parser.add_argument('--json-output', type=str, default=None, help='JSON出力先ファイルパス')
    parser.add_argument('--progress-interval', type=int, default=1000, help='進捗表示の間隔')
    
    args = parser.parse_args()
    
    # XMLファイルを検索
    xml_files = glob.glob(os.path.join(args.directory, '*.xml'))
    
    if not xml_files:
        print(f"エラー: {args.directory} にXMLファイルが見つかりません")
        return
    
    if args.max_files:
        xml_files = xml_files[:args.max_files]
    
    print(f"解析対象ファイル数: {len(xml_files):,}")
    print("解析を開始します...\n")
    
    # 統計分析
    analyzer = OverallMahjongStatistics()
    
    for i, filepath in enumerate(xml_files, 1):
        if i % args.progress_interval == 0:
            print(f"処理中... {i:,}/{len(xml_files):,} ({i/len(xml_files)*100:.1f}%)")
        analyzer.parse_xml_file(filepath)
    
    print(f"処理完了: {len(xml_files):,}/{len(xml_files):,} (100.0%)\n")
    
    # 統計計算
    results = analyzer.calculate_statistics()
    
    # 結果表示
    analyzer.print_statistics(results)
    
    # JSON出力
    if args.json_output:
        analyzer.export_to_json(results, args.json_output)


if __name__ == '__main__':
    main()

