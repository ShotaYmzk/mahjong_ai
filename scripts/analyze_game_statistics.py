#!/usr/bin/env python3
"""
麻雀ゲームログの統計分析スクリプト

天鳳のXMLゲームログから以下の指標を計算します：
- 平均順位、和了率、放銃率、和了率-放銃率
- 副露率、副露成功率、リーチ率、ダマ和了率
- 平均打点、平均放銃点、流局聴牌率
"""

import os
import re
import glob
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


class MahjongStatistics:
    """麻雀の統計情報を計算するクラス"""
    
    def __init__(self):
        # プレイヤーごとの統計
        self.player_stats = defaultdict(lambda: {
            'total_games': 0,
            'total_rounds': 0,
            'rankings': [],  # 各ゲームの順位
            'wins': 0,  # 和了回数
            'losses': 0,  # 放銃回数
            'win_points': [],  # 和了時の点数
            'loss_points': [],  # 放銃時の点数
            'riichi_count': 0,  # リーチ回数
            'riichi_wins': 0,  # リーチからの和了回数
            'dama_wins': 0,  # ダマテン和了回数
            'call_rounds': 0,  # 副露した局数
            'call_wins': 0,  # 副露からの和了回数
            'ryuukyoku_tenpai': 0,  # 流局時の聴牌回数
            'ryuukyoku_count': 0,  # 流局に遭遇した回数
        })
        
        # 全体統計
        self.total_games = 0
    
    def parse_xml_file(self, filepath: str):
        """XMLファイルを解析して統計情報を更新"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # プレイヤー名を取得
            player_names = self._parse_player_names(content)
            if not player_names:
                return
            
            # ラウンドごとの統計を収集
            self._analyze_rounds(content, player_names)
            
            # 最終スコアから順位を計算
            self._calculate_rankings(content, player_names)
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    def _parse_player_names(self, content: str) -> List[str]:
        """プレイヤー名を解析"""
        un_match = re.search(r'<UN n0="([^"]*)" n1="([^"]*)" n2="([^"]*)" n3="([^"]*)"', content)
        if un_match:
            from urllib.parse import unquote
            return [unquote(un_match.group(i + 1)) for i in range(4)]
        return []
    
    def _analyze_rounds(self, content: str, player_names: List[str]):
        """ラウンドごとの統計を分析"""
        # ラウンドごとに分割
        round_splits = re.split(r'<INIT ', content)
        
        for round_idx, round_content in enumerate(round_splits[1:]):
            round_content = '<INIT ' + round_content
            
            # 各プレイヤーの総ラウンド数をカウント
            for name in player_names:
                self.player_stats[name]['total_rounds'] += 1
            
            # このラウンドでのリーチ宣言を記録
            riichi_players = set()
            for match in re.finditer(r'<REACH who="(\d)" step="1"', round_content):
                player_id = int(match.group(1))
                player_name = player_names[player_id]
                self.player_stats[player_name]['riichi_count'] += 1
                riichi_players.add(player_id)
            
            # このラウンドでの副露を記録
            call_players = set()
            for match in re.finditer(r'<N who="(\d)"', round_content):
                player_id = int(match.group(1))
                call_players.add(player_id)
            
            # 副露した局数をカウント
            for player_id in call_players:
                player_name = player_names[player_id]
                self.player_stats[player_name]['call_rounds'] += 1
            
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
                
                winner_name = player_names[winner]
                
                # 和了
                self.player_stats[winner_name]['wins'] += 1
                
                # 点数を記録（2番目の値が実際の点数）
                # ten形式: "符,点数,役数" (例: "30,1000,0" or "30,18000,2")
                if len(ten_parts) >= 2:
                    points = int(ten_parts[1])
                    self.player_stats[winner_name]['win_points'].append(points)
                
                # リーチからの和了かダマテン和了か
                if winner in riichi_players:
                    self.player_stats[winner_name]['riichi_wins'] += 1
                else:
                    # ダマテン和了（リーチなしで和了）
                    self.player_stats[winner_name]['dama_wins'] += 1
                
                # 副露からの和了か
                if winner in call_players:
                    self.player_stats[winner_name]['call_wins'] += 1
                
                # ロン和了の場合、放銃者を記録
                if winner != from_who:
                    loser_name = player_names[from_who]
                    self.player_stats[loser_name]['losses'] += 1
                    if len(ten_parts) >= 2:
                        points = int(ten_parts[1])
                        self.player_stats[loser_name]['loss_points'].append(points)
            
            # 流局情報を解析
            if not agari_found:
                ryuukyoku_match = re.search(r'<RYUUKYOKU[^>]*', round_content)
                if ryuukyoku_match:
                    ryuukyoku_tag = ryuukyoku_match.group(0)
                    # 各プレイヤーの聴牌状況を確認
                    for i in range(4):
                        player_name = player_names[i]
                        self.player_stats[player_name]['ryuukyoku_count'] += 1
                        # haiX属性があれば聴牌
                        if f'hai{i}="' in ryuukyoku_tag:
                            self.player_stats[player_name]['ryuukyoku_tenpai'] += 1
    
    def _calculate_rankings(self, content: str, player_names: List[str]):
        """最終スコアから順位を計算"""
        # owari属性から最終スコアを取得
        owari_match = re.search(r'owari="([^"]+)"', content)
        if owari_match:
            scores_str = owari_match.group(1).split(',')
            # owari format: score0, delta0, score1, delta1, ...
            scores = [(int(scores_str[i]), i // 2) for i in range(0, 8, 2)]
            # スコア順でソート（降順）
            scores.sort(reverse=True)
            
            # 順位を割り当て（1位から4位）
            for rank, (score, player_id) in enumerate(scores, 1):
                player_name = player_names[player_id]
                self.player_stats[player_name]['rankings'].append(rank)
                if rank == 1:  # ゲームに参加したとカウント
                    pass
            
            # 全プレイヤーのゲーム数を増加
            for name in player_names:
                self.player_stats[name]['total_games'] += 1
            
            self.total_games += 1
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """統計指標を計算"""
        results = {}
        
        for player_name, stats in self.player_stats.items():
            if stats['total_games'] == 0:
                continue
            
            player_results = {}
            
            # 平均順位
            if stats['rankings']:
                player_results['平均順位'] = sum(stats['rankings']) / len(stats['rankings'])
            else:
                player_results['平均順位'] = 0.0
            
            # 和了率（％）
            if stats['total_rounds'] > 0:
                player_results['和了率'] = (stats['wins'] / stats['total_rounds']) * 100
            else:
                player_results['和了率'] = 0.0
            
            # 放銃率（％）
            if stats['total_rounds'] > 0:
                player_results['放銃率'] = (stats['losses'] / stats['total_rounds']) * 100
            else:
                player_results['放銃率'] = 0.0
            
            # 和了率 - 放銃率
            player_results['和了率-放銃率'] = player_results['和了率'] - player_results['放銃率']
            
            # 副露率（％）- 副露した局 / 総局数
            if stats['total_rounds'] > 0:
                player_results['副露率'] = (stats['call_rounds'] / stats['total_rounds']) * 100
            else:
                player_results['副露率'] = 0.0
            
            # 副露成功率（％）- 副露から和了した回数 / 副露した局数
            if stats['call_rounds'] > 0:
                player_results['副露成功率'] = (stats['call_wins'] / stats['call_rounds']) * 100
            else:
                player_results['副露成功率'] = 0.0
            
            # リーチ率（％）
            if stats['total_rounds'] > 0:
                player_results['リーチ率'] = (stats['riichi_count'] / stats['total_rounds']) * 100
            else:
                player_results['リーチ率'] = 0.0
            
            # ダマ和了率（％）
            if stats['wins'] > 0:
                player_results['ダマ和了率'] = (stats['dama_wins'] / stats['wins']) * 100
            else:
                player_results['ダマ和了率'] = 0.0
            
            # 平均打点
            if stats['win_points']:
                player_results['平均打点'] = sum(stats['win_points']) / len(stats['win_points'])
            else:
                player_results['平均打点'] = 0.0
            
            # 平均放銃点
            if stats['loss_points']:
                player_results['平均放銃点'] = sum(stats['loss_points']) / len(stats['loss_points'])
            else:
                player_results['平均放銃点'] = 0.0
            
            # 流局聴牌率（％）
            if stats['ryuukyoku_count'] > 0:
                player_results['流局聴牌率'] = (stats['ryuukyoku_tenpai'] / stats['ryuukyoku_count']) * 100
            else:
                player_results['流局聴牌率'] = 0.0
            
            # 基本情報
            player_results['総ゲーム数'] = stats['total_games']
            player_results['総局数'] = stats['total_rounds']
            player_results['和了回数'] = stats['wins']
            player_results['放銃回数'] = stats['losses']
            
            results[player_name] = player_results
        
        return results
    
    def export_to_csv(self, results: Dict[str, Dict[str, float]], output_path: str, min_games: int = 10):
        """統計結果をCSVファイルに出力"""
        # 最小ゲーム数でフィルタリング
        filtered_results = {name: stats for name, stats in results.items() if stats['総ゲーム数'] >= min_games}
        
        # 平均順位でソート
        sorted_results = sorted(filtered_results.items(), key=lambda x: (x[1]['平均順位'], -x[1]['総ゲーム数']))
        
        # CSVに書き込み
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow([
                'プレイヤー名', '平均順位', '和了率', '放銃率', '和了率-放銃率',
                '副露率', '副露成功率', 'リーチ率', 'ダマ和了率',
                '平均打点', '平均放銃点', '流局聴牌率',
                '総ゲーム数', '総局数', '和了回数', '放銃回数'
            ])
            
            # データ
            for player_name, stats in sorted_results:
                writer.writerow([
                    player_name,
                    f"{stats['平均順位']:.3f}",
                    f"{stats['和了率']:.2f}",
                    f"{stats['放銃率']:.2f}",
                    f"{stats['和了率-放銃率']:.2f}",
                    f"{stats['副露率']:.2f}",
                    f"{stats['副露成功率']:.2f}",
                    f"{stats['リーチ率']:.2f}",
                    f"{stats['ダマ和了率']:.2f}",
                    f"{stats['平均打点']:.0f}",
                    f"{stats['平均放銃点']:.0f}",
                    f"{stats['流局聴牌率']:.2f}",
                    stats['総ゲーム数'],
                    stats['総局数'],
                    stats['和了回数'],
                    stats['放銃回数']
                ])
        
        print(f"\nCSVファイルに出力しました: {output_path}")
    
    def print_statistics(self, results: Dict[str, Dict[str, float]], top_n: int = 20):
        """統計結果を表示"""
        print("=" * 100)
        print("麻雀ゲームログ統計分析")
        print("=" * 100)
        print(f"\n総ゲーム数: {self.total_games}")
        print(f"分析プレイヤー数: {len(results)}")
        
        # プレイヤーを平均順位でソート
        sorted_players = sorted(results.items(), key=lambda x: (x[1]['平均順位'], -x[1]['総ゲーム数']))
        
        # 最小ゲーム数でフィルタリング（統計的に意味のあるデータのみ）
        min_games = 10
        filtered_players = [(name, stats) for name, stats in sorted_players if stats['総ゲーム数'] >= min_games]
        
        print(f"\n※ {min_games}ゲーム以上プレイしたプレイヤーのみ表示")
        print(f"該当プレイヤー数: {len(filtered_players)}")
        
        if not filtered_players:
            print("\n統計データが不足しています。")
            return
        
        # トップNプレイヤーを表示
        print(f"\n{'='*100}")
        print(f"トップ{top_n}プレイヤー（平均順位順）")
        print(f"{'='*100}")
        
        header = f"{'順位':<4} {'プレイヤー名':<20} {'平均順位':<8} {'和了率':<8} {'放銃率':<8} {'差分':<8} {'総ゲーム':<8}"
        print(header)
        print("-" * 100)
        
        for rank, (player_name, stats) in enumerate(filtered_players[:top_n], 1):
            display_name = player_name[:18] + '..' if len(player_name) > 20 else player_name
            print(f"{rank:<4} {display_name:<20} "
                  f"{stats['平均順位']:<8.3f} "
                  f"{stats['和了率']:<8.2f} "
                  f"{stats['放銃率']:<8.2f} "
                  f"{stats['和了率-放銃率']:<8.2f} "
                  f"{stats['総ゲーム数']:<8}")
        
        # 詳細統計を表示（上位3プレイヤー）
        print(f"\n{'='*100}")
        print("詳細統計（トップ3プレイヤー）")
        print(f"{'='*100}")
        
        for rank, (player_name, stats) in enumerate(filtered_players[:3], 1):
            print(f"\n第{rank}位: {player_name}")
            print("-" * 100)
            print(f"  【総合指標】")
            print(f"    平均順位: {stats['平均順位']:.3f}")
            print(f"    和了率: {stats['和了率']:.2f}%")
            print(f"    放銃率: {stats['放銃率']:.2f}%")
            print(f"    和了率-放銃率: {stats['和了率-放銃率']:.2f}%")
            print(f"  【プレイスタイル】")
            print(f"    副露率: {stats['副露率']:.2f}%")
            print(f"    副露成功率: {stats['副露成功率']:.2f}%")
            print(f"    リーチ率: {stats['リーチ率']:.2f}%")
            print(f"    ダマ和了率: {stats['ダマ和了率']:.2f}%")
            print(f"  【その他】")
            print(f"    平均打点: {stats['平均打点']:.0f}点")
            print(f"    平均放銃点: {stats['平均放銃点']:.0f}点")
            print(f"    流局聴牌率: {stats['流局聴牌率']:.2f}%")
            print(f"  【基本情報】")
            print(f"    総ゲーム数: {stats['総ゲーム数']}")
            print(f"    総局数: {stats['総局数']}")
            print(f"    和了回数: {stats['和了回数']}")
            print(f"    放銃回数: {stats['放銃回数']}")
        
        # 全体統計
        print(f"\n{'='*100}")
        print("全体平均統計")
        print(f"{'='*100}")
        
        avg_stats = {}
        for key in ['平均順位', '和了率', '放銃率', '和了率-放銃率', '副露率', '副露成功率', 
                    'リーチ率', 'ダマ和了率', '平均打点', '平均放銃点', '流局聴牌率']:
            values = [stats[key] for _, stats in filtered_players if key in stats]
            if values:
                avg_stats[key] = sum(values) / len(values)
        
        print(f"  【総合指標】")
        print(f"    平均順位: {avg_stats.get('平均順位', 0):.3f}")
        print(f"    和了率: {avg_stats.get('和了率', 0):.2f}%")
        print(f"    放銃率: {avg_stats.get('放銃率', 0):.2f}%")
        print(f"    和了率-放銃率: {avg_stats.get('和了率-放銃率', 0):.2f}%")
        print(f"  【プレイスタイル】")
        print(f"    副露率: {avg_stats.get('副露率', 0):.2f}%")
        print(f"    副露成功率: {avg_stats.get('副露成功率', 0):.2f}%")
        print(f"    リーチ率: {avg_stats.get('リーチ率', 0):.2f}%")
        print(f"    ダマ和了率: {avg_stats.get('ダマ和了率', 0):.2f}%")
        print(f"  【その他】")
        print(f"    平均打点: {avg_stats.get('平均打点', 0):.0f}点")
        print(f"    平均放銃点: {avg_stats.get('平均放銃点', 0):.0f}点")
        print(f"    流局聴牌率: {avg_stats.get('流局聴牌率', 0):.2f}%")


def main():
    parser = argparse.ArgumentParser(description='麻雀ゲームログの統計分析')
    parser.add_argument('directory', help='XMLファイルが格納されているディレクトリ')
    parser.add_argument('--max-files', type=int, default=None, help='解析する最大ファイル数')
    parser.add_argument('--top-n', type=int, default=20, help='表示する上位プレイヤー数')
    parser.add_argument('--csv-output', type=str, default=None, help='CSV出力先ファイルパス')
    parser.add_argument('--min-games', type=int, default=10, help='最小ゲーム数（フィルタ用）')
    
    args = parser.parse_args()
    
    # XMLファイルを検索
    xml_files = glob.glob(os.path.join(args.directory, '*.xml'))
    
    if not xml_files:
        print(f"エラー: {args.directory} にXMLファイルが見つかりません")
        return
    
    if args.max_files:
        xml_files = xml_files[:args.max_files]
    
    print(f"解析対象ファイル数: {len(xml_files)}")
    
    # 統計分析
    analyzer = MahjongStatistics()
    
    for i, filepath in enumerate(xml_files, 1):
        if i % 100 == 0:
            print(f"処理中... {i}/{len(xml_files)}")
        analyzer.parse_xml_file(filepath)
    
    # 統計計算
    results = analyzer.calculate_statistics()
    
    # 結果表示
    analyzer.print_statistics(results, top_n=args.top_n)
    
    # CSV出力
    if args.csv_output:
        analyzer.export_to_csv(results, args.csv_output, min_games=args.min_games)


if __name__ == '__main__':
    main()

