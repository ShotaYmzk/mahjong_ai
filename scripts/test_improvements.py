#!/usr/bin/env python3
"""テストスクリプト: 改善された機能のデモンストレーション

This script demonstrates the improvements made to the mahjong_ai project,
particularly the accurate parsing of naki (calls) events.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import AdvancedGameState
from preprocessing.advanced_game_state import decode_naki, tile_id_to_string
import re


def test_naki_decoding():
    """鳴きデコードのテスト"""
    print("=" * 80)
    print("鳴きデコードテスト")
    print("=" * 80)
    print()
    
    # テストケース: ver_3.0.0 で検証済みの鳴きコード
    test_cases = [
        (35913, "ポン", "中をポン"),
        (41236, "チー", "順子をチー"),
        (10524, "ポン", "数牌をポン"),
    ]
    
    for meld_code, expected_type, description in test_cases:
        print(f"Test: {description} (m={meld_code})")
        naki_info = decode_naki(meld_code)
        
        print(f"  Type: {naki_info['type']} (expected: {expected_type})")
        print(f"  Tiles: {[tile_id_to_string(t) for t in naki_info['tiles']]}")
        print(f"  From who (relative): {naki_info['from_who_relative']}")
        
        if naki_info['type'] == expected_type:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
        print()


def test_game_state_tracking():
    """ゲーム状態追跡のテスト"""
    print("=" * 80)
    print("ゲーム状態追跡テスト")
    print("=" * 80)
    print()
    
    # ゲーム状態の初期化
    game_state = AdvancedGameState()
    
    # 簡易的なラウンドデータを作成
    round_data = {
        "round_index": 1,
        "init": {
            "seed": "0,0,0,0,0,124",  # 東1局、ドラ表示牌=124 (7m)
            "ten": "250,250,250,250",
            "oya": "0",
            "hai0": "4,8,12,36,40,44,48,72,76,80,84,112,128",  # Player 0の配牌
            "hai1": "0,20,24,32,52,64,76,88,92,100,108,128,132",
            "hai2": "16,28,56,60,72,88,92,96,112,116,120,124",
            "hai3": "12,16,40,44,52,56,68,72,80,84,104,108,116"
        },
        "events": []
    }
    
    game_state.init_round(round_data)
    
    print(f"Round: {game_state.round_num_wind}")
    print(f"Dealer: Player {game_state.dealer}")
    print(f"Dora indicators: {[tile_id_to_string(t) for t in game_state.dora_indicators]}")
    print(f"Initial scores: {game_state.initial_scores}")
    print()
    
    # Player 0の手牌を表示
    hand_0 = game_state.player_hands[0]
    print(f"Player 0 hand: {[tile_id_to_string(t) for t in sorted(hand_0)]}")
    print(f"  Count: {len(hand_0)} tiles")
    
    # ツモのテスト
    print()
    print("Processing tsumo for Player 0 (tile ID 20)...")
    game_state.process_tsumo(0, 20)
    print(f"  Hand after tsumo: {len(game_state.player_hands[0])} tiles")
    print(f"  Wall remaining: {game_state.wall_tile_count} tiles")
    
    # 打牌のテスト
    print()
    print("Processing discard for Player 0 (tile ID 20)...")
    game_state.process_discard(0, 20, tsumogiri=True)
    print(f"  Hand after discard: {len(game_state.player_hands[0])} tiles")
    print(f"  Discards: {[tile_id_to_string(t) for t, _ in game_state.player_discards[0]]}")
    
    print()
    print("✅ ゲーム状態追跡テスト完了")


def test_xml_parsing_with_real_file():
    """実際のXMLファイルで鳴き解析をテスト"""
    print()
    print("=" * 80)
    print("実際のXMLファイルで鳴き解析テスト")
    print("=" * 80)
    print()
    
    # XMLファイルのパスを探す
    xml_path = Path(__file__).parent.parent / "2009080100gm-00e1-0000-63d644dd.xml"
    
    if not xml_path.exists():
        print(f"⚠️  XMLファイルが見つかりません: {xml_path}")
        return
    
    print(f"XMLファイル: {xml_path.name}")
    
    # XMLを読み込んで鳴きイベントを探す
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 鳴きイベントを抽出
    naki_matches = list(re.finditer(r'<N who="(\d)" m="(\d+)"', content))
    
    if naki_matches:
        print(f"Found {len(naki_matches)} naki events")
        print()
        
        # 最初の3つの鳴きイベントを表示
        for i, match in enumerate(naki_matches[:3]):
            player_id = int(match.group(1))
            meld_code = int(match.group(2))
            
            naki_info = decode_naki(meld_code)
            
            print(f"Naki {i+1}:")
            print(f"  Player: {player_id}")
            print(f"  Meld code: {meld_code}")
            print(f"  Type: {naki_info['type']}")
            print(f"  Tiles: {[tile_id_to_string(t) for t in naki_info['tiles']]}")
            print(f"  From who (relative): {naki_info['from_who_relative']}")
            print()
    else:
        print("鳴きイベントが見つかりませんでした")
    
    print("✅ 実XMLファイルテスト完了")


def main():
    """メイン関数"""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "麻雀AI改善機能テストスイート" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # テスト1: 鳴きデコード
        test_naki_decoding()
        
        # テスト2: ゲーム状態追跡
        test_game_state_tracking()
        
        # テスト3: 実際のXMLファイル
        test_xml_parsing_with_real_file()
        
        print()
        print("=" * 80)
        print("全テスト完了！")
        print("=" * 80)
        print()
        print("詳細な改善内容については IMPROVEMENTS.md を参照してください。")
        print()
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

