#!/usr/bin/env python3
"""XMLファイルの読み取り確認スクリプト

誰が何をツモって、何を捨てて、何を鳴いたかを分かりやすく表示します。
"""

import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.parse_xml import TenhouXMLParser
import re


def tile_to_string(tile_id: int) -> str:
    """牌IDを人間が読める形式に変換する
    
    Args:
        tile_id: 牌ID (0-135)
        
    Returns:
        牌の文字列表現 (例: "1m", "東", "5p赤")
    """
    tile_type = tile_id // 4
    
    # 萬子 (0-8)
    if tile_type < 9:
        num = tile_type + 1
        # 赤5m は tile_id 16,17,18,19 のうち 16
        if num == 5 and tile_id == 16:
            return f"5m赤"
        return f"{num}m"
    
    # 筒子 (9-17)
    elif tile_type < 18:
        num = tile_type - 9 + 1
        # 赤5p は tile_id 52,53,54,55 のうち 52
        if num == 5 and tile_id == 52:
            return f"5p赤"
        return f"{num}p"
    
    # 索子 (18-26)
    elif tile_type < 27:
        num = tile_type - 18 + 1
        # 赤5s は tile_id 88,89,90,91 のうち 88
        if num == 5 and tile_id == 88:
            return f"5s赤"
        return f"{num}s"
    
    # 字牌 (27-33)
    else:
        honors = ["東", "南", "西", "北", "白", "発", "中"]
        honor_idx = tile_type - 27
        return honors[honor_idx]


def display_hand(hand: list[int]) -> str:
    """手牌を表示用の文字列に変換"""
    return " ".join(tile_to_string(t) for t in sorted(hand))


def decode_meld_proper(meld_code: int) -> dict:
    """天鳳の鳴きコードを正しくデコード
    
    Returns:
        dict with keys: type, tiles (tile indices), from_who_relative, raw_value
    """
    result = {
        "type": "不明",
        "tiles": [],
        "from_who_relative": -1,
        "raw_value": meld_code
    }
    
    try:
        from_who_relative = meld_code & 3
        result["from_who_relative"] = from_who_relative
        
        # チー (Chi)
        if meld_code & (1 << 2):
            result["type"] = "チー"
            t = meld_code >> 10
            r = t % 3
            t //= 3
            
            # Base index calculation
            if 0 <= t <= 6:
                base_index = t  # 1m-7m
            elif 7 <= t <= 13:
                base_index = (t - 7) + 9  # 1p-7p
            elif 14 <= t <= 20:
                base_index = (t - 14) + 18  # 1s-7s
            else:
                return result
            
            # Get the three tile IDs in the sequence
            offsets = [(meld_code >> 3) & 3, (meld_code >> 5) & 3, (meld_code >> 7) & 3]
            tiles = []
            for i in range(3):
                tile_kind = base_index + i
                tile_id = tile_kind * 4 + offsets[i]
                tiles.append(tile_id)
            
            result["tiles"] = sorted(tiles)
            result["called_position"] = r  # Which of the 3 tiles was called
        
        # ポン (Pon)
        elif meld_code & (1 << 3):
            result["type"] = "ポン"
            t = meld_code >> 9
            t //= 3
            
            if not (0 <= t <= 33):
                return result
            
            base_id = t * 4
            unused_offset = (meld_code >> 5) & 3
            
            tiles = []
            for i in range(4):
                if i != unused_offset:
                    tiles.append(base_id + i)
            
            result["tiles"] = sorted(tiles)
        
        # 加槓 (Kakan)
        elif meld_code & (1 << 4):
            result["type"] = "加槓"
            result["from_who_relative"] = -1
            t = meld_code >> 9
            t //= 3
            
            if not (0 <= t <= 33):
                return result
            
            base_id = t * 4
            result["tiles"] = sorted([base_id, base_id + 1, base_id + 2, base_id + 3])
        
        # 暗槓 or 大明槓 (Ankan or Daiminkan)
        else:
            tile_id_raw = meld_code >> 8
            tile_index = tile_id_raw // 4
            
            if not (0 <= tile_index <= 33):
                return result
            
            base_id = tile_index * 4
            result["tiles"] = sorted([base_id, base_id + 1, base_id + 2, base_id + 3])
            
            if from_who_relative != 3:
                result["type"] = "大明槓"
            else:
                result["type"] = "暗槓"
                result["from_who_relative"] = -1
        
        return result
    
    except Exception as e:
        print(f"[Error] decode_meld failed for m={meld_code}: {e}")
        return result


def display_meld(meld_info: dict, called_tile_id: int, from_player: int, naki_player: int, 
                 player_names: list, consumed_tiles: list = None) -> str:
    """鳴きを表示用の文字列に変換（詳細版）
    
    Args:
        meld_info: decode_meld_properの結果
        called_tile_id: 実際に鳴かれた牌のID
        from_player: 鳴き元のプレイヤーID (絶対位置)
        naki_player: 鳴いたプレイヤーID
        player_names: プレイヤー名のリスト
        consumed_tiles: 手牌から消費された牌のリスト
    """
    meld_type = meld_info["type"]
    all_tiles = meld_info["tiles"]
    
    # 全ての牌を文字列に変換
    all_tiles_str = " ".join(tile_to_string(t) for t in sorted(all_tiles))
    
    # 鳴かれた牌の表示
    if called_tile_id != -1:
        called_tile_str = tile_to_string(called_tile_id)
    else:
        called_tile_str = tile_to_string(all_tiles[0]) if all_tiles else "?"
    
    # from_whoの表示
    if from_player != -1 and from_player != naki_player:
        from_str = f"Player {from_player} ({player_names[from_player]})"
    else:
        from_str = "自分"
    
    # 詳細な構成を表示
    if meld_type in ["チー", "ポン", "大明槓"] and consumed_tiles:
        # 手牌から使った牌を表示
        consumed_str = " ".join(tile_to_string(t) for t in sorted(consumed_tiles))
        return (f"{meld_type} [{all_tiles_str}] "
                f"(鳴き: {called_tile_str} から{from_str}, "
                f"手牌: {consumed_str})")
    elif meld_type == "加槓":
        return f"{meld_type} [{all_tiles_str}] (追加: {called_tile_str})"
    elif meld_type == "暗槓":
        return f"{meld_type} [{all_tiles_str}]"
    else:
        return f"{meld_type} [{all_tiles_str}] から{from_str}"


def parse_xml_detailed(xml_file: str, max_actions: int = 100):
    """XMLファイルを詳細に解析して表示する
    
    Args:
        xml_file: XMLファイルのパス
        max_actions: 表示する最大アクション数
    """
    print(f"{'='*80}")
    print(f"XMLファイル解析チェック: {xml_file}")
    print(f"{'='*80}\n")
    
    # XMLファイルを読み込む
    with open(xml_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # プレイヤー名を取得
    un_match = re.search(r'<UN n0="([^"]*)" n1="([^"]*)" n2="([^"]*)" n3="([^"]*)"', content)
    if un_match:
        from urllib.parse import unquote
        player_names = [unquote(un_match.group(i + 1)) for i in range(4)]
    else:
        player_names = ["Player0", "Player1", "Player2", "Player3"]
    
    print(f"【プレイヤー】")
    for i, name in enumerate(player_names):
        print(f"  Player {i}: {name}")
    print()
    
    # 各局を解析
    round_splits = re.split(r'<INIT ', content)
    
    action_count = 0
    
    for round_idx, round_content in enumerate(round_splits[1:]):
        round_content = '<INIT ' + round_content
        
        # INIT タグを解析
        init_match = re.search(
            r'<INIT seed="([^"]+)" ten="([^"]+)" oya="(\d)" hai0="([^"]+)" hai1="([^"]+)" hai2="([^"]+)" hai3="([^"]+)"',
            round_content
        )
        
        if not init_match:
            continue
        
        # 局情報
        seed_parts = init_match.group(1).split(',')
        round_num = int(seed_parts[0])
        honba = int(seed_parts[1])
        riichi_sticks = int(seed_parts[2])
        dora_indicator = int(seed_parts[5])
        
        scores = [int(s) for s in init_match.group(2).split(',')]
        dealer = int(init_match.group(3))
        
        # 配牌
        initial_hands = []
        for i in range(4):
            hand_str = init_match.group(4 + i)
            hand = [int(t) for t in hand_str.split(',')]
            initial_hands.append(hand)
        
        # 局名
        round_names = {
            0: "東1局", 1: "東2局", 2: "東3局", 3: "東4局",
            4: "南1局", 5: "南2局", 6: "南3局", 7: "南4局",
        }
        round_name = round_names.get(round_num, f"Round {round_num}")
        
        print(f"\n{'='*80}")
        print(f"{round_name} (本場: {honba}, 供託: {riichi_sticks})")
        print(f"{'='*80}")
        print(f"親: Player {dealer} ({player_names[dealer]})")
        print(f"ドラ表示牌: {tile_to_string(dora_indicator)}")
        print(f"点数: {' / '.join(str(s) for s in scores)}")
        print()
        
        # 配牌を表示
        print("【配牌】")
        for i, hand in enumerate(initial_hands):
            marker = " (親)" if i == dealer else ""
            print(f"  Player {i}{marker}: {display_hand(hand)}")
        print()
        
        print("【進行】")
        print(f"{'-'*80}")
        
        # ツモと打牌を解析
        turn = 0
        last_discard_player = -1
        last_discard_tile = -1
        
        # 手牌の追跡（各プレイヤー）
        player_hands = [list(hand) for hand in initial_hands]
        
        # すべてのタグを順番に処理
        tags = re.finditer(r'<([A-Z])(\d+)?([^>]*)/?>', round_content)
        
        for tag_match in tags:
            tag_name = tag_match.group(1)
            tag_value = tag_match.group(2)
            tag_attrs = tag_match.group(3)
            
            if action_count >= max_actions:
                break
            
            # ツモ: T(player0), U(player1), V(player2), W(player3)
            if tag_name in 'TUVW' and tag_value:
                player_id = 'TUVW'.index(tag_name)
                tile = int(tag_value)
                
                # 手牌に追加
                player_hands[player_id].append(tile)
                
                # 最初のツモか通常のツモか判定
                if turn == 0 and player_id == dealer:
                    marker = " (親の14枚目)"
                else:
                    marker = ""
                
                print(f"[{turn:3d}] Player {player_id} ({player_names[player_id]}): "
                      f"ツモ → {tile_to_string(tile)}{marker}")
                action_count += 1
            
            # 打牌: D(player0), E(player1), F(player2), G(player3)
            elif tag_name in 'DEFG' and tag_value:
                player_id = 'DEFG'.index(tag_name)
                tile = int(tag_value)
                
                # 手牌から削除
                if tile in player_hands[player_id]:
                    player_hands[player_id].remove(tile)
                else:
                    # 同じ種類の牌を探して削除
                    tile_index = tile // 4
                    for hand_tile in player_hands[player_id]:
                        if hand_tile // 4 == tile_index:
                            player_hands[player_id].remove(hand_tile)
                            break
                
                # 打牌情報を記録（鳴きで使用）
                last_discard_player = player_id
                last_discard_tile = tile
                
                print(f"[{turn:3d}] Player {player_id} ({player_names[player_id]}): "
                      f"打牌 → {tile_to_string(tile)}")
                turn += 1
                action_count += 1
            
            # 鳴き
            elif tag_name == 'N':
                who_match = re.search(r'who="(\d)"', tag_attrs)
                m_match = re.search(r'm="(\d+)"', tag_attrs)
                if who_match and m_match:
                    naki_player = int(who_match.group(1))
                    meld_code = int(m_match.group(1))
                    
                    # 鳴きをデコード
                    meld_info = decode_meld_proper(meld_code)
                    meld_type = meld_info["type"]
                    from_who_relative = meld_info["from_who_relative"]
                    all_tiles = meld_info["tiles"]
                    
                    # 鳴かれた牌と鳴き元を特定
                    called_tile_id = -1
                    from_player = -1
                    consumed_tiles = []
                    
                    if meld_type in ["チー", "ポン", "大明槓"]:
                        # 直前の打牌から鳴いた
                        called_tile_id = last_discard_tile
                        from_player = last_discard_player
                        
                        # 手牌から使った牌を特定
                        called_tile_kind = called_tile_id // 4
                        
                        if meld_type == "チー":
                            # チーの場合、鳴いた牌以外の2枚を手牌から探す
                            for tile_id in all_tiles:
                                if tile_id != called_tile_id and tile_id in player_hands[naki_player]:
                                    consumed_tiles.append(tile_id)
                                    player_hands[naki_player].remove(tile_id)
                            
                            # 見つからない場合は同じ種類の牌を探す
                            if len(consumed_tiles) < 2:
                                for tile_id in all_tiles:
                                    if tile_id == called_tile_id:
                                        continue
                                    tile_kind = tile_id // 4
                                    for hand_tile in player_hands[naki_player]:
                                        if hand_tile // 4 == tile_kind and hand_tile not in consumed_tiles:
                                            consumed_tiles.append(hand_tile)
                                            player_hands[naki_player].remove(hand_tile)
                                            break
                                    if len(consumed_tiles) >= 2:
                                        break
                        
                        elif meld_type in ["ポン", "大明槓"]:
                            # ポン・大明槓の場合、同じ種類の牌を手牌から探す
                            needed = 2 if meld_type == "ポン" else 3
                            for hand_tile in list(player_hands[naki_player]):
                                if hand_tile // 4 == called_tile_kind and len(consumed_tiles) < needed:
                                    consumed_tiles.append(hand_tile)
                                    player_hands[naki_player].remove(hand_tile)
                    
                    elif meld_type == "加槓":
                        # 加槓は手牌から1枚追加
                        called_tile_id = all_tiles[0] if all_tiles else -1
                        from_player = naki_player
                        called_tile_kind = called_tile_id // 4
                        
                        for hand_tile in player_hands[naki_player]:
                            if hand_tile // 4 == called_tile_kind:
                                consumed_tiles.append(hand_tile)
                                player_hands[naki_player].remove(hand_tile)
                                break
                    
                    elif meld_type == "暗槓":
                        # 暗槓は手牌から4枚
                        called_tile_id = all_tiles[0] if all_tiles else -1
                        from_player = naki_player
                        called_tile_kind = called_tile_id // 4
                        
                        for hand_tile in list(player_hands[naki_player]):
                            if hand_tile // 4 == called_tile_kind:
                                consumed_tiles.append(hand_tile)
                                player_hands[naki_player].remove(hand_tile)
                    
                    # 鳴き情報を表示
                    meld_str = display_meld(meld_info, called_tile_id, from_player, 
                                           naki_player, player_names, consumed_tiles)
                    print(f"[{turn:3d}] Player {naki_player} ({player_names[naki_player]}): "
                          f"鳴き → {meld_str}")
                    action_count += 1
                    
                    # 鳴きの後は最後の打牌情報をクリア
                    last_discard_player = -1
                    last_discard_tile = -1
            
            # リーチ
            elif tag_name == 'R':  # REACH
                who_match = re.search(r'who="(\d)"', tag_attrs)
                step_match = re.search(r'step="(\d)"', tag_attrs)
                if who_match and step_match:
                    player_id = int(who_match.group(1))
                    step = int(step_match.group(1))
                    if step == 1:
                        print(f"[{turn:3d}] Player {player_id} ({player_names[player_id]}): "
                              f"リーチ宣言")
                        action_count += 1
            
            # 和了
            elif tag_name == 'A':  # AGARI
                who_match = re.search(r'who="(\d)"', tag_attrs)
                from_match = re.search(r'fromWho="(\d)"', tag_attrs)
                ten_match = re.search(r'ten="([^"]+)"', tag_attrs)
                if who_match and from_match:
                    winner = int(who_match.group(1))
                    from_who = int(from_match.group(1))
                    if winner == from_who:
                        agari_type = "ツモ"
                    else:
                        agari_type = f"ロン (放銃: Player {from_who})"
                    
                    points_info = ""
                    if ten_match:
                        ten_parts = ten_match.group(1).split(',')
                        if len(ten_parts) >= 2:
                            fu = ten_parts[0]
                            points = ten_parts[1]
                            points_info = f" ({fu}符 {points}点)"
                    
                    print(f"[{turn:3d}] Player {winner} ({player_names[winner]}): "
                          f"和了 → {agari_type}{points_info}")
                    action_count += 1
                break  # 局終了
            
            # 流局
            elif tag_name == 'R' and 'RYUUKYOKU' in tag_match.group(0):
                print(f"[{turn:3d}] 流局")
                action_count += 1
                break
        
        if action_count >= max_actions:
            remaining = sum(1 for _ in re.finditer(r'<[TUVWDEFG]\d+/>', content)) - action_count
            print(f"\n... 残り約 {remaining}+ アクションは省略 ...")
            break
        
        print()
    
    print(f"{'-'*80}")
    print("\n✅ XML解析確認完了")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='XMLファイルの解析結果を詳細に確認する')
    parser.add_argument('xml_file', nargs='?', 
                       default='2009080100gm-00e1-0000-63d644dd.xml',
                       help='解析するXMLファイルのパス')
    parser.add_argument('--max-actions', type=int, default=100,
                       help='表示する最大アクション数（デフォルト: 100）')
    parser.add_argument('--all', action='store_true',
                       help='すべてのアクションを表示する')
    
    args = parser.parse_args()
    
    # XMLファイルのパスを解決
    xml_path = Path(args.xml_file)
    if not xml_path.is_absolute():
        xml_path = Path(__file__).parent.parent / args.xml_file
    
    if not xml_path.exists():
        print(f"❌ エラー: ファイルが見つかりません: {xml_path}")
        sys.exit(1)
    
    max_actions = float('inf') if args.all else args.max_actions
    
    parse_xml_detailed(str(xml_path), max_actions=max_actions)


if __name__ == '__main__':
    main()

