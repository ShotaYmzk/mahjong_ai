"""シャンテン数計算ユーティリティ

python-mahjongライブラリを使用してシャンテン数を計算します。
"""

import re
from typing import List, Dict, Tuple, Optional

try:
    from mahjong.shanten import Shanten
    from mahjong.tile import TilesConverter
    MAHJONG_LIB_AVAILABLE = True
except ImportError:
    MAHJONG_LIB_AVAILABLE = False
    print("Warning: python-mahjong library not found. Shanten calculation will use fallback method.")


def robust_hand_parser(hand_string: str) -> List[int]:
    """
    "123m456m"や"0s"(赤5索)のような変則的な文字列も正しく解析するパーサー。
    
    Args:
        hand_string: 手牌文字列 (例: "123m456p789s1122z")
        
    Returns:
        34種類の牌の枚数配列
    """
    if not MAHJONG_LIB_AVAILABLE:
        raise ImportError("python-mahjong library is required for hand parsing")
    
    # 正規表現を使って、各スーツの数字をすべて抽出して結合する
    man = "".join(re.findall(r'([0-9]+)m', hand_string))
    pin = "".join(re.findall(r'([0-9]+)p', hand_string))
    sou = "".join(re.findall(r'([0-9]+)s', hand_string))
    honors = "".join(re.findall(r'([0-9]+)z', hand_string))
    
    # まず has_aka_dora=True オプション付きで136種の牌に変換
    tiles_136 = TilesConverter.string_to_136_array(
        man=man, pin=pin, sou=sou, honors=honors, has_aka_dora=True
    )
    
    # 136種の牌から34種の配列に変換して返す
    return TilesConverter.to_34_array(tiles_136)


def tiles_list_to_34_array(tile_ids: List[int]) -> List[int]:
    """
    牌IDのリスト (0-135) を34種類の牌の枚数配列に変換
    
    Args:
        tile_ids: 牌IDのリスト (Tenhou形式: 0-135)
        
    Returns:
        34種類の牌の枚数配列
    """
    tiles_34 = [0] * 34
    for tile_id in tile_ids:
        tile_type = tile_id // 4  # 4で割って牌の種類を取得
        if 0 <= tile_type < 34:
            tiles_34[tile_type] += 1
    return tiles_34


def format_tiles_for_display(tile_indices: List[int]) -> str:
    """
    牌のインデックスのリストを表示用の文字列に変換します。
    
    Args:
        tile_indices: 牌インデックスのリスト (0-33)
        
    Returns:
        表示用文字列 (例: "123m456p")
    """
    if not tile_indices:
        return ""
        
    man = sorted([i for i in tile_indices if 0 <= i <= 8])
    pin = sorted([i for i in tile_indices if 9 <= i <= 17])
    sou = sorted([i for i in tile_indices if 18 <= i <= 26])
    honors = sorted([i for i in tile_indices if 27 <= i <= 33])
    
    result_str = ""
    if man:
        # 赤5萬は0mと表示
        result_str += "".join(['0' if t == 4 else str(t + 1) for t in man]) + "m"
    if pin:
        # 赤5筒は0pと表示
        result_str += "".join(['0' if t == 13 else str(t - 9 + 1) for t in pin]) + "p"
    if sou:
        # 赤5索は0sと表示
        result_str += "".join(['0' if t == 22 else str(t - 18 + 1) for t in sou]) + "s"
    if honors:
        result_str += "".join([str(t - 27 + 1) for t in honors]) + "z"
        
    return result_str


def format_shanten(shanten_value: int) -> str:
    """
    シャンテン数を「N向聴」または「聴牌」の文字列に変換します。
    
    Args:
        shanten_value: シャンテン数
        
    Returns:
        表示用文字列
    """
    if shanten_value == 0:
        return "聴牌"
    if shanten_value < 0:
        return "和了"
    return f"{shanten_value}向聴"


def calculate_shanten(tiles_34: List[int], shanten_type: str = 'regular') -> int:
    """
    シャンテン数を計算
    
    Args:
        tiles_34: 34種類の牌の枚数配列
        shanten_type: 'regular', 'chiitoitsu', 'kokushi', 'all' (最小値)
        
    Returns:
        シャンテン数 (-1=和了, 0=聴牌, 1以上=向聴)
    """
    if not MAHJONG_LIB_AVAILABLE:
        # Fallback: 簡易シャンテン計算
        return calculate_shanten_simple(tiles_34)
    
    calculator = Shanten()
    
    if shanten_type == 'regular':
        return calculator.calculate_shanten_for_regular_hand(tiles_34)
    elif shanten_type == 'chiitoitsu':
        return calculator.calculate_shanten_for_chiitoitsu_hand(tiles_34)
    elif shanten_type == 'kokushi':
        return calculator.calculate_shanten_for_kokushi_hand(tiles_34)
    else:  # 'all' or default
        return calculator.calculate_shanten(tiles_34)


def calculate_shanten_simple(tiles_34: List[int]) -> int:
    """
    簡易シャンテン計算 (python-mahjongが使えない場合のフォールバック)
    
    Args:
        tiles_34: 34種類の牌の枚数配列
        
    Returns:
        推定シャンテン数
    """
    total_tiles = sum(tiles_34)
    if total_tiles not in [13, 14]:
        return 8  # 異常な枚数
    
    # 孤立牌の数をカウント
    isolated_tiles = 0
    for i, count in enumerate(tiles_34):
        if count > 0:
            # 前後の牌がなく、同じ牌も1枚しかない場合は孤立
            has_neighbor = False
            if i > 0 and tiles_34[i-1] > 0:
                has_neighbor = True
            if i < 33 and tiles_34[i+1] > 0:
                has_neighbor = True
            if count > 1:
                has_neighbor = True
            
            if not has_neighbor:
                isolated_tiles += 1
    
    # 簡易推定: 孤立牌が多いほどシャンテン数が大きい
    if isolated_tiles <= 1:
        return 0  # 聴牌の可能性
    elif isolated_tiles <= 3:
        return 1
    elif isolated_tiles <= 5:
        return 2
    else:
        return 3


def get_shanten_after_best_discard(tiles_14: List[int], shanten_type: str = 'all') -> int:
    """
    14枚の手牌から1枚捨てて13枚にした時の、最小シャンテン数を計算します。
    
    Args:
        tiles_14: 14枚の手牌 (34種配列)
        shanten_type: シャンテン計算タイプ
        
    Returns:
        最小シャンテン数
    """
    if not MAHJONG_LIB_AVAILABLE:
        return calculate_shanten_simple(tiles_14)
    
    calculator = Shanten()
    min_shanten = 8
    
    unique_tiles_in_hand = [i for i, count in enumerate(tiles_14) if count > 0]
    if not unique_tiles_in_hand:
        return min_shanten

    for discard_index in unique_tiles_in_hand:
        temp_hand_13 = list(tiles_14)
        temp_hand_13[discard_index] -= 1
        
        if shanten_type == 'all':
            shanten = calculator.calculate_shanten(temp_hand_13)
        elif shanten_type == 'regular':
            shanten = calculator.calculate_shanten_for_regular_hand(temp_hand_13)
        elif shanten_type == 'chiitoitsu':
            shanten = calculator.calculate_shanten_for_chiitoitsu_hand(temp_hand_13)
        elif shanten_type == 'kokushi':
            shanten = calculator.calculate_shanten_for_kokushi_hand(temp_hand_13)
        else:
            shanten = calculator.calculate_shanten(temp_hand_13)
        
        if shanten < min_shanten:
            min_shanten = shanten
            
    return min_shanten


def analyze_hand_details(hand_string: str) -> Dict:
    """
    手牌のシャンテン数を形式別に計算し、最適な打牌と受け入れを分析します。
    
    Args:
        hand_string: 手牌文字列
        
    Returns:
        分析結果の辞書
    """
    if not MAHJONG_LIB_AVAILABLE:
        raise ImportError("python-mahjong library is required for detailed analysis")
    
    shanten_calculator = Shanten()
    
    # 新しいパーサーを使って手牌を34種配列に変換
    tiles_34_14 = robust_hand_parser(hand_string)
    unique_tiles_in_hand_14 = sorted([i for i, count in enumerate(tiles_34_14) if count > 0])

    # --- 形式別シャンテン数計算 ---
    shanten_regular = get_shanten_after_best_discard(tiles_34_14, 'regular')
    shanten_chiitoitsu = get_shanten_after_best_discard(tiles_34_14, 'chiitoitsu')
    shanten_kokushi = get_shanten_after_best_discard(tiles_34_14, 'kokushi')

    # --- 最適な打牌と受け入れの計算 ---
    analysis_results = []
    
    for discard_index in unique_tiles_in_hand_14:
        hand_13_tiles = list(tiles_34_14)
        hand_13_tiles[discard_index] -= 1
        shanten_13 = shanten_calculator.calculate_shanten(hand_13_tiles)
        
        ukeire_for_discard = {}
        
        if shanten_13 == 0:  # 聴牌の場合: 待ち牌を計算
            for draw_index in range(34):
                # 5枚目になる牌は引けない and 自分が捨てた牌はフリテンになるので待ちに含めない
                if tiles_34_14[draw_index] < 4 and draw_index != discard_index:
                    temp_hand_14 = list(hand_13_tiles)
                    temp_hand_14[draw_index] += 1
                    # アガリ(-1)になる牌を探す
                    if shanten_calculator.calculate_shanten(temp_hand_14) == -1:
                        remaining_count = 4 - tiles_34_14[draw_index]
                        ukeire_for_discard[draw_index] = remaining_count
        else:  # 聴牌していない場合: シャンテン数を進める牌を計算
            for draw_index in range(34):
                if tiles_34_14[draw_index] < 4:
                    hand_14_after_draw = list(hand_13_tiles)
                    hand_14_after_draw[draw_index] += 1
                    shanten_after_draw_and_discard = get_shanten_after_best_discard(
                        hand_14_after_draw, 'all'
                    )
                    if shanten_after_draw_and_discard < shanten_13:
                        remaining_count = 4 - tiles_34_14[draw_index]
                        ukeire_for_discard[draw_index] = remaining_count

        total_ukeire_count = sum(ukeire_for_discard.values())
        analysis_results.append({
            "discard_index": discard_index,
            "ukeire": ukeire_for_discard,
            "total_count": total_ukeire_count,
            "shanten_after_discard": shanten_13
        })
    
    sorted_results = sorted(
        analysis_results, 
        key=lambda x: (x['shanten_after_discard'], -x['total_count'])
    )
    
    return {
        'hand_string': hand_string,
        'tiles_34': tiles_34_14,
        'shanten_regular': shanten_regular,
        'shanten_chiitoitsu': shanten_chiitoitsu,
        'shanten_kokushi': shanten_kokushi,
        'best_shanten': min(shanten_regular, shanten_chiitoitsu, shanten_kokushi),
        'discard_analysis': sorted_results
    }


def print_hand_analysis(analysis: Dict):
    """
    手牌分析結果を表示
    
    Args:
        analysis: analyze_hand_details()の戻り値
    """
    print(f"現在の手牌: {analysis['hand_string']}")
    print("--- 形式別シャンテン数 ---")
    print(f"一般形: {format_shanten(analysis['shanten_regular'])}")
    print(f"七対子: {format_shanten(analysis['shanten_chiitoitsu'])}")
    print(f"国士無双: {format_shanten(analysis['shanten_kokushi'])}")
    print("-" * 30)
    
    print("--- 打牌候補と受け入れ（シャンテン数・枚数順）---")
    for result in analysis['discard_analysis']:
        discard_str = format_tiles_for_display([result["discard_index"]])
        ukeire_str = format_tiles_for_display(sorted(result["ukeire"].keys()))
        total_枚数 = result["total_count"]
        shanten_display = format_shanten(result["shanten_after_discard"])
        
        print(f"打{discard_str} ({shanten_display}) 摸[{ukeire_str} {total_枚数}枚]")

