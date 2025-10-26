# 変更内容: 教師あり学習のみへの移行

## 削除されたファイル

### トレーニング関連
- ❌ `src/training/train_rl.py` - 強化学習トレーナー
- ❌ `src/training/reward_fanback.py` - Fan Backward報酬システム

### 設定ファイル
- ❌ `configs/reward_config.yaml` - RL報酬設定

## 追加されたファイル

### ユーティリティ
- ✅ `src/utils/shanten.py` - シャンテン数計算機能
  - `calculate_shanten()` - 基本シャンテン計算
  - `analyze_hand_details()` - 詳細な手牌分析
  - `tiles_list_to_34_array()` - 牌変換

### ゲーム状態管理
- ✅ `src/preprocessing/game_state.py` - 打牌履歴管理
  - `RoundGameState` - 局の状態
  - `DiscardHistory` - 打牌履歴データクラス
  - `GameStateManager` - 状態管理マネージャー

### ドキュメント
- ✅ `SUPERVISED_ONLY.md` - 教師あり学習専用ガイド
- ✅ `CHANGES_SUPERVISED_ONLY.md` - このファイル

## 更新されたファイル

### 設定ファイル
#### `configs/train_config.yaml`
```yaml
# 追加
data:
  use_discard_history: true
  max_discard_history: 64

# 追加
evaluation:
  compute_shanten: true
  top_k_accuracies: [1, 3, 5]

# 削除: RL関連の設定
```

#### `configs/model_config.yaml`
```yaml
# 追加
discard_history:
  max_history_length: 64
  use_player_embedding: true
  use_temporal_encoding: true

# 削除: sequence.max_sequence_length
```

### ソースコード
#### `src/training/__init__.py`
```python
# 削除
from .train_rl import RLTrainer, PPOMemory
from .reward_fanback import FanBackwardReward, RewardShaper

# 現在
from .train_supervised import SupervisedTrainer, create_optimizer, create_scheduler
```

#### `src/preprocessing/__init__.py`
```python
# 追加
from .game_state import RoundGameState, DiscardHistory, GameStateManager
```

#### `src/utils/__init__.py`
```python
# 追加
from .shanten import (
    calculate_shanten,
    get_shanten_after_best_discard,
    analyze_hand_details,
    print_hand_analysis,
    tiles_list_to_34_array,
    format_tiles_for_display,
    format_shanten
)
```

#### `requirements.txt`
```txt
# 追加
mahjong>=1.2.0
```

### ドキュメント
#### `README.md`
- RL関連の記述を削除
- シャンテン計算の説明を追加
- 打牌履歴機能の説明を追加

## 主要な変更点

### 1. 学習アプローチ
**変更前:**
- 教師あり学習 → 強化学習（PPO）
- Fan Backward報酬システム

**変更後:**
- 教師あり学習のみ
- シンプルなクロスエントロピー損失

### 2. 特徴量
**変更前:**
- 直近4手のみを保持

**変更後:**
- 局の初めから全打牌を保持（最大64手）
- 誰がいつ打牌したかを記録
- リーチ宣言も記録

### 3. シャンテン数
**変更前:**
- 簡易的な推定のみ

**変更後:**
- python-mahjongライブラリを使用した正確な計算
- 一般形・七対子・国士無双の全形式対応
- 詳細な受け入れ分析

## 使用方法の変更

### データ構築
```python
# 変更なし
from src.preprocessing import MahjongDatasetBuilder
builder = MahjongDatasetBuilder('data/raw', 'data/processed')
builder.build_dataset()
```

### トレーニング
```python
# 変更前: RL Trainerも使用
from src.training import SupervisedTrainer, RLTrainer

# 変更後: SupervisedTrainerのみ
from src.training import SupervisedTrainer
```

### 打牌履歴の使用（新機能）
```python
from src.preprocessing import GameStateManager

manager = GameStateManager()
manager.start_new_round(round_num=0, dealer=0)
manager.add_discard(player_id=0, tile=120, turn=1)

state = manager.get_current_state()
history = state.discard_history  # 全打牌履歴
recent = state.get_recent_discards(num_turns=4)  # 直近4手
```

### シャンテン数計算（新機能）
```python
from src.utils import calculate_shanten, analyze_hand_details

# 基本計算
shanten = calculate_shanten(tiles_34_array)

# 詳細分析
analysis = analyze_hand_details("123m456p789s1122z")
print(f"Best shanten: {analysis['best_shanten']}")
```

## マイグレーションガイド

### 既存のコードがある場合

1. **RL関連のimportを削除:**
```python
# 削除
from src.training import RLTrainer
from src.training import FanBackwardReward
```

2. **設定ファイルを更新:**
```bash
# configs/reward_config.yamlは不要
rm configs/reward_config.yaml
```

3. **新機能を追加（オプション）:**
```python
# シャンテン計算を追加
from src.utils import calculate_shanten

# 打牌履歴管理を追加
from src.preprocessing import GameStateManager
```

### 新規プロジェクトの場合

1. **インストール:**
```bash
pip install -r requirements.txt
```

2. **確認:**
```bash
python3 test_installation.py
python3 check_model_params.py
```

3. **トレーニング:**
```bash
bash run_train.sh
```

## 期待される効果

### メリット
1. ✅ **シンプル化**: RLの複雑さを排除
2. ✅ **安定性**: 学習が安定
3. ✅ **デバッグ容易**: 問題の特定が簡単
4. ✅ **高速**: 学習時間の短縮
5. ✅ **打牌履歴**: より豊富な文脈情報
6. ✅ **シャンテン**: 正確な手牌評価

### デメリット
1. ❌ 報酬ベースの最適化ができない
2. ❌ エキスパートデータの質に依存
3. ❌ 新戦略の創造は期待できない

しかし、教師あり学習のみでも**十分な精度**が得られます！

## トラブルシューティング

### エラー: "cannot import RLTrainer"
**原因**: 削除されたモジュールをimportしようとしています

**解決策**:
```python
# 削除
# from src.training import RLTrainer

# これのみ使用
from src.training import SupervisedTrainer
```

### エラー: "mahjong module not found"
**原因**: python-mahjongライブラリがインストールされていません

**解決策**:
```bash
pip install mahjong
```

### エラー: "reward_config.yaml not found"
**原因**: 削除された設定ファイルを読み込もうとしています

**解決策**: 
`train_config.yaml`と`model_config.yaml`のみを使用してください

## まとめ

- ✅ RL機能を完全に削除
- ✅ 教師あり学習のみに集中
- ✅ シャンテン数計算を追加
- ✅ 打牌履歴管理を追加
- ✅ シンプルで安定した学習パイプライン

**これで教師あり学習のみでトレーニング可能です！** 🎓


