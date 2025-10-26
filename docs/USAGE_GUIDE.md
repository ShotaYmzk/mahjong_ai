# データ変換パイプライン使用ガイド

## 目次

1. [概要](#概要)
2. [インストール](#インストール)
3. [クイックスタート](#クイックスタート)
4. [詳細な使用方法](#詳細な使用方法)
5. [設定オプション](#設定オプション)
6. [トラブルシューティング](#トラブルシューティング)
7. [高度な使用例](#高度な使用例)

## 概要

本パイプラインは、天鳳の麻雀XMLログを教師あり学習用の固定長ベクトルデータに変換します。

### 主な機能

- ✅ XML解析（手出し/ツモ切り検出、赤ドラ追跡）
- ✅ 包括的なゲーム状態追跡
- ✅ 2,099次元（可変）の固定長特徴ベクトル生成
- ✅ ゲーム単位でのtrain/val/test分割（データリーク防止）
- ✅ 全ステップでのデータ検証
- ✅ 詳細な統計情報出力

## インストール

### 1. 依存パッケージのインストール

```bash
cd /home/ubuntu/Documents/mahjong_ai
pip install -r requirements.txt
```

### 2. 追加パッケージ（オプション）

```bash
# プログレスバー表示用
pip install tqdm

# 高速化用（マルチプロセス）
pip install joblib
```

## クイックスタート

### 最も簡単な使用方法

```bash
# データ収集（天鳳XMLダウンロード）
python scripts/collect_tenhou_data.py --max-files 100

# データセット構築
python scripts/build_comprehensive_dataset.py --max-games 100
```

これで以下が生成されます：

```
data/processed_v2/
├── train/
│   ├── X_train.npy          # 訓練データ特徴量
│   ├── y_train.npy          # 訓練データラベル
│   └── metadata_train.json  # メタデータ
├── val/
│   ├── X_val.npy
│   ├── y_val.npy
│   └── metadata_val.json
├── test/
│   ├── X_test.npy
│   ├── y_test.npy
│   └── metadata_test.json
├── split_info.json          # 分割情報
└── dataset_info.json        # 統計情報
```

## 詳細な使用方法

### Step 1: XMLログの準備

XMLログは以下のいずれかの方法で取得します：

#### 方法A: データ収集スクリプト使用

```bash
python scripts/collect_tenhou_data.py \
    --years 2024 2023 \
    --game-type 四鳳 \
    --max-files 10000
```

#### 方法B: 手動配置

XMLファイルを以下に配置：
```
data/xml_logs/
├── 2024010100gm-00e1-0000-xxxxxxxx.xml
├── 2024010101gm-00e1-0000-xxxxxxxx.xml
└── ...
```

### Step 2: データセット構築

#### 基本的な使用

```bash
python scripts/build_comprehensive_dataset.py
```

#### カスタム設定

```bash
python scripts/build_comprehensive_dataset.py \
    --xml-dir data/xml_logs \
    --output-dir data/processed_v2 \
    --max-games 50000 \
    --draw-history 8 \
    --discard-history 32 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Step 3: データセットの確認

#### Pythonで統計確認

```python
from src.preprocessing import ComprehensiveDatasetBuilder

builder = ComprehensiveDatasetBuilder(
    xml_dir='',
    output_dir='data/processed_v2'
)

# 統計情報を表示
builder.print_statistics()
```

#### データの読み込み

```python
import numpy as np

# 訓練データの読み込み
X_train = np.load('data/processed_v2/train/X_train.npy')
y_train = np.load('data/processed_v2/train/y_train.npy')

print(f"訓練データサイズ: {X_train.shape}")
print(f"特徴量次元数: {X_train.shape[1]}")
print(f"サンプル数: {X_train.shape[0]}")
```

### Step 4: 学習の開始

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# PyTorch Dataset作成
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

# DataLoader作成
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4
)

# モデルの学習
for epoch in range(100):
    for batch_X, batch_y in train_loader:
        # ... 学習コード ...
        pass
```

## 設定オプション

### ディレクトリ設定

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--xml-dir` | `data/xml_logs` | XMLファイルのディレクトリ |
| `--output-dir` | `data/processed_v2` | 出力ディレクトリ |

### データ設定

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--max-games` | `None` | 処理する最大ゲーム数 |

### 特徴量設定

| オプション | デフォルト | 説明 | 影響する次元数 |
|-----------|----------|------|--------------|
| `--draw-history` | `8` | ツモ履歴の長さ (k) | k × 34 |
| `--discard-history` | `32` | 捨て牌履歴の長さ (m) | m × 39 |
| `--no-shanten` | False | 向聴数計算を無効化 | - |
| `--no-danger` | False | 危険度推定を無効化 | - |

**特徴量次元数の計算:**

```
総次元数 = 340 + (draw_history × 34) + (discard_history × 39) + 31 + 38 + 170

デフォルト (draw=8, discard=32):
  = 340 + (8×34) + (32×39) + 31 + 38 + 170
  = 340 + 272 + 1,248 + 31 + 38 + 170
  = 2,099次元
```

### 分割設定

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--train-ratio` | `0.80` | 訓練データの割合 |
| `--val-ratio` | `0.10` | 検証データの割合 |
| `--test-ratio` | `0.10` | テストデータの割合 |
| `--random-seed` | `42` | 乱数シード |

### 処理設定

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--no-validate` | False | データ検証を無効化 |
| `--no-progress` | False | プログレスバーを非表示 |
| `--log-file` | `dataset_building.log` | ログファイルのパス |

## トラブルシューティング

### 問題1: メモリ不足

**症状:** `MemoryError` が発生する

**解決策:**
```bash
# ゲーム数を減らす
python scripts/build_comprehensive_dataset.py --max-games 10000

# 履歴長を減らす（特徴量次元数を削減）
python scripts/build_comprehensive_dataset.py \
    --draw-history 4 \
    --discard-history 16
```

### 問題2: 処理が遅い

**症状:** 処理に時間がかかりすぎる

**解決策:**
```bash
# プログレスバーで進捗確認
python scripts/build_comprehensive_dataset.py

# 並列処理（将来のアップデートで対応）
# 現時点では sequential のみ
```

### 問題3: XMLファイルが見つからない

**症状:** `No games found in ...`

**解決策:**
```bash
# XMLファイルの確認
ls data/xml_logs/*.xml | head

# パスの確認
python scripts/build_comprehensive_dataset.py \
    --xml-dir /path/to/your/xml_logs
```

### 問題4: 検証エラー

**症状:** `Validation failed`

**解決策:**
```bash
# 検証を無効化して続行（推奨しない）
python scripts/build_comprehensive_dataset.py --no-validate

# ログを確認してエラー原因を特定
cat dataset_building.log | grep ERROR
```

## 高度な使用例

### 例1: Pythonスクリプトからの使用

```python
from src.preprocessing import ComprehensiveDatasetBuilder

# カスタム設定
config = {
    'draw_history_length': 16,
    'discard_history_length': 64,
    'enable_shanten_calc': True,
    'enable_danger_estimation': True,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
}

# ビルダー初期化
builder = ComprehensiveDatasetBuilder(
    xml_dir='data/xml_logs',
    output_dir='data/processed_custom',
    config=config
)

# データセット構築
stats = builder.build_complete_dataset(
    max_games=100000,
    validate=True,
    show_progress=True
)

# 統計表示
builder.print_statistics()

# データ読み込み
X_train, y_train, metadata_train = builder.load_dataset('train')
X_val, y_val, metadata_val = builder.load_dataset('val')
X_test, y_test, metadata_test = builder.load_dataset('test')
```

### 例2: 特定プレイヤーのデータのみ抽出

```python
from src.preprocessing import (
    EnhancedXMLParser,
    ComprehensiveGameStateTracker,
    AdvancedFeatureEncoderV2
)

# 解析
parser = EnhancedXMLParser()
games = parser.parse_directory('data/xml_logs', max_files=1000)

# 高レート（2000以上）のプレイヤーのみフィルタ
high_rated_games = [
    game for game in games
    if any(rating >= 2000 for rating in game.player_ratings)
]

# 状態抽出
tracker = ComprehensiveGameStateTracker()
states = []
for game in high_rated_games:
    game_states = tracker.extract_all_states(game)
    # レート2000以上のプレイヤーのみ
    filtered_states = [
        state for state in game_states
        if game.player_ratings[state.player_id] >= 2000
    ]
    states.extend(filtered_states)

# エンコード
encoder = AdvancedFeatureEncoderV2()
X = np.stack([encoder.encode_state(state) for state in states])
y = np.array([state.label_tile_type for state in states])

print(f"高レートプレイヤーデータ: {X.shape}")
```

### 例3: カスタム特徴量の追加

```python
from src.preprocessing.feature_encoder_v2 import AdvancedFeatureEncoderV2
import numpy as np

class CustomFeatureEncoder(AdvancedFeatureEncoderV2):
    """カスタム特徴量エンコーダー"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # カスタム特徴量の次元を追加
        self.custom_dim = 10
        self.total_dim += self.custom_dim
    
    def encode_state(self, state):
        """カスタム特徴量を追加"""
        # 基本的な特徴量
        base_features = super().encode_state(state)
        
        # カスタム特徴量（例: 点数関連）
        custom_features = self._encode_custom_features(state)
        
        # 結合
        return np.concatenate([base_features, custom_features])
    
    def _encode_custom_features(self, state):
        """カスタム特徴量の計算"""
        features = []
        
        # 1. 点差（自分と各プレイヤー）
        for score in state.scores:
            score_diff = (state.scores[state.player_id] - score) / 100000.0
            features.append(score_diff)
        
        # 2. 平均点との差
        avg_score = np.mean(state.scores)
        features.append((state.scores[state.player_id] - avg_score) / 100000.0)
        
        # 3. 局の進行度
        round_progress = (state.round_num + state.round_index / 4.0) / 8.0
        features.append(round_progress)
        
        # 4-10. その他のカスタム特徴
        features.extend([0.0] * 4)
        
        return np.array(features, dtype=np.float32)

# 使用
custom_encoder = CustomFeatureEncoder(
    draw_history_length=8,
    discard_history_length=32
)
print(f"カスタムエンコーダーの特徴量次元: {custom_encoder.total_dim}")
```

### 例4: バッチ処理（大規模データ）

```python
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_in_batches(xml_dir, output_dir, batch_size=1000):
    """バッチ処理でメモリ効率を向上"""
    
    from src.preprocessing import (
        EnhancedXMLParser,
        ComprehensiveGameStateTracker,
        AdvancedFeatureEncoderV2
    )
    
    parser = EnhancedXMLParser()
    tracker = ComprehensiveGameStateTracker()
    encoder = AdvancedFeatureEncoderV2()
    
    # XMLファイルのリスト取得
    xml_files = list(Path(xml_dir).glob('*.xml'))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # バッチごとに処理
    for batch_idx, i in enumerate(range(0, len(xml_files), batch_size)):
        batch_files = xml_files[i:i+batch_size]
        
        print(f"Processing batch {batch_idx+1}/{(len(xml_files)-1)//batch_size+1}")
        
        # 解析
        games = []
        for xml_file in tqdm(batch_files, desc="Parsing"):
            game = parser.parse_file(str(xml_file))
            if game:
                games.append(game)
        
        # 状態抽出
        all_states = []
        for game in tqdm(games, desc="Extracting"):
            states = tracker.extract_all_states(game)
            all_states.extend(states)
        
        # エンコード
        X_batch = []
        y_batch = []
        for state in tqdm(all_states, desc="Encoding"):
            X_batch.append(encoder.encode_state(state))
            y_batch.append(state.label_tile_type)
        
        # 保存
        X_batch = np.stack(X_batch)
        y_batch = np.array(y_batch)
        
        np.save(output_dir / f'X_batch_{batch_idx:04d}.npy', X_batch)
        np.save(output_dir / f'y_batch_{batch_idx:04d}.npy', y_batch)
        
        print(f"Batch {batch_idx} saved: {X_batch.shape}")

# 使用
process_in_batches(
    xml_dir='data/xml_logs',
    output_dir='data/processed_batches',
    batch_size=1000
)
```

## データ形式の詳細

### 特徴ベクトル (X)

Shape: `[N_samples, D]` where D = 2,099 (default)

```python
# X の構造
X[i, :340]              # 基本特徴量（1x34 × 10）
X[i, 340:612]           # ツモ履歴（8 × 34）
X[i, 612:1860]          # 捨て牌履歴（32 × 39）
X[i, 1860:1891]         # メタ特徴量（31）
X[i, 1891:1929]         # 待ち・向聴数（38）
X[i, 1929:2099]         # 捨て牌メタ（170）
```

### ラベルベクトル (y)

Shape: `[N_samples]`

```python
# y の値: 0-33（打牌した牌のタイプ）
# 0-8:   1m-9m
# 9-17:  1p-9p
# 18-26: 1s-9s
# 27-33: E,S,W,N,白,発,中
```

### メタデータ

```json
{
  "game_id": "2024010100gm-00e1-0000-12345678",
  "round_num": 0,
  "player_id": 0,
  "turn": 5,
  "label_tile_type": 12
}
```

## パフォーマンス目安

### 処理時間（目安）

| ゲーム数 | 処理時間（概算） |
|---------|---------------|
| 100 | 1-2分 |
| 1,000 | 10-20分 |
| 10,000 | 2-4時間 |
| 100,000 | 20-40時間 |

**注:** 8コアCPU、32GB RAMの環境での目安

### メモリ使用量

| ゲーム数 | 特徴量次元 | メモリ使用量（概算） |
|---------|----------|------------------|
| 1,000 | 2,099 | ~400 MB |
| 10,000 | 2,099 | ~4 GB |
| 100,000 | 2,099 | ~40 GB |

## 参考資料

- [データ変換パイプライン設計書](DATA_CONVERSION_PIPELINE.md)
- [天鳳ログフォーマット](https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format)
- [README](../README.md)

---

**問い合わせ:** 問題が解決しない場合は、GitHubのIssueを作成してください。

