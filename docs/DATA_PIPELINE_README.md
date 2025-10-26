# 天鳳麻雀データ変換パイプライン

## 概要

このパイプラインは、天鳳の約20万試合のXMLログから、教師あり学習モデル用の固定長ベクトル化データへの完全な変換を提供します。

## 主な特徴

✅ **包括的なXML解析**
- 手出し/ツモ切り判定
- 赤ドラ（5m/5p/5s）検出
- リーチ宣言巡目の記録
- ドラ増加（カン）の追跡

✅ **完全なゲーム状態追跡**
- 手牌・副露の再構成
- ツモ履歴の記録
- 河（捨て牌）履歴の管理
- リーチ状態の追跡
- 点数変動の記録

✅ **高度な特徴量エンコーディング（2,099次元）**
- 基本特徴量（340次元）: 1x34 × 10グループ
- 時系列特徴量（1,520次元）: ツモ・捨て牌履歴
- メタ特徴量（31次元）: 局情報・点数・リーチ状態
- 待ち・向聴数（38次元）: テンパイ・待ち牌情報
- 捨て牌メタ（170次元）: 手出し/ツモ切り比率・危険度

✅ **データ整合性保証**
- 全ステップでの検証
- ゲーム単位での分割（データリーク防止）
- 詳細なエラーレポート

✅ **使いやすいインターフェース**
- ワンコマンドでのデータセット構築
- 柔軟な設定オプション
- 詳細な統計情報出力

## クイックスタート

### 1. インストール

```bash
cd /home/ubuntu/Documents/mahjong_ai
pip install -r requirements.txt
pip install tqdm  # オプション: プログレスバー表示
```

### 2. データ収集

```bash
# 天鳳XMLログをダウンロード（例: 100ゲーム）
python scripts/collect_tenhou_data.py --max-files 100
```

### 3. データセット構築

```bash
# 包括的データセットを構築
python scripts/build_comprehensive_dataset.py --max-games 100
```

### 4. 構築されたデータの確認

```python
import numpy as np

# データの読み込み
X_train = np.load('data/processed_v2/train/X_train.npy')
y_train = np.load('data/processed_v2/train/y_train.npy')

print(f"訓練データ: {X_train.shape}")
print(f"特徴量次元: {X_train.shape[1]}")
```

## ドキュメント

- **[設計書](DATA_CONVERSION_PIPELINE.md)**: 詳細なアーキテクチャと仕様
- **[使用ガイド](USAGE_GUIDE.md)**: 完全な使用方法とトラブルシューティング

## モジュール構成

### 1. Enhanced XML Parser (`enhanced_parser.py`)
XMLログを解析し、拡張された情報を抽出

```python
from src.preprocessing import EnhancedXMLParser

parser = EnhancedXMLParser()
games = parser.parse_directory('data/xml_logs')
```

### 2. Comprehensive Game State Tracker (`game_state_tracker.py`)
各打牌タイミングでの完全なゲーム状態を構築

```python
from src.preprocessing import ComprehensiveGameStateTracker

tracker = ComprehensiveGameStateTracker()
states = tracker.extract_all_states(game)
```

### 3. Advanced Feature Encoder V2 (`feature_encoder_v2.py`)
固定長ベクトル（2,099次元）への変換

```python
from src.preprocessing import AdvancedFeatureEncoderV2

encoder = AdvancedFeatureEncoderV2(
    draw_history_length=8,
    discard_history_length=32
)
features = encoder.encode_state(state)
```

### 4. Data Validator (`data_validator.py`)
データ整合性の検証

```python
from src.preprocessing import DataValidator

validator = DataValidator()
report = validator.validate_parsed_game(game)
```

### 5. Comprehensive Dataset Builder (`dataset_builder_v2.py`)
全てを統合したデータセット構築

```python
from src.preprocessing import ComprehensiveDatasetBuilder

builder = ComprehensiveDatasetBuilder(
    xml_dir='data/xml_logs',
    output_dir='data/processed_v2'
)
stats = builder.build_complete_dataset()
```

## 出力データ構造

```
data/processed_v2/
├── train/
│   ├── X_train.npy          # [N_train, 2099] float32
│   ├── y_train.npy          # [N_train] int64 (0-33)
│   └── metadata_train.json  # ゲームID、局番号など
├── val/
│   ├── X_val.npy
│   ├── y_val.npy
│   └── metadata_val.json
├── test/
│   ├── X_test.npy
│   ├── y_test.npy
│   └── metadata_test.json
├── split_info.json          # データ分割情報
└── dataset_info.json        # 統計情報
```

## 特徴量ベクトル構成（2,099次元）

| カテゴリ | 次元数 | 説明 |
|---------|--------|------|
| **基本特徴量** | 340 | 1x34 × 10グループ |
| - 自分の手牌 | 34 | 各牌種の枚数 |
| - 自分の副露 | 34 | 鳴いた牌 |
| - 自分の捨て牌 | 34 | 自分の河 |
| - 対家1-3の捨て牌 | 102 | 3人の河 |
| - 対家1-3の副露 | 102 | 3人の鳴き |
| - ドラ表示牌 | 34 | ドラ |
| **時系列特徴量** | 1,520 | |
| - ツモ履歴 | 272 | 8 × 34 |
| - 捨て牌履歴 | 1,248 | 32 × 39 |
| **メタ特徴量** | 31 | 局情報・点数・リーチ |
| **待ち・向聴数** | 38 | テンパイ・待ち牌 |
| **捨て牌メタ** | 170 | 手出し比率・危険度 |
| **合計** | **2,099** | |

## 設定オプション

### 基本設定

```python
config = {
    'draw_history_length': 8,       # ツモ履歴の長さ
    'discard_history_length': 32,   # 捨て牌履歴の長さ
    'enable_shanten_calc': True,    # 向聴数計算
    'enable_danger_estimation': True, # 危険度推定
    'train_ratio': 0.80,
    'val_ratio': 0.10,
    'test_ratio': 0.10,
    'random_seed': 42,
}
```

### 特徴量次元数の調整

| 設定 | draw | discard | 総次元数 |
|-----|------|---------|----------|
| 軽量 | 4 | 16 | 1,395 |
| 標準 | 8 | 32 | 2,099 |
| 高精度 | 16 | 64 | 3,507 |

## テスト

```bash
# 全モジュールのテスト
python scripts/test_data_pipeline.py

# 小規模テスト（1ゲーム）
python scripts/build_comprehensive_dataset.py --max-games 1
```

## パフォーマンス

### 処理時間（8コアCPU、32GB RAM）

| ゲーム数 | 処理時間 |
|---------|---------|
| 100 | 1-2分 |
| 1,000 | 10-20分 |
| 10,000 | 2-4時間 |
| 100,000 | 20-40時間 |

### メモリ使用量

| ゲーム数 | メモリ |
|---------|--------|
| 1,000 | ~400 MB |
| 10,000 | ~4 GB |
| 100,000 | ~40 GB |

## トラブルシューティング

### メモリ不足

```bash
# ゲーム数を減らす
python scripts/build_comprehensive_dataset.py --max-games 10000

# 履歴長を減らす（特徴量次元数を削減）
python scripts/build_comprehensive_dataset.py \
    --draw-history 4 \
    --discard-history 16
```

### XMLファイルが見つからない

```bash
# パスを確認
ls data/xml_logs/*.xml

# カスタムパス指定
python scripts/build_comprehensive_dataset.py \
    --xml-dir /path/to/xml_logs
```

## 使用例

### 基本的な使用

```python
from src.preprocessing import ComprehensiveDatasetBuilder

builder = ComprehensiveDatasetBuilder(
    xml_dir='data/xml_logs',
    output_dir='data/processed_v2'
)

# データセット構築
stats = builder.build_complete_dataset(
    max_games=10000,
    validate=True,
    show_progress=True
)

# データ読み込み
X_train, y_train, metadata = builder.load_dataset('train')
X_val, y_val, _ = builder.load_dataset('val')
X_test, y_test, _ = builder.load_dataset('test')
```

### PyTorch DataLoaderとの統合

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Dataset作成
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

# DataLoader作成
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 学習ループ
for batch_X, batch_y in train_loader:
    # ... your training code ...
    pass
```

## 今後の拡張

- [ ] マルチプロセス並列化
- [ ] C++での向聴数計算（高速化）
- [ ] オンザフライ特徴量生成
- [ ] データ圧縮（量子化）
- [ ] 追加特徴量（和了確率など）

## ライセンス

MIT License

## 参考文献

- [天鳳ログフォーマット](https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format)
- Tjong et al. "Deep Learning for Mahjong AI"
- Yuan et al. "Tokens-to-Token ViT"

## 連絡先

問題や質問がある場合は、GitHubのIssueを作成してください。

---

**バージョン:** 2.0  
**最終更新:** 2025-10-26  
**作成者:** Mahjong AI Team

