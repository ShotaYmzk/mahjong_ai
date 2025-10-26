# 麻雀AI - 教師あり学習専用設定

## 概要

このプロジェクトは**教師あり学習のみ**で構成されています。
エキスパートの打牌データから直接学習します。

## システム構成

### 削除された機能
- ❌ 強化学習 (PPO)
- ❌ Fan Backward報酬システム
- ❌ 価値関数推定
- ❌ ポリシー勾配

### 実装されている機能
- ✅ 教師あり学習（クロスエントロピー損失）
- ✅ TIT (Transformer-in-Transformer) アーキテクチャ
- ✅ 打牌履歴の時系列処理
- ✅ シャンテン数計算
- ✅ 注意機構の可視化

## トレーニング設定

### モデル
- **バックボーン**: TIT (Transformer-in-Transformer)
- **パラメータ数**: 15.4M
- **Inner層**: 3層
- **Outer層**: 3層
- **d_model**: 448

### ハイパーパラメータ
- **Batch size**: 512
- **Learning rate**: 1e-4
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing
- **Epochs**: 150
- **Early stopping**: Patience 20

### 特徴量
1. **自分の手牌** (34次元)
2. **自分の副露** (34次元)
3. **全プレイヤーの捨て牌** (4×34次元)
4. **相手の副露** (3×34次元)
5. **ドラ表示牌** (34次元)
6. **打牌履歴** (64手 × 39次元)
   - 打牌タイル (34次元)
   - プレイヤーID (4次元)
   - リーチフラグ (1次元)

合計特徴量次元: 340 + 64×39 = **2,836次元**

## データの流れ

```
XML game logs
    ↓
Parse & Extract
    ↓
Feature Encoding (with discard history)
    ↓
TensorDataset
    ↓
DataLoader (batch=512)
    ↓
TIT Model
    ↓
Discard Prediction (34 classes)
    ↓
CrossEntropy Loss
    ↓
Backpropagation
```

## 学習プロセス

### Phase 1: データ準備 (1回のみ)
```bash
# XMLファイルを配置
cp /path/to/*.xml data/raw/

# データセット構築
python3 -c "
from src.preprocessing import MahjongDatasetBuilder
builder = MahjongDatasetBuilder('data/raw', 'data/processed')
builder.build_dataset()
"
```

### Phase 2: トレーニング
```bash
# トレーニング開始
bash run_train.sh

# または
python3 -c "
from src.training import SupervisedTrainer
from src.model import TIT, SimplifiedDiscardOnlyHead, CompleteMahjongModel
from src.utils import load_config, get_device

# Setup
config = load_config('configs/train_config.yaml')
device = get_device()

# Model
model = CompleteMahjongModel(
    TIT(...),
    SimplifiedDiscardOnlyHead(...)
)

# Train
trainer = SupervisedTrainer(model, train_loader, val_loader, optimizer, device)
trainer.train(num_epochs=150)
"
```

### Phase 3: 評価
```bash
python3 -c "
from src.evaluation import MetricsCalculator

metrics = MetricsCalculator()
# ... evaluate on test set
results = metrics.compute()

print(f'Accuracy: {results[\"accuracy\"]:.4f}')
print(f'Top-3 Acc: {results[\"top_3_accuracy\"]:.4f}')
print(f'Top-5 Acc: {results[\"top_5_accuracy\"]:.4f}')
"
```

## 評価指標

### Primary Metrics
1. **Accuracy**: 完全一致率
   - 目標: >35%
   
2. **Top-3 Accuracy**: 上位3手に含まれる率
   - 目標: >60%
   
3. **Top-5 Accuracy**: 上位5手に含まれる率
   - 目標: >75%

### Secondary Metrics
4. **Success Probability (SP)**: 正解牌への確率割り当て平均
   - 目標: >0.15

5. **Hit Rate (HR)**: トップk内のヒット率
   - 目標: HR@5 > 0.70

6. **Shanten Improvement Rate**: シャンテン数改善率
   - 正解打牌 vs AI打牌のシャンテン数比較

## 打牌履歴の活用

### 全員の打牌を保持
```python
from src.preprocessing import GameStateManager

manager = GameStateManager()
manager.start_new_round(round_num=0, dealer=0)

# 打牌を追加
manager.add_discard(player_id=0, tile=120, turn=1)
manager.add_discard(player_id=1, tile=45, turn=2)
manager.add_discard(player_id=2, tile=67, turn=3)

# 履歴取得
state = manager.get_current_state()
recent_discards = state.get_recent_discards(num_turns=4)
all_discards = state.discard_history
```

### 時系列エンコーディング
```python
# 打牌履歴をエンコード (64手 × 39次元)
history_seq = state.encode_discard_history_sequence(max_length=64)

# shape: (64, 39)
# - [:, :34]: 打牌タイル (one-hot)
# - [:, 34:38]: プレイヤーID (one-hot)
# - [:, 38]: リーチフラグ
```

## シャンテン数の活用

### 基本的な使用
```python
from src.utils import calculate_shanten, tiles_list_to_34_array

# 手牌をシャンテン数計算
tiles_34 = tiles_list_to_34_array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
shanten = calculate_shanten(tiles_34)

print(f"Shanten: {shanten}")
# -1: 和了
#  0: 聴牌
#  1+: N向聴
```

### 詳細分析
```python
from src.utils import analyze_hand_details, print_hand_analysis

# 手牌文字列から分析
analysis = analyze_hand_details("123m456p789s1122z")
print_hand_analysis(analysis)

# 出力:
# 一般形: 1向聴
# 七対子: 2向聴
# 国士無双: 8向聴
# ---
# 打1z (聴牌) 摸[1m2m3m 12枚]
# 打2z (聴牌) 摸[1m2m3m 12枚]
# ...
```

## ファイル構成（教師あり学習のみ）

```
mahjong_ai/
├── src/
│   ├── preprocessing/
│   │   ├── parse_xml.py          ✅ XML解析
│   │   ├── feature_encoder.py    ✅ 特徴量エンコーディング
│   │   ├── dataset_builder.py    ✅ データセット構築
│   │   └── game_state.py         ✅ 打牌履歴管理
│   │
│   ├── model/
│   │   ├── transformer_tit.py    ✅ TIT実装
│   │   ├── hierarchical_head.py  ✅ 打牌予測ヘッド
│   │   └── xai_hooks.py          ✅ XAI機能
│   │
│   ├── training/
│   │   └── train_supervised.py   ✅ 教師あり学習のみ
│   │
│   ├── evaluation/
│   │   ├── metrics.py            ✅ 評価指標
│   │   └── visualize_attention.py ✅ 可視化
│   │
│   └── utils/
│       ├── shanten.py            ✅ シャンテン数計算
│       ├── logger.py
│       ├── config_loader.py
│       └── seed.py
│
├── configs/
│   ├── train_config.yaml         ✅ トレーニング設定
│   └── model_config.yaml         ✅ モデル設定
│
├── run_train.sh                  ✅ トレーニングスクリプト
├── check_model_params.py         ✅ パラメータ確認
└── test_installation.py          ✅ インストール確認
```

## よくある質問

### Q: なぜ強化学習を使わないのか？
A: 教師あり学習の方がシンプルで安定しており、十分な精度が得られます。
   強化学習は実装が複雑で、報酬設計が難しく、学習が不安定になりがちです。

### Q: データ量はどのくらい必要？
A: 
- 最小: 1,000ゲーム（テスト用）
- 推奨: 10,000ゲーム
- 理想: 100,000ゲーム以上

### Q: 学習時間は？
A: GPU (VRAM 12GB) で約3-5時間 (10,000ゲーム、150 epochs)

### Q: CPUでも学習できる？
A: 可能ですが、非常に遅くなります（10-20倍以上）。
   GPU使用を強く推奨します。

### Q: 打牌履歴を使うメリットは？
A: 
- 相手の手牌傾向を学習
- 危険牌の推定が可能
- より文脈に基づいた判断
- 時系列パターンの学習

## トラブルシューティング

### エラー: "python-mahjong not found"
```bash
pip install mahjong
```

### エラー: "CUDA Out of Memory"
```yaml
# configs/train_config.yaml
data:
  batch_size: 256  # 512から削減
```

### 精度が上がらない
1. データ量を増やす
2. Epochsを増やす
3. Learning rateを調整
4. ラベルスムージングを試す

## 次のステップ

1. ✅ インストール確認
2. ✅ データ準備
3. ✅ トレーニング開始
4. ⏳ 評価・分析
5. ⏳ チューニング
6. ⏳ 本番デプロイ

---

**準備完了！教師あり学習のみでトレーニングを開始してください！** 🎓

