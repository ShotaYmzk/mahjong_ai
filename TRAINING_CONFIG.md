# トレーニング設定サマリー

## ✅ 最終確認済み設定

### モデルアーキテクチャ

| 項目 | 値 | 備考 |
|-----|-----|-----|
| **バックボーン** | TIT (Transformer-in-Transformer) | 階層的注意機構 |
| **Inner Transformer層数** | 3層 | タイルグループ内処理 |
| **Outer Transformer層数** | 3層 | グループ間関係処理 |
| **d_model** | 448 | 隠れ層次元数 |
| **nhead (Inner)** | 8ヘッド | Inner attention heads |
| **nhead (Outer)** | 8ヘッド | Outer attention heads |
| **dim_feedforward** | 1792 | FFN次元数 (4 × d_model) |
| **パラメータ数** | **15,415,938** | 目標15Mの102.8% ✓ |
| **推定VRAM使用量** | 242 MB | Batch 512での推定 |

### トレーニングハイパーパラメータ

| 項目 | 値 | 備考 |
|-----|-----|-----|
| **Batch size** | 512 | VRAM 12GB想定 |
| **Epochs** | 150 | 早期終了あり (patience=20) |
| **Learning rate** | 1e-4 | Adam/AdamWベース |
| **Optimizer** | AdamW | Weight decay: 0.01 |
| **Scheduler** | Cosine Annealing | T_max=150, eta_min=1e-6 |
| **Gradient clipping** | 1.0 | Max norm |
| **Dropout** | 0.1 | 正則化 |
| **Sequence length** | 4 | 直近4手分のメモリ |

### データ設定

| 項目 | 値 |
|-----|-----|
| **Train/Val/Test比率** | 70% / 15% / 15% |
| **入力次元数** | 340 (34 tiles × 10 groups) |
| **出力クラス数** | 34 (tile types) |
| **損失関数** | CrossEntropyLoss |

### 強化学習 (PPO)

| 項目 | 値 | 備考 |
|-----|-----|-----|
| **Policy clip** | 0.2 | PPO clipping parameter |
| **Value clip** | 0.3 | Value function clipping |
| **γ (Discount)** | 0.99 | 割引率 |
| **GAE λ** | 0.95 | Generalized Advantage Estimation |
| **報酬設計** | Fan Backward | 和了時スコア逆伝播 |

### 評価指標

| 指標 | 説明 | 目標 |
|-----|-----|-----|
| **Accuracy** | 打牌の完全一致率 | >30% |
| **Top-3 Accuracy** | 上位3手に含まれる率 | >60% |
| **Top-5 Accuracy** | 上位5手に含まれる率 | >75% |
| **SP (Success Probability)** | 正解牌への確率割り当て平均 | >0.15 |
| **HR (Hit Rate)** | トップk内ヒット率 | >0.70 (k=5) |
| **Per-class F1** | クラス別F1スコア | >0.25 |

## 実行方法

### 1. パラメータ数の確認
```bash
python3 check_model_params.py
```

### 2. インストール確認
```bash
python3 test_installation.py
```

### 3. データ準備
```bash
# XMLファイルをdata/raw/に配置
cp /path/to/xml/*.xml data/raw/
```

### 4. トレーニング開始
```bash
# デフォルト設定で実行
bash run_train.sh

# カスタムパラメータで実行
bash run_train.sh 150 512 0.0001  # epochs batch_size learning_rate
```

### 5. トレーニングのモニタリング
```bash
# ログを確認
tail -f outputs/logs/train.log

# チェックポイントを確認
ls -lh outputs/checkpoints/
```

## 設定ファイル詳細

### configs/train_config.yaml
```yaml
# メインのトレーニング設定
data:
  batch_size: 512
  sequence_length: 4
  
training:
  num_epochs: 150
  learning_rate: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  scheduler_params:
    T_max: 150
    eta_min: 1e-6
```

### configs/model_config.yaml
```yaml
# モデルアーキテクチャ設定
tit:
  d_model: 448
  nhead_inner: 8
  nhead_outer: 8
  num_inner_layers: 3
  num_outer_layers: 3
  dim_feedforward: 1792
  dropout: 0.1
  
sequence:
  max_sequence_length: 4
  use_temporal_context: true
```

### configs/reward_config.yaml
```yaml
# 強化学習の報酬設定
ppo:
  policy_clip: 0.2
  value_clip: 0.3
  gamma: 0.99
  gae_lambda: 0.95
  
fan_backward:
  win_reward_multiplier: 1.0
  shanten_improvement_reward: 0.1
  discount_factor: 0.99
```

## 期待される結果

### フェーズ1: 教師あり学習 (Epochs 0-100)
- **Train Accuracy**: 25% → 35%
- **Val Accuracy**: 22% → 32%
- **Loss**: 3.5 → 2.8
- **学習時間**: 約3-5時間 (GPU依存)

### フェーズ2: ファインチューニング (Epochs 100-150)
- **Val Accuracy**: 32% → 38%
- **Top-5 HR**: 65% → 78%
- **収束**: Epoch 120-140で早期終了の可能性

### メモリ使用量
- **モデルパラメータ**: 59 MB
- **オプティマイザ状態**: 235 MB
- **Batch処理**: 242 MB (total)
- **VRAM余裕**: 12GB - 0.24GB = **11.76GB空き**

## トラブルシューティング

### CUDA Out of Memory
```bash
# Batch sizeを減らす
# configs/train_config.yaml
data:
  batch_size: 256  # 512 → 256
```

### 学習が遅い
```bash
# Workers数を増やす
data:
  num_workers: 8  # 4 → 8
```

### Accuracyが上がらない
```yaml
# Learning rateを調整
training:
  learning_rate: 5e-5  # 1e-4 → 5e-5
```

## チェックリスト

- [x] パラメータ数が15M付近 (✓ 15.4M)
- [x] Inner/Outer層数が3層
- [x] Batch size 512設定
- [x] Cosine scheduler設定
- [x] PPO parameters設定
- [x] Sequence length 4設定
- [x] 評価指標の確認
- [ ] データの準備 (XMLファイル)
- [ ] トレーニング実行

## 追加情報

### 学習曲線の保存
トレーニング中、以下が自動保存されます：
- `outputs/checkpoints/best_acc.pth` - 最高精度モデル
- `outputs/checkpoints/best_loss.pth` - 最小損失モデル
- `outputs/logs/training_history.json` - 学習履歴
- `outputs/visualizations/` - 可視化画像

### モデルの評価
```python
from src.evaluation import MetricsCalculator

metrics_calc = MetricsCalculator(num_classes=34)
metrics_calc.update(predictions, targets)
results = metrics_calc.compute()

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Top-3 Acc: {results['top_3_accuracy']:.4f}")
print(f"Top-5 Acc: {results['top_5_accuracy']:.4f}")
print(f"SP: {results['success_probability']:.4f}")
print(f"HR@5: {results['hit_rate_top5']:.4f}")
```

---

**最終更新**: 2025-01-XX  
**ステータス**: ✅ 設定完了・実行準備完了

