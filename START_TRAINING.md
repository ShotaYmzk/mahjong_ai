# トレーニング開始ガイド

## 🚀 クイックスタート

### ステップ1: 環境確認
```bash
cd /home/ubuntu/Documents/mahjong_ai

# インストール確認
python3 test_installation.py

# パラメータ確認
python3 check_model_params.py
```

期待される出力:
```
✓ All modules imported successfully
✓ XML parser working
✓ Model: 15,415,938 params (102.8% of target)
```

### ステップ2: データ準備
```bash
# XMLファイルをdata/raw/にコピー
# 例: Tenhouのゲームログ
cp /path/to/tenhou/logs/*.xml data/raw/

# ファイル数確認
ls -1 data/raw/*.xml | wc -l
```

推奨データ量:
- **最小**: 100ゲーム (テスト用)
- **推奨**: 10,000ゲーム
- **理想**: 100,000ゲーム以上

### ステップ3: トレーニング開始
```bash
# デフォルト設定で開始
bash run_train.sh

# または、Pythonから直接
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from src.preprocessing import MahjongDatasetBuilder
from src.model import TIT, SimplifiedDiscardOnlyHead, CompleteMahjongModel
from src.training import SupervisedTrainer, create_optimizer, create_scheduler
from src.utils import setup_logger, set_seed, get_device, load_config

# Setup
set_seed(42)
device = get_device()
logger = setup_logger('training', log_file='outputs/logs/train.log')

# Load configs
train_config = load_config('configs/train_config.yaml')
model_config = load_config('configs/model_config.yaml')

# Build dataset
builder = MahjongDatasetBuilder('data/raw', 'data/processed')
if not Path('data/processed/X.npy').exists():
    builder.build_dataset(max_games=train_config['data'].get('max_games'))

# Create dataloaders
dataloaders = builder.create_dataloaders(
    batch_size=train_config['data']['batch_size'],
    num_workers=train_config['data']['num_workers']
)

# Build model
backbone = TIT(
    input_dim=model_config['input']['input_dim'],
    d_model=model_config['tit']['d_model'],
    nhead_inner=model_config['tit']['nhead_inner'],
    nhead_outer=model_config['tit']['nhead_outer'],
    dim_feedforward=model_config['tit']['dim_feedforward'],
    dropout=model_config['tit']['dropout'],
    num_inner_layers=model_config['tit']['num_inner_layers'],
    num_outer_layers=model_config['tit']['num_outer_layers']
)

head = SimplifiedDiscardOnlyHead(
    d_model=model_config['tit']['d_model'],
    num_tiles=model_config['input']['num_tile_types'],
    dropout=model_config['tit']['dropout']
)

model = CompleteMahjongModel(backbone, head)

# Training
optimizer = create_optimizer(
    model,
    optimizer_name=train_config['training']['optimizer'],
    learning_rate=train_config['training']['learning_rate'],
    weight_decay=train_config['training']['weight_decay']
)

scheduler = create_scheduler(
    optimizer,
    scheduler_name=train_config['training']['scheduler'],
    **train_config['training']['scheduler_params']
)

trainer = SupervisedTrainer(
    model=model,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    optimizer=optimizer,
    device=device
)

print('Starting training...')
trainer.train(num_epochs=train_config['training']['num_epochs'], scheduler=scheduler)
print(f'Training completed! Best accuracy: {trainer.best_val_acc:.4f}')
"
```

## 📊 トレーニングモニタリング

### リアルタイムログ監視
```bash
# ログをリアルタイム表示
tail -f outputs/logs/train.log

# 学習曲線を確認 (別ターミナル)
watch -n 10 'grep "Epoch" outputs/logs/train.log | tail -5'
```

### GPU使用状況
```bash
# GPU使用率を監視
watch -n 1 nvidia-smi
```

期待される使用量:
- **VRAM**: ~2-3GB (Batch 512)
- **GPU使用率**: 80-95%
- **Temperature**: <80℃

### チェックポイント確認
```bash
# 保存されたモデルを確認
ls -lht outputs/checkpoints/

# 最高精度モデル
ls -lh outputs/checkpoints/best_acc.pth
```

## 📈 期待される学習曲線

### Epoch 1-20 (初期学習)
```
Epoch   Train Loss   Train Acc   Val Loss   Val Acc
-----   ----------   ---------   --------   -------
1       3.526        0.098       3.489      0.102
5       3.234        0.156       3.198      0.162
10      3.012        0.201       2.987      0.208
20      2.856        0.254       2.843      0.261
```

### Epoch 21-50 (安定期)
```
Epoch   Train Loss   Train Acc   Val Loss   Val Acc
-----   ----------   ---------   --------   -------
30      2.745        0.289       2.738      0.294
40      2.668        0.312       2.665      0.316
50      2.612        0.329       2.611      0.331
```

### Epoch 51-100 (収束期)
```
Epoch   Train Loss   Train Acc   Val Loss   Val Acc
-----   ----------   ---------   --------   -------
60      2.571        0.341       2.573      0.343
80      2.538        0.352       2.545      0.349
100     2.518        0.358       2.531      0.353
```

### Epoch 101-150 (微調整)
```
Epoch   Train Loss   Train Acc   Val Loss   Val Acc
-----   ----------   ---------   --------   -------
120     2.505        0.362       2.522      0.356
140     2.498        0.364       2.518      0.357
150     2.494        0.365       2.516      0.358
```

## 🎯 評価

### トレーニング後の評価
```bash
python3 <<EOF
from src.evaluation import MetricsCalculator
from src.preprocessing import MahjongDatasetBuilder
from src.model import CompleteMahjongModel
import torch

# Load model
model = CompleteMahjongModel.load('outputs/checkpoints/best_acc.pth')
model.eval()

# Load test data
builder = MahjongDatasetBuilder('data/raw', 'data/processed')
dataloaders = builder.create_dataloaders(batch_size=256)
test_loader = dataloaders['test']

# Evaluate
metrics_calc = MetricsCalculator(num_classes=34)

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs, _ = model(inputs)
        if isinstance(outputs, dict):
            outputs = outputs['discard']
        metrics_calc.update(outputs, targets)

results = metrics_calc.compute()
print("\n【テスト結果】")
print(f"Accuracy:        {results['accuracy']:.4f}")
print(f"Top-3 Accuracy:  {results['top_3_accuracy']:.4f}")
print(f"Top-5 Accuracy:  {results['top_5_accuracy']:.4f}")
print(f"Success Prob:    {results['success_probability']:.4f}")
print(f"Hit Rate (k=5):  {results['hit_rate_top5']:.4f}")
print(f"Precision:       {results['precision_macro']:.4f}")
print(f"Recall:          {results['recall_macro']:.4f}")
print(f"F1 Score:        {results['f1_macro']:.4f}")
EOF
```

## 🔧 トラブルシューティング

### 問題: CUDA Out of Memory
**解決策**:
```yaml
# configs/train_config.yaml
data:
  batch_size: 256  # 512から削減
```

### 問題: 学習が進まない (Accuracy < 15%)
**解決策**:
1. Learning rateを下げる: `1e-4` → `5e-5`
2. Warmup追加を検討
3. データの質を確認

### 問題: Overfitting (Train >> Val)
**解決策**:
```yaml
# configs/model_config.yaml
tit:
  dropout: 0.2  # 0.1から増加
```

### 問題: 学習が遅い
**解決策**:
```yaml
# configs/train_config.yaml
data:
  num_workers: 8  # データローダーの並列数
```

```bash
# または、Mixed Precision Training
# configs/train_config.yaml
mixed_precision: true
```

## 📝 ログの見方

### 正常な学習の例
```
2025-XX-XX 10:00:00 - training - INFO - Starting training for 150 epochs
2025-XX-XX 10:00:05 - training - INFO - Epoch 1/150 - Train Loss: 3.526, Train Acc: 0.0980, Val Loss: 3.489, Val Acc: 0.1020
2025-XX-XX 10:00:10 - training - INFO - Saved best accuracy checkpoint: 0.1020
2025-XX-XX 10:02:15 - training - INFO - Epoch 2/150 - Train Loss: 3.234, Train Acc: 0.1560, Val Loss: 3.198, Val Acc: 0.1620
2025-XX-XX 10:02:20 - training - INFO - Saved best accuracy checkpoint: 0.1620
```

### 問題がある学習の例
```
# Loss が増加 → Learning rate が高すぎる
Epoch 10/150 - Train Loss: 3.012, Val Loss: 2.987
Epoch 11/150 - Train Loss: 3.245, Val Loss: 3.156  # ⚠ Loss増加

# Accuracy が停滞 → データ不足 or モデル容量不足
Epoch 50/150 - Train Acc: 0.329, Val Acc: 0.331
Epoch 60/150 - Train Acc: 0.330, Val Acc: 0.332
Epoch 70/150 - Train Acc: 0.331, Val Acc: 0.332  # ⚠ 停滞
```

## 🎓 次のステップ

### 1. 教師あり学習完了後
```bash
# モデルをロードして強化学習へ
# (RL trainer実装後に使用)
```

### 2. 注意機構の可視化
```python
from src.evaluation import AttentionVisualizer
from src.model import XAIHooks

# Attention可視化
xai_hooks = XAIHooks(model)
xai_hooks.register_all_attention_hooks()

# Forward pass
outputs, attention = model(test_input)
attention_weights = xai_hooks.get_attention_weights()

# 可視化
visualizer = AttentionVisualizer()
visualizer.plot_attention_heatmap(
    attention_weights['outer_transformer.layers.0'],
    save_name='attention_layer0.png'
)
```

### 3. モデルのデプロイ
```python
# モデルを保存
torch.save(model.state_dict(), 'final_model.pth')

# 推論用
model.eval()
with torch.no_grad():
    prediction = model(game_state)
```

---

**準備完了！トレーニングを開始してください！** 🚀

