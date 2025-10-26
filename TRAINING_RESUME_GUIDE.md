# 学習の再開機能と可視化ガイド

## 🎯 新機能

### 1. 各エポックで学習曲線の画像を自動生成 📊
- Loss曲線（訓練・検証）
- Accuracy曲線（訓練・検証）
- Learning Rate変化

### 2. 学習の中断・再開機能 💾
- 各エポック後に自動的にチェックポイント保存
- 中断した場合でも続きから再開可能
- 学習率、optimizer状態、全履歴を保存

### 3. 詳細なデータ保存 📈
- accuracy（訓練・検証）
- loss（訓練・検証）
- 学習率の履歴
- ベストモデルの自動保存

---

## 🚀 使用方法

### 基本的な学習

```bash
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50
```

### 学習を中断した場合

学習中に `Ctrl+C` を押すと、中断されます。最後のエポックまでのデータは保存されています。

### 学習を再開する方法

#### 方法1: 自動再開（推奨）

```bash
# 同じコマンドを再実行すると、自動的に検出されます
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50

# 確認メッセージが表示されます:
# "前回の学習を検出しました: outputs/demo/checkpoints/latest.pth"
# "前回の学習から再開しますか？ (y/N): "
# → "y" を入力
```

#### 方法2: 明示的に指定

```bash
# 特定のチェックポイントから再開
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50 \
    --resume outputs/demo/checkpoints/latest.pth
```

#### 方法3: ベストモデルから再開

```bash
# 最良の精度のモデルから再開
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 100 \
    --resume outputs/demo/checkpoints/best_acc.pth
```

---

## 📁 保存されるファイル

### チェックポイント

```
outputs/demo/checkpoints/
├── latest.pth                    # 最新のエポック（再開用）
├── best_acc.pth                  # 最高精度のモデル
├── best_loss.pth                 # 最小lossのモデル
├── checkpoint_epoch_10.pth       # 10エポックごと
├── checkpoint_epoch_20.pth
└── ...
```

**各チェックポイントに含まれる情報:**
- エポック番号
- モデルの重み
- optimizerの状態
- schedulerの状態
- 訓練履歴（loss、accuracy）
- 検証履歴（loss、accuracy）
- 学習率履歴
- 最良スコア（best_val_acc、best_val_loss）

### 学習曲線の画像

```
outputs/demo/logs/plots/
├── training_curves_latest.png        # 最新の曲線
├── training_curves_epoch_0001.png    # エポック1
├── training_curves_epoch_0002.png    # エポック2
└── ...
```

**画像の内容:**
- 左: Loss曲線（訓練・検証）
- 中央: Accuracy曲線（訓練・検証）
- 右: Learning Rate変化

### 学習履歴JSON

```
outputs/demo/logs/training_history.json
```

**内容:**
```json
{
  "train": [
    {"loss": 2.5, "accuracy": 0.35},
    {"loss": 2.3, "accuracy": 0.38},
    ...
  ],
  "val": [
    {"loss": 2.6, "accuracy": 0.34},
    {"loss": 2.4, "accuracy": 0.36},
    ...
  ],
  "learning_rate": [0.0001, 0.0001, ...],
  "best_val_acc": 0.42,
  "best_val_loss": 2.1
}
```

---

## 💡 実行例

### 例1: 通常の学習（50エポック）

```bash
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 1e-4
```

**出力:**
- 各エポック終了時に画像を生成
- 10エポックごとにチェックポイントを保存
- ベストモデルを自動保存

### 例2: 学習が中断された場合

```bash
# エポック25で中断（Ctrl+C）
# ... 処理中 ...
# ^C
# ⚠️  学習が中断されました

# 再開
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50

# 出力例:
# "前回の学習を検出しました: outputs/demo/checkpoints/latest.pth"
# "前回の学習から再開しますか？ (y/N): y"
# "✅ Loaded checkpoint from epoch 25"
# "   Best val acc: 0.3850"
# "   Best val loss: 2.2500"
# "   Resuming from epoch 26"
```

### 例3: さらに長く学習

```bash
# 50エポック完了後、さらに50エポック追加

python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 100 \
    --resume outputs/demo/checkpoints/latest.pth
```

### 例4: 学習率を変更して再開

```bash
# ベストモデルから、学習率を下げて再開
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo_finetune \
    --epochs 80 \
    --resume outputs/demo/checkpoints/best_acc.pth \
    --learning-rate 5e-5
```

---

## 📊 学習曲線の確認

### リアルタイムで確認

```bash
# 別のターミナルで
watch -n 5 "ls -lh outputs/demo/logs/plots/training_curves_latest.png"

# または画像ビューアで開く
eog outputs/demo/logs/plots/training_curves_latest.png
# または
xdg-open outputs/demo/logs/plots/training_curves_latest.png
```

### 履歴JSONを確認

```bash
# 最新のメトリクスを表示
python -c "
import json
with open('outputs/demo/logs/training_history.json', 'r') as f:
    history = json.load(f)
    print(f'エポック数: {len(history[\"train\"])}')
    print(f'最新Train Acc: {history[\"train\"][-1][\"accuracy\"]:.4f}')
    print(f'最新Val Acc: {history[\"val\"][-1][\"accuracy\"]:.4f}')
    print(f'ベストVal Acc: {history[\"best_val_acc\"]:.4f}')
    print(f'ベストVal Loss: {history[\"best_val_loss\"]:.4f}')
"
```

---

## 🔧 トラブルシューティング

### Q1: 画像が生成されない

**A:** matplotlibがインストールされているか確認:
```bash
pip install matplotlib
```

### Q2: チェックポイントから再開できない

**A:** チェックポイントファイルが存在するか確認:
```bash
ls -lh outputs/demo/checkpoints/
```

エラーメッセージを確認:
```bash
cat outputs/demo/train.log | tail -20
```

### Q3: 学習曲線が表示されない環境

**A:** Non-interactiveバックエンドを使用しているため、画像ファイルとして保存されます。
```bash
# 画像ファイルを確認
ls outputs/demo/logs/plots/

# 別のマシンにコピーして表示
scp user@server:outputs/demo/logs/plots/training_curves_latest.png ./
```

### Q4: メモリ不足で中断された

**A:** バッチサイズを減らして再開:
```bash
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50 \
    --resume outputs/demo/checkpoints/latest.pth \
    --batch-size 128
```

---

## 📈 学習の進捗確認

### ログファイルをリアルタイムで監視

```bash
# 別のターミナルで
tail -f outputs/demo/train.log

# または、エポック情報だけを抽出
tail -f outputs/demo/train.log | grep "Epoch"
```

### 現在のステータスを確認

```bash
python -c "
import torch
checkpoint = torch.load('outputs/demo/checkpoints/latest.pth', map_location='cpu')
print(f'現在のエポック: {checkpoint[\"epoch\"]}')
print(f'ベストVal Acc: {checkpoint[\"best_val_acc\"]:.4f}')
print(f'ベストVal Loss: {checkpoint[\"best_val_loss\"]:.4f}')
print(f'総エポック数: {len(checkpoint[\"train_history\"])}')
"
```

---

## 🎯 ベストプラクティス

### 1. 定期的なチェックポイント保存

```bash
# 5エポックごとに保存
python scripts/train_demo.py \
    --save-every 5 \
    --epochs 50
```

### 2. 複数の実験を並行実行

```bash
# 実験1: 標準設定
python scripts/train_demo.py \
    --output-dir outputs/exp1_standard

# 実験2: 大きいモデル
python scripts/train_demo.py \
    --output-dir outputs/exp2_large \
    --d-model 512 \
    --num-layers 8

# 実験3: 低学習率
python scripts/train_demo.py \
    --output-dir outputs/exp3_lowlr \
    --learning-rate 5e-5
```

### 3. 長時間学習のバックグラウンド実行

```bash
# nohupで実行
nohup python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 100 \
    > train.out 2>&1 &

# プロセスIDを確認
echo $!

# ログを監視
tail -f train.out
```

### 4. GPU使用率の監視

```bash
# 別のターミナルで
watch -n 1 nvidia-smi
```

---

## 📝 まとめ

### 新機能のメリット

✅ **学習曲線の可視化**
- 各エポックで自動的に画像生成
- 学習の進捗を視覚的に確認可能

✅ **学習の再開機能**
- 中断しても続きから再開
- 学習率、optimizer状態を完全に復元

✅ **データの完全保存**
- accuracy、loss、学習率の全履歴
- ベストモデルの自動保存

### 基本的な使い方

```bash
# 1. 学習開始
python scripts/train_demo.py --data-dir data/processed_demo --epochs 50

# 2. 中断された場合（Ctrl+C）

# 3. 再開（自動検出）
python scripts/train_demo.py --data-dir data/processed_demo --epochs 50
# → "y" を入力

# 4. 学習曲線を確認
xdg-open outputs/demo/logs/plots/training_curves_latest.png
```

---

**Happy Training! 🚀🀄**

