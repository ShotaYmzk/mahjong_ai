# クイックスタート（10,000ゲームデモ）

10,000ゲームのデータセットで教師あり学習を実行する最も簡単な方法です。

## 🚀 3ステップで実行

### 前提条件

```bash
cd /home/ubuntu/Documents/mahjong_ai
pip install -r requirements.txt
pip install tqdm
```

---

## 方法1: 全自動実行（推奨）⚡

```bash
# オールインワンスクリプトを実行
bash scripts/run_demo_all.sh
```

これだけで以下が自動実行されます：
1. ✅ XMLデータの確認
2. ✅ データセット構築（10,000ゲーム）
3. ✅ モデルの学習（50エポック）

**処理時間:** 約2-3時間

---

## 方法2: 手動実行（ステップごと）

### ステップ1: XMLデータを準備

#### オプションA: サンプルデータでテスト（最速）

```bash
mkdir -p data/xml_logs
cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/
```

#### オプションB: 実データを収集

```bash
python scripts/collect_tenhou_data.py \
    --html-dir /home/ubuntu/Documents/tenhou_dataset \
    --xml-dir data/xml_logs \
    --max-files 10000
```

### ステップ2: データセットを構築

```bash
python scripts/build_comprehensive_dataset.py \
    --xml-dir data/xml_logs \
    --output-dir data/processed_demo \
    --max-games 10000
```

**処理時間:** 約1-2時間

### ステップ3: モデルを学習

```bash
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50
```

**処理時間:** 約30分-1時間

---

## 📊 結果の確認

### 学習ログを確認

```bash
# リアルタイムで確認
tail -f outputs/demo/train.log

# 完了後に確認
cat outputs/demo/train.log | grep "Epoch"
```

### 学習曲線の画像を確認 🎨

各エポック終了時に自動的に学習曲線の画像が生成されます！

```bash
# 最新の学習曲線を表示
xdg-open outputs/demo/logs/plots/training_curves_latest.png

# または
eog outputs/demo/logs/plots/training_curves_latest.png
```

**画像の内容:**
- 左: Loss曲線（訓練・検証）
- 中央: Accuracy曲線（訓練・検証）
- 右: Learning Rate変化

### テスト結果を確認

```bash
cat outputs/demo/metrics/test_metrics.json
```

期待される結果（10,000ゲーム、50エポック）：
```json
{
  "accuracy": 0.35-0.42,
  "top_3_accuracy": 0.60-0.70,
  "top_5_accuracy": 0.70-0.80
}
```

### データセット統計を確認

```bash
cat data/processed_demo/dataset_info.json | python -m json.tool | head -30
```

---

## 💡 トラブルシューティング

### XMLファイルが見つからない

```bash
# ファイル数を確認
ls data/xml_logs/*.xml | wc -l

# サンプルデータをコピー
mkdir -p data/xml_logs
cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/
```

### メモリ不足

軽量設定で実行：

```bash
python scripts/build_comprehensive_dataset.py \
    --draw-history 4 \
    --discard-history 16 \
    --max-games 5000

python scripts/train_demo.py \
    --batch-size 128
```

### GPUが使えない

CPUで実行：

```bash
python scripts/train_demo.py --device cpu --epochs 20
```

---

## 📈 次のステップ

### より大規模なデータで学習

```bash
# 50,000ゲーム
bash scripts/run_demo_all.sh  # スクリプト内のMAX_GAMESを変更

# または
python scripts/build_comprehensive_dataset.py --max-games 50000
python scripts/train_demo.py --epochs 100
```

### ハイパーパラメータ調整

```bash
python scripts/train_demo.py \
    --learning-rate 5e-4 \
    --d-model 512 \
    --num-layers 8 \
    --dropout 0.2
```

---

## 🎯 実行例

### 例1: サンプルデータで最速テスト（5分）

```bash
# XMLを準備
mkdir -p data/xml_logs
cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/

# データセット構築（1ゲーム）
python scripts/build_comprehensive_dataset.py \
    --xml-dir data/xml_logs \
    --output-dir data/processed_test \
    --max-games 1

# 学習（5エポック）
python scripts/train_demo.py \
    --data-dir data/processed_test \
    --output-dir outputs/test \
    --epochs 5 \
    --batch-size 32
```

### 例2: 標準的なデモ実行（2-3時間）

```bash
# 全自動
bash scripts/run_demo_all.sh
```

### 例3: カスタム設定

```bash
# データセット構築（軽量設定）
python scripts/build_comprehensive_dataset.py \
    --xml-dir data/xml_logs \
    --output-dir data/processed_custom \
    --max-games 10000 \
    --draw-history 4 \
    --discard-history 16

# 学習（カスタム設定）
python scripts/train_demo.py \
    --data-dir data/processed_custom \
    --output-dir outputs/custom \
    --epochs 30 \
    --batch-size 512 \
    --learning-rate 2e-4 \
    --d-model 128
```

---

## 🔄 学習の中断・再開

学習が途中で止まっても大丈夫！続きから再開できます。

### 学習を中断

学習中に `Ctrl+C` を押すと中断されます。最後のエポックまでのデータは自動保存されています。

### 学習を再開

```bash
# 同じコマンドを再実行
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50

# 確認メッセージが表示されます:
# "前回の学習から再開しますか？ (y/N): "
# → "y" を入力
```

または、明示的に指定：

```bash
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50 \
    --resume outputs/demo/checkpoints/latest.pth
```

**保存される情報:**
- ✅ モデルの重み
- ✅ Optimizer状態
- ✅ 学習率
- ✅ 訓練・検証の履歴（loss、accuracy）
- ✅ ベストスコア

詳細は [TRAINING_RESUME_GUIDE.md](TRAINING_RESUME_GUIDE.md) を参照。

---

## 📚 詳細ドキュメント

- **完全ガイド:** [DEMO_GUIDE.md](DEMO_GUIDE.md)
- **学習の再開:** [TRAINING_RESUME_GUIDE.md](TRAINING_RESUME_GUIDE.md) 🆕
- **使用方法:** [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
- **設計書:** [docs/DATA_CONVERSION_PIPELINE.md](docs/DATA_CONVERSION_PIPELINE.md)

---

## ⚡ 最も簡単な実行方法

```bash
# これだけでOK!
bash scripts/run_demo_all.sh
```

**処理時間:** 約2-3時間  
**結果:** `outputs/demo/` に学習済みモデル

---

**Happy Training! 🚀🀄**

