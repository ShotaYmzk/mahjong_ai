# 10,000ゲームデモ実行ガイド

このガイドでは、10,000ゲームのデータセットを使用して教師あり学習のデモを実行する手順を説明します。

## 📋 実行手順（所要時間: 約2-3時間）

### 前提条件

```bash
cd /home/ubuntu/Documents/mahjong_ai
pip install -r requirements.txt
pip install tqdm  # プログレスバー表示用
```

---

## ステップ1: XMLデータの収集（約30分）

天鳳のXMLログを10,000ゲーム収集します。

### オプションA: データ収集スクリプトを使用（推奨）

```bash
# 天鳳のHTMLアーカイブから10,000ゲーム収集
python scripts/collect_tenhou_data.py \
    --html-dir /home/ubuntu/Documents/tenhou_dataset \
    --xml-dir data/xml_logs \
    --years 2024 2023 \
    --game-type 四鳳 \
    --max-files 10000
```

**注意:** 天鳳のHTMLアーカイブが必要です。まだダウンロードしていない場合は、下記を参照：
- [天鳳牌譜アーカイブ](http://tenhou.net/sc/raw/)

### オプションB: 既存のXMLファイルを使用

既にXMLファイルがある場合は、`data/xml_logs/` に配置してください。

```bash
# 確認
ls data/xml_logs/*.xml | wc -l
# 出力: 10000 以上であればOK
```

### オプションC: サンプルデータで小規模テスト

まず1ゲームでテストする場合：

```bash
# サンプルXMLをコピー
mkdir -p data/xml_logs
cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/
```

---

## ステップ2: データセット構築（約1-2時間）

XMLログを固定長ベクトル（2,099次元）に変換します。

### 基本実行

```bash
python scripts/build_comprehensive_dataset.py \
    --xml-dir data/xml_logs \
    --output-dir data/processed_demo \
    --max-games 10000 \
    --draw-history 8 \
    --discard-history 32 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### 軽量版（メモリ節約、特徴量次元数削減）

```bash
python scripts/build_comprehensive_dataset.py \
    --xml-dir data/xml_logs \
    --output-dir data/processed_demo_light \
    --max-games 10000 \
    --draw-history 4 \
    --discard-history 16
```

**処理時間の目安:**
- 100ゲーム: 1-2分
- 1,000ゲーム: 10-20分
- 10,000ゲーム: 1-2時間

### 出力確認

処理が完了すると、以下のファイルが生成されます：

```
data/processed_demo/
├── train/
│   ├── X_train.npy          # 訓練データ特徴量
│   ├── y_train.npy          # 訓練データラベル
│   └── metadata_train.json
├── val/
│   ├── X_val.npy
│   ├── y_val.npy
│   └── metadata_val.json
├── test/
│   ├── X_test.npy
│   ├── y_test.npy
│   └── metadata_test.json
└── dataset_info.json        # 統計情報
```

統計情報を確認：

```bash
cat data/processed_demo/dataset_info.json | python -m json.tool | head -30
```

---

## ステップ3: モデルの学習（約30分-1時間）

変換したデータでTransformerモデルを学習します。

### デモ用学習スクリプトの実行

```bash
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 1e-4
```

### 学習のモニタリング

別のターミナルで学習状況を確認：

```bash
# ログを表示
tail -f outputs/demo/train.log

# TensorBoard（オプション）
tensorboard --logdir outputs/demo/tensorboard
```

### 学習完了後

学習が完了すると、以下が生成されます：

```
outputs/demo/
├── checkpoints/
│   ├── best_model.pt        # 最良モデル
│   ├── checkpoint_epoch_10.pt
│   ├── checkpoint_epoch_20.pt
│   └── ...
├── logs/
│   └── train.log
└── metrics/
    ├── train_metrics.json
    └── val_metrics.json
```

---

## クイックスタート（テスト用）

まず小規模データでテストする場合：

```bash
# ステップ1: サンプルデータを準備
mkdir -p data/xml_logs
cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/

# ステップ2: データセット構築（1ゲーム）
python scripts/build_comprehensive_dataset.py \
    --xml-dir data/xml_logs \
    --output-dir data/processed_test \
    --max-games 1

# ステップ3: 学習（テストモード）
python scripts/train_demo.py \
    --data-dir data/processed_test \
    --output-dir outputs/test \
    --epochs 5 \
    --batch-size 32
```

---

## トラブルシューティング

### 1. XMLファイルが見つからない

```bash
# XMLファイルの確認
ls -lh data/xml_logs/*.xml | head

# ファイル数の確認
ls data/xml_logs/*.xml | wc -l
```

### 2. メモリ不足

メモリが不足する場合は、特徴量次元数を減らすか、バッチサイズを小さくします：

```bash
# 軽量設定
python scripts/build_comprehensive_dataset.py \
    --draw-history 4 \
    --discard-history 16 \
    --max-games 5000

# 学習時のバッチサイズを減らす
python scripts/train_demo.py --batch-size 128
```

### 3. 処理が遅い

```bash
# プログレスバーで進捗確認
python scripts/build_comprehensive_dataset.py --max-games 10000
# Ctrl+C で中断可能（途中まで処理したデータは保存されません）
```

### 4. CUDA（GPU）が使えない

GPUが無い場合でも学習可能ですが、時間がかかります：

```bash
# CPU での学習
python scripts/train_demo.py --device cpu --epochs 20
```

---

## 期待される結果

### データセット統計（10,000ゲーム）

- 総サンプル数: 約500,000-600,000
- 訓練データ: 約400,000-500,000
- 検証データ: 約50,000-60,000
- テストデータ: 約50,000-60,000
- 特徴量次元: 2,099（標準設定）

### 学習結果（50エポック）

- 訓練精度: 35-45%
- 検証精度: 33-42%
- Top-3精度: 60-70%
- 損失: 2.0-2.5

**注意:** 麻雀の打牌選択は高度な戦略性を持つため、完璧な予測は困難です。上級者でも一致率は40-50%程度です。

---

## 次のステップ

### 1. より大規模なデータセット

```bash
# 50,000ゲームで学習
python scripts/build_comprehensive_dataset.py --max-games 50000
python scripts/train_demo.py --epochs 100
```

### 2. ハイパーパラメータ調整

```bash
# 学習率の調整
python scripts/train_demo.py --learning-rate 5e-4

# モデルサイズの調整
python scripts/train_demo.py --d-model 512 --num-layers 8
```

### 3. 評価と分析

```bash
# テストデータでの評価
python scripts/evaluate_model.py \
    --model outputs/demo/checkpoints/best_model.pt \
    --data-dir data/processed_demo
```

---

## よくある質問

### Q1: 処理時間はどのくらい？

**A:** 8コアCPU、32GB RAMの環境での目安：
- データ収集: 30分（10,000ゲーム）
- データセット構築: 1-2時間（10,000ゲーム）
- モデル学習: 30分-1時間（50エポック、GPU使用）

### Q2: どのくらいのメモリが必要？

**A:** 10,000ゲームの場合：
- データセット構築: 4-8 GB
- モデル学習: 2-4 GB（GPU VRAM含む）

### Q3: GPUは必須？

**A:** 必須ではありませんが、GPUがあると学習が10-20倍高速になります。

### Q4: エラーが出た場合は？

**A:** ログファイルを確認してください：
```bash
# データセット構築のログ
cat dataset_building.log

# 学習のログ
cat outputs/demo/train.log
```

---

## サポート

問題が解決しない場合：
1. [使用ガイド](docs/USAGE_GUIDE.md) を確認
2. [テストスクリプト](scripts/test_data_pipeline.py) を実行
3. GitHubでIssueを作成

**Happy Training! 🚀🀄**

