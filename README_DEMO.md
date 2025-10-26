# 10,000ゲームデモ実行方法

このガイドでは、10,000ゲームのデータセットで麻雀AIの教師あり学習を実行する方法を説明します。

## 🚀 最速スタート（2コマンド）

```bash
# 1. テストを実行（5分）
bash scripts/quick_test.sh

# 2. 本番デモを実行（2-3時間）
bash scripts/run_demo_all.sh
```

それだけです！✨

---

## 📋 実行手順の詳細

### 方法1: 全自動実行（推奨）

```bash
bash scripts/run_demo_all.sh
```

以下が自動実行されます：
1. XMLデータの確認
2. データセット構築（10,000ゲーム → 固定長ベクトル）
3. モデルの学習（Transformer、50エポック）

**所要時間:** 約2-3時間  
**必要なもの:** XMLログファイル（10,000ゲーム）

### 方法2: ステップごとに実行

#### ステップ1: XMLデータを準備

```bash
# オプションA: サンプルデータでテスト
mkdir -p data/xml_logs
cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/

# オプションB: 実データを収集（天鳳アーカイブが必要）
python scripts/collect_tenhou_data.py --max-files 10000
```

#### ステップ2: データセットを構築

```bash
python scripts/build_comprehensive_dataset.py \
    --xml-dir data/xml_logs \
    --output-dir data/processed_demo \
    --max-games 10000
```

出力: `data/processed_demo/` に固定長ベクトル（2,099次元）

#### ステップ3: モデルを学習

```bash
python scripts/train_demo.py \
    --data-dir data/processed_demo \
    --output-dir outputs/demo \
    --epochs 50
```

出力: `outputs/demo/` に学習済みモデル

---

## 📊 実行例とコマンド一覧

### クイックテスト（5分）

```bash
bash scripts/quick_test.sh
```

サンプルデータで全パイプラインをテスト。

### 標準デモ（2-3時間）

```bash
bash scripts/run_demo_all.sh
```

10,000ゲームで本格的なデモを実行。

### カスタム設定

```bash
# 軽量設定（メモリ節約）
python scripts/build_comprehensive_dataset.py \
    --draw-history 4 \
    --discard-history 16 \
    --max-games 5000

python scripts/train_demo.py \
    --batch-size 128 \
    --d-model 128 \
    --num-layers 4
```

### 大規模データ

```bash
# 50,000ゲーム
python scripts/build_comprehensive_dataset.py --max-games 50000
python scripts/train_demo.py --epochs 100
```

---

## 📈 期待される結果

### データセット（10,000ゲーム）

```
総サンプル数: 約500,000-600,000
訓練データ: 約400,000-500,000 (80%)
検証データ: 約50,000-60,000 (10%)
テストデータ: 約50,000-60,000 (10%)
特徴量次元: 2,099
```

### 学習結果（50エポック）

```
訓練精度: 35-45%
検証精度: 33-42%
Top-3精度: 60-70%
Top-5精度: 70-80%
```

**注:** 麻雀は高度な戦略ゲームで、上級者でも一致率は40-50%程度です。

---

## 🔧 出力ファイル

### データセット

```
data/processed_demo/
├── train/
│   ├── X_train.npy          # [N, 2099] float32
│   ├── y_train.npy          # [N] int64 (0-33)
│   └── metadata_train.json
├── val/
├── test/
└── dataset_info.json        # 統計情報
```

### モデル

```
outputs/demo/
├── checkpoints/
│   ├── best_model.pt        # 最良モデル
│   ├── checkpoint_epoch_10.pt
│   └── ...
├── train.log                # 学習ログ
└── metrics/
    ├── test_metrics.json    # テスト結果
    └── ...
```

---

## 💡 よくある質問

### Q: XMLデータはどこで入手？

**A:** 以下の方法があります：

1. **天鳳アーカイブをダウンロード:**
   ```bash
   # 天鳳の公式アーカイブ
   wget http://tenhou.net/sc/raw/2024.tar.bz2
   tar -xjf 2024.tar.bz2
   ```

2. **データ収集スクリプトを使用:**
   ```bash
   python scripts/collect_tenhou_data.py \
       --html-dir /path/to/tenhou_dataset \
       --max-files 10000
   ```

3. **サンプルデータでテスト:**
   ```bash
   mkdir -p data/xml_logs
   cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/
   bash scripts/quick_test.sh
   ```

### Q: 処理時間はどのくらい？

**A:** 8コアCPU、32GB RAMの環境での目安：

| ステップ | 処理時間 |
|---------|---------|
| データセット構築（10,000ゲーム） | 1-2時間 |
| モデル学習（50エポック、GPU） | 30分-1時間 |
| モデル学習（50エポック、CPU） | 5-10時間 |

### Q: メモリ不足エラーが出る

**A:** 以下を試してください：

```bash
# 軽量設定
python scripts/build_comprehensive_dataset.py \
    --draw-history 4 \
    --discard-history 16 \
    --max-games 5000

python scripts/train_demo.py \
    --batch-size 64
```

### Q: GPUは必須？

**A:** 必須ではありませんが、GPUがあると学習が10-20倍高速になります。

CPUで実行する場合：
```bash
python scripts/train_demo.py --device cpu --epochs 20
```

---

## 🎯 推奨実行フロー

### 初めての方

1. **クイックテスト（5分）**
   ```bash
   bash scripts/quick_test.sh
   ```
   → パイプラインが正常に動作することを確認

2. **小規模デモ（30分）**
   ```bash
   # 100ゲームで実行
   python scripts/build_comprehensive_dataset.py --max-games 100
   python scripts/train_demo.py --epochs 10
   ```
   → データセット構築と学習の流れを理解

3. **本番デモ（2-3時間）**
   ```bash
   bash scripts/run_demo_all.sh
   ```
   → 10,000ゲームで本格的なデモ

### 経験者の方

直接本番デモを実行：
```bash
bash scripts/run_demo_all.sh
```

---

## 📚 詳細ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| [QUICKSTART.md](QUICKSTART.md) | クイックスタートガイド |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | 詳細なデモ実行ガイド |
| [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) | 完全な使用方法 |
| [docs/DATA_CONVERSION_PIPELINE.md](docs/DATA_CONVERSION_PIPELINE.md) | 設計書 |

---

## 🆘 トラブルシューティング

### エラーが発生した場合

1. **ログを確認:**
   ```bash
   cat dataset_building.log
   cat outputs/demo/train.log
   ```

2. **テストを実行:**
   ```bash
   python scripts/test_data_pipeline.py
   ```

3. **バージョンを確認:**
   ```bash
   python --version
   pip list | grep torch
   ```

### よくあるエラー

| エラー | 解決方法 |
|-------|---------|
| XMLファイルが見つからない | `mkdir -p data/xml_logs`<br>`cp 2009*.xml data/xml_logs/` |
| メモリ不足 | `--draw-history 4 --discard-history 16` |
| CUDA out of memory | `--batch-size 64` または `--device cpu` |

---

## ✅ チェックリスト

実行前に確認：

- [ ] Python 3.8以上がインストールされている
- [ ] 必要なパッケージがインストールされている（`pip install -r requirements.txt`）
- [ ] XMLログファイルがある（または`quick_test.sh`でテスト）
- [ ] 十分なディスク容量がある（約10GB以上）
- [ ] 十分なメモリがある（8GB以上推奨）

---

## 🎉 次のステップ

デモが成功したら：

1. **より大規模なデータで学習**
   ```bash
   # 50,000ゲーム
   python scripts/build_comprehensive_dataset.py --max-games 50000
   python scripts/train_demo.py --epochs 100
   ```

2. **ハイパーパラメータ調整**
   ```bash
   python scripts/train_demo.py \
       --learning-rate 5e-4 \
       --d-model 512 \
       --num-layers 8
   ```

3. **評価と分析**
   ```bash
   # テストデータでの詳細評価
   python scripts/evaluate_model.py \
       --model outputs/demo/checkpoints/best_model.pt
   ```

---

**Happy Training! 🚀🀄**

質問や問題がある場合は、[DEMO_GUIDE.md](DEMO_GUIDE.md)の「トラブルシューティング」セクションを参照してください。

