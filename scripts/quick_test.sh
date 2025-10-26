#!/bin/bash
# クイックテストスクリプト（5分で完了）
# 
# サンプルデータを使用して、パイプライン全体が正常に動作することを確認します。

set -e

echo "=================================="
echo "クイックテスト（5分）"
echo "=================================="
echo ""

# プロジェクトルートに移動
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "プロジェクトルート: $PROJECT_ROOT"
echo ""

# テスト用ディレクトリ
XML_DIR="$PROJECT_ROOT/data/xml_logs_test"
PROCESSED_DIR="$PROJECT_ROOT/data/processed_test"
OUTPUT_DIR="$PROJECT_ROOT/outputs/test"

# クリーンアップ
echo "既存のテストデータをクリーンアップ..."
rm -rf "$XML_DIR" "$PROCESSED_DIR" "$OUTPUT_DIR"

# ステップ1: サンプルXMLを準備
echo ""
echo "=================================="
echo "ステップ1: サンプルXMLを準備"
echo "=================================="

mkdir -p "$XML_DIR"
cp 2009080100gm-00e1-0000-63d644dd.xml "$XML_DIR/"

XML_COUNT=$(find "$XML_DIR" -name "*.xml" | wc -l)
echo "XMLファイル数: $XML_COUNT"
echo "✅ サンプルXML準備完了"
echo ""

# ステップ2: データセット構築
echo "=================================="
echo "ステップ2: データセット構築"
echo "=================================="
echo "（処理時間: 約1分）"
echo ""

python scripts/build_comprehensive_dataset.py \
    --xml-dir "$XML_DIR" \
    --output-dir "$PROCESSED_DIR" \
    --max-games 1 \
    --draw-history 4 \
    --discard-history 16 \
    --no-progress

if [ $? -ne 0 ]; then
    echo "❌ データセット構築に失敗しました"
    exit 1
fi

echo ""
echo "✅ データセット構築完了"
echo ""

# データセット統計を表示
if [ -f "$PROCESSED_DIR/dataset_info.json" ]; then
    echo "データセット統計:"
    python -c "
import json
with open('$PROCESSED_DIR/dataset_info.json', 'r') as f:
    info = json.load(f)
    print(f\"  総サンプル数: {info['dataset_info']['total_samples']:,}\")
    print(f\"  特徴量次元: {info['dataset_info']['feature_dimension']}\")
    print(f\"  訓練データ: {info['split_statistics']['train_size']:,}\")
    print(f\"  検証データ: {info['split_statistics']['val_size']:,}\")
    print(f\"  テストデータ: {info['split_statistics']['test_size']:,}\")
"
fi
echo ""

# ステップ3: モデルの学習
echo "=================================="
echo "ステップ3: モデルの学習"
echo "=================================="
echo "（処理時間: 約2-3分）"
echo ""

python scripts/train_demo.py \
    --data-dir "$PROCESSED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 3 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --d-model 128 \
    --num-layers 2 \
    --num-workers 0

if [ $? -ne 0 ]; then
    echo "❌ 学習に失敗しました"
    exit 1
fi

echo ""
echo "✅ 学習完了"
echo ""

# 結果を表示
echo "=================================="
echo "テスト結果"
echo "=================================="
echo ""

if [ -f "$OUTPUT_DIR/metrics/test_metrics.json" ]; then
    echo "テストメトリクス:"
    cat "$OUTPUT_DIR/metrics/test_metrics.json"
    echo ""
fi

echo "生成されたファイル:"
echo "  - データセット: $PROCESSED_DIR"
echo "  - モデル: $OUTPUT_DIR/checkpoints/"
echo "  - ログ: $OUTPUT_DIR/train.log"
echo ""

# クリーンアップの確認
echo "=================================="
echo "クリーンアップ"
echo "=================================="
echo ""
read -p "テストデータを削除しますか？ (Y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "テストデータを削除..."
    rm -rf "$XML_DIR" "$PROCESSED_DIR" "$OUTPUT_DIR"
    echo "✅ クリーンアップ完了"
else
    echo "テストデータを保持します"
    echo "  - $XML_DIR"
    echo "  - $PROCESSED_DIR"
    echo "  - $OUTPUT_DIR"
fi

echo ""
echo "=================================="
echo "🎉 クイックテスト完了!"
echo "=================================="
echo ""
echo "次のステップ:"
echo "  1. 本番デモを実行:"
echo "     bash scripts/run_demo_all.sh"
echo ""
echo "  2. クイックスタートガイドを確認:"
echo "     cat QUICKSTART.md"
echo ""
echo "=================================="

