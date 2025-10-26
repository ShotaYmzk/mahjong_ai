#!/bin/bash
# デモ実行オールインワンスクリプト
# 
# 使用方法:
#   bash scripts/run_demo_all.sh
#
# このスクリプトは以下を自動実行します:
#   1. XMLデータの確認
#   2. データセット構築（10,000ゲーム）
#   3. モデルの学習（50エポック）

set -e  # エラーで停止

echo "=================================="
echo "麻雀AIデモ実行スクリプト"
echo "=================================="
echo ""

# プロジェクトルートに移動
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "プロジェクトルート: $PROJECT_ROOT"
echo ""

# 設定
XML_DIR="$PROJECT_ROOT/data/xml_logs"
PROCESSED_DIR="$PROJECT_ROOT/data/processed_demo"
OUTPUT_DIR="$PROJECT_ROOT/outputs/demo"
MAX_GAMES=10000
EPOCHS=50
BATCH_SIZE=256

# ステップ1: XMLデータの確認
echo "=================================="
echo "ステップ1: XMLデータの確認"
echo "=================================="

if [ ! -d "$XML_DIR" ]; then
    echo "❌ XMLディレクトリが見つかりません: $XML_DIR"
    echo ""
    echo "以下のいずれかを実行してください:"
    echo "  1. データ収集:"
    echo "     python scripts/collect_tenhou_data.py --max-files 10000"
    echo ""
    echo "  2. サンプルデータでテスト:"
    echo "     mkdir -p data/xml_logs"
    echo "     cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/"
    exit 1
fi

XML_COUNT=$(find "$XML_DIR" -name "*.xml" | wc -l)
echo "XMLファイル数: $XML_COUNT"

if [ "$XML_COUNT" -eq 0 ]; then
    echo "❌ XMLファイルが見つかりません"
    echo ""
    echo "サンプルデータでテストする場合:"
    echo "  cp 2009080100gm-00e1-0000-63d644dd.xml data/xml_logs/"
    exit 1
fi

if [ "$XML_COUNT" -lt 10 ]; then
    echo "⚠️  警告: XMLファイルが少なすぎます（$XML_COUNT 個）"
    echo "デモ実行には最低10,000ゲーム推奨です"
    echo ""
    read -p "少ないデータで続行しますか？ (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "中断しました"
        exit 1
    fi
    # 実際のファイル数に調整
    MAX_GAMES=$XML_COUNT
fi

echo "✅ XMLデータ確認完了"
echo ""

# ステップ2: データセット構築
echo "=================================="
echo "ステップ2: データセット構築"
echo "=================================="
echo "入力: $XML_DIR"
echo "出力: $PROCESSED_DIR"
echo "最大ゲーム数: $MAX_GAMES"
echo ""

if [ -d "$PROCESSED_DIR" ]; then
    echo "⚠️  処理済みデータが既に存在します: $PROCESSED_DIR"
    read -p "既存データを削除して再構築しますか？ (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "既存データを削除..."
        rm -rf "$PROCESSED_DIR"
    else
        echo "既存データを使用します"
        echo "✅ データセット確認完了"
        echo ""
        SKIP_DATASET_BUILD=true
    fi
fi

if [ "$SKIP_DATASET_BUILD" != "true" ]; then
    echo "データセット構築を開始..."
    echo "（処理時間: 約1-2時間）"
    echo ""
    
    python scripts/build_comprehensive_dataset.py \
        --xml-dir "$XML_DIR" \
        --output-dir "$PROCESSED_DIR" \
        --max-games "$MAX_GAMES" \
        --draw-history 8 \
        --discard-history 32 \
        --train-ratio 0.8 \
        --val-ratio 0.1 \
        --test-ratio 0.1
    
    if [ $? -ne 0 ]; then
        echo "❌ データセット構築に失敗しました"
        exit 1
    fi
    
    echo ""
    echo "✅ データセット構築完了"
fi

# データセット統計を表示
echo ""
echo "データセット統計:"
if [ -f "$PROCESSED_DIR/dataset_info.json" ]; then
    python -c "
import json
with open('$PROCESSED_DIR/dataset_info.json', 'r') as f:
    info = json.load(f)
    print(f\"  総ゲーム数: {info['dataset_info']['total_games']:,}\")
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
echo "データ: $PROCESSED_DIR"
echo "出力: $OUTPUT_DIR"
echo "エポック数: $EPOCHS"
echo "バッチサイズ: $BATCH_SIZE"
echo ""

if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  学習済みモデルが既に存在します: $OUTPUT_DIR"
    read -p "既存モデルを削除して再学習しますか？ (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "既存モデルを削除..."
        rm -rf "$OUTPUT_DIR"
    else
        echo "既存モデルを保持します"
        echo "（上書きされる可能性があります）"
    fi
fi

echo "学習を開始..."
echo "（処理時間: 約30分-1時間）"
echo ""
echo "ヒント: 別のターミナルで進捗確認"
echo "  tail -f $OUTPUT_DIR/train.log"
echo ""

python scripts/train_demo.py \
    --data-dir "$PROCESSED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate 1e-4

if [ $? -ne 0 ]; then
    echo "❌ 学習に失敗しました"
    exit 1
fi

echo ""
echo "✅ 学習完了"
echo ""

# 最終サマリー
echo "=================================="
echo "完了サマリー"
echo "=================================="
echo ""
echo "✅ XMLデータ: $XML_COUNT ファイル"
echo "✅ データセット: $PROCESSED_DIR"
echo "✅ モデル: $OUTPUT_DIR"
echo ""
echo "生成されたファイル:"
echo "  - チェックポイント: $OUTPUT_DIR/checkpoints/"
echo "  - ログ: $OUTPUT_DIR/train.log"
echo "  - メトリクス: $OUTPUT_DIR/metrics/"
echo ""
echo "次のステップ:"
echo "  1. 学習曲線を確認:"
echo "     cat $OUTPUT_DIR/train.log | grep 'Epoch'"
echo ""
echo "  2. テスト評価を確認:"
echo "     cat $OUTPUT_DIR/metrics/test_metrics.json"
echo ""
echo "  3. より大規模なデータで学習:"
echo "     bash scripts/run_demo_all.sh  # MAX_GAMESを変更"
echo ""
echo "=================================="
echo "🎉 デモ実行完了!"
echo "=================================="

