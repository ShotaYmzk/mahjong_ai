

# 天鳳データ収集チュートリアル

このチュートリアルでは、天鳳の対局ログを収集する方法を説明します。

## 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [クイックスタート](#クイックスタート)
4. [詳細な使用方法](#詳細な使用方法)
5. [プログラマティックな使用](#プログラマティックな使用)
6. [トラブルシューティング](#トラブルシューティング)

## 概要

天鳳データ収集システムは、以下の機能を提供します：

- **HTMLログからの自動URL抽出**
- **XMLファイルの並列ダウンロード**
- **ダウンロードしたファイルの自動検証**
- **無効なファイルのクリーンアップ**
- **詳細な統計情報とログ**

## 前提条件

### 必要なパッケージ

```bash
pip install requests
```

### ディレクトリ構造

```
/home/ubuntu/Documents/
├── tenhou_dataset/          # 展開済みHTMLファイル
│   ├── 2024/
│   │   ├── scc20240101.html
│   │   ├── scc20240102.html
│   │   └── ...
│   ├── 2023/
│   └── ...
└── mahjong_ai/
    ├── data/
    │   └── xml_logs/        # XMLファイルの保存先（自動作成）
    └── scripts/
        └── collect_tenhou_data.py
```

## クイックスタート

### 1. 基本的な使用

```bash
# デフォルト設定でデータ収集
python scripts/collect_tenhou_data.py
```

これにより、2024年と2023年の「四鳳」のデータが収集されます。

### 2. 特定の年のデータを収集

```bash
# 2024年のみ
python scripts/collect_tenhou_data.py --years 2024

# 複数の年を指定
python scripts/collect_tenhou_data.py --years 2024 2023 2022
```

### 3. ゲームタイプを指定

```bash
# 三麻（三鳳）のデータを収集
python scripts/collect_tenhou_data.py --game-type 三鳳

# 四麻一般
python scripts/collect_tenhou_data.py --game-type 四般
```

利用可能なゲームタイプ：
- `四鳳` (デフォルト) - 四麻鳳凰卓
- `三鳳` - 三麻鳳凰卓
- `四般` - 四麻一般卓
- `三般` - 三麻一般卓

### 4. テスト実行

```bash
# 最初の10ファイルのみ処理（テスト用）
python scripts/collect_tenhou_data.py --max-files 10

# 無効なファイルもクリーンアップ
python scripts/collect_tenhou_data.py --max-files 10 --clean
```

## 詳細な使用方法

### コマンドラインオプション

```bash
python scripts/collect_tenhou_data.py [OPTIONS]

オプション:
  --html-dir PATH         HTMLファイルのディレクトリ（デフォルト: /home/ubuntu/Documents/tenhou_dataset）
  --xml-dir PATH          XMLファイルの保存先（デフォルト: ./data/xml_logs）
  --years YEAR [YEAR ...] 処理する年（デフォルト: 2024 2023）
  --game-type TYPE        ゲームタイプ（デフォルト: 四鳳）
  --max-files N          最大HTMLファイル数（テスト用）
  --clean                無効なファイルを削除
  --log-file PATH        ログファイルのパス（デフォルト: tenhou_data_collection.log）
```

### 実用例

#### 例1: 2024年の四鳳データを大量収集

```bash
python scripts/collect_tenhou_data.py \
    --years 2024 \
    --game-type 四鳳 \
    --clean
```

#### 例2: 複数年の三麻データを収集

```bash
python scripts/collect_tenhou_data.py \
    --years 2024 2023 2022 \
    --game-type 三鳳 \
    --xml-dir /path/to/sanma_data
```

#### 例3: カスタムディレクトリで実行

```bash
python scripts/collect_tenhou_data.py \
    --html-dir /mnt/storage/tenhou_html \
    --xml-dir /mnt/storage/tenhou_xml \
    --years 2024
```

## プログラマティックな使用

Pythonコードから直接使用することもできます。

### 基本的な使用

```python
from pathlib import Path
from data_collection import TenhouDataFetcher

# フェッチャーの初期化
fetcher = TenhouDataFetcher(
    html_base_dir="/home/ubuntu/Documents/tenhou_dataset",
    xml_save_dir="/home/ubuntu/Documents/mahjong_ai/data/xml_logs"
)

# データ収集
xml_files = fetcher.fetch_from_html_directory(
    year_range=range(2024, 2022, -1),
    game_type="四鳳",
    max_files=None  # 制限なし
)

print(f"収集されたファイル数: {len(xml_files)}")

# 統計情報
print(f"成功: {fetcher.stats['successful']}")
print(f"失敗: {fetcher.stats['failed']}")
```

### 特定のログIDをダウンロード

```python
from data_collection import TenhouDataFetcher

fetcher = TenhouDataFetcher()

# ログIDから直接ダウンロード
log_id = "2024010100gm-00a9-0000-12345678"
xml_file = fetcher.download_from_log_id(log_id)

if xml_file:
    # プレイヤー名を取得
    players = fetcher.get_player_names_from_xml(xml_file)
    print(f"プレイヤー: {', '.join(players)}")
    
    # ファイルの検証
    is_valid = fetcher.validate_xml_file(xml_file)
    print(f"有効性: {is_valid}")
```

### HTMLログの解析

```python
from pathlib import Path
from data_collection import HTMLLogParser

parser = HTMLLogParser()

# HTMLファイルを解析
html_file = Path("/path/to/scc20240101.html")
logs = parser.parse_html_file(html_file, game_type="四鳳")

# 結果を表示
for log in logs[:5]:
    print(f"ログID: {log['log_id']}")
    print(f"URL: {log['display_url']}")
    print(f"タイプ: {log['game_type']}")
    print()

# 統計情報を取得
stats = parser.get_statistics(html_file)
print(f"統計: {stats}")
```

### 並列ダウンロード

```python
from pathlib import Path
from data_collection import XMLDownloader

downloader = XMLDownloader(
    save_dir=Path("./xml_logs"),
    max_workers=10,  # 並列数を増やして高速化
    request_delay=0.3  # 遅延を短く
)

# URLリストから並列ダウンロード
urls = [
    "http://tenhou.net/0/log/?2024010100gm-00a9-0000-12345678",
    "http://tenhou.net/0/log/?2024010101gm-00a9-0000-23456789",
    # ... 他のURL
]

xml_files = downloader.download_batch(urls, show_progress=True)
print(f"ダウンロード完了: {len(xml_files)} / {len(urls)}")

# ファイル検証
valid, invalid = downloader.verify_downloaded_files()
print(f"有効: {valid}, 無効: {invalid}")

# 無効なファイルを削除
if invalid > 0:
    removed = downloader.cleanup_invalid_files()
    print(f"削除: {removed} ファイル")
```

## トラブルシューティング

### 問題1: HTMLファイルが見つからない

**エラーメッセージ:**
```
HTMLディレクトリが存在しません: /home/ubuntu/Documents/tenhou_dataset
```

**解決方法:**
1. HTMLファイルを正しいディレクトリに配置してください
2. `--html-dir` オプションで正しいパスを指定してください

```bash
python scripts/collect_tenhou_data.py --html-dir /correct/path/to/html
```

### 問題2: ダウンロードが失敗する

**症状:**
- `failed` カウントが多い
- タイムアウトエラー

**解決方法:**
1. インターネット接続を確認
2. 天鳳のサーバーが応答しているか確認
3. リトライ間隔を長くする（コードを編集）

```python
# tenhou_fetcher.py の request_delay を増やす
sleep(1.0)  # デフォルト: 0.5
```

### 問題3: 無効なXMLファイルが多い

**症状:**
- XMLファイルが小さすぎる
- `<mjloggm>` タグがない

**解決方法:**
```bash
# 無効なファイルをクリーンアップ
python scripts/collect_tenhou_data.py --clean

# または、Python で直接実行
python -c "from data_collection import TenhouDataFetcher; \
           f = TenhouDataFetcher(); \
           print(f'削除: {f.clean_invalid_files()} ファイル')"
```

### 問題4: メモリ不足

**症状:**
- 大量のファイル処理時にメモリエラー

**解決方法:**
```bash
# ファイル数を制限して分割処理
python scripts/collect_tenhou_data.py --max-files 100
```

### 問題5: ログファイルが大きくなりすぎる

**解決方法:**
```bash
# ログファイルを定期的に削除
rm tenhou_data_collection.log

# または、カスタムログファイルを使用
python scripts/collect_tenhou_data.py --log-file /tmp/collection_$(date +%Y%m%d).log
```

## ベストプラクティス

### 1. 段階的なデータ収集

大量のデータを収集する場合は、段階的に実行：

```bash
# ステップ1: 2024年のみ（テスト）
python scripts/collect_tenhou_data.py --years 2024 --max-files 10

# ステップ2: 2024年全体
python scripts/collect_tenhou_data.py --years 2024

# ステップ3: 複数年
python scripts/collect_tenhou_data.py --years 2024 2023 2022
```

### 2. 定期的なクリーンアップ

```bash
# 週次でクリーンアップを実行
python scripts/collect_tenhou_data.py --clean
```

### 3. ログの確認

```bash
# ログファイルを確認して問題を特定
tail -f tenhou_data_collection.log

# エラーのみを抽出
grep ERROR tenhou_data_collection.log
```

### 4. ディスク容量の確認

```bash
# XMLディレクトリのサイズを確認
du -sh data/xml_logs

# ファイル数を確認
ls data/xml_logs | wc -l
```

## パフォーマンスチューニング

### 並列処理の最適化

```python
# xml_downloader.py を編集
downloader = XMLDownloader(
    save_dir=save_dir,
    max_workers=20,  # CPU数に応じて調整
    request_delay=0.1  # 短くして高速化（サーバーに注意）
)
```

### バッチサイズの調整

```bash
# 小さいバッチで実行して安定性を確保
for year in 2024 2023 2022; do
    python scripts/collect_tenhou_data.py --years $year
    sleep 60  # 各年の間に休憩
done
```

## まとめ

このデータ収集システムにより：

✅ **自動化**: HTMLからXMLへの自動変換
✅ **並列処理**: 高速なダウンロード
✅ **検証**: ファイルの自動検証
✅ **ロギング**: 詳細なログ記録
✅ **拡張性**: カスタマイズ可能な設計

完璧な麻雀AIのための大量データ収集が可能になりました！

