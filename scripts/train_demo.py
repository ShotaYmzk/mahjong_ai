#!/usr/bin/env python3
"""
デモ用学習スクリプト

10,000ゲームのデータセットで教師あり学習を実行します。

使用例:
    # 基本実行
    python scripts/train_demo.py \
        --data-dir data/processed_demo \
        --output-dir outputs/demo
    
    # カスタム設定
    python scripts/train_demo.py \
        --data-dir data/processed_demo \
        --output-dir outputs/demo \
        --epochs 50 \
        --batch-size 256 \
        --learning-rate 1e-4
"""

import sys
import argparse
import logging
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

# srcをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import TIT, SimplifiedDiscardOnlyHead, CompleteMahjongModel
from training import SupervisedTrainer
from evaluation import MetricsCalculator
from utils import setup_logger, set_seed


def setup_logging(log_file: Path):
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )


def load_dataset(data_dir: Path, split: str = 'train'):
    """データセットの読み込み
    
    Args:
        data_dir: データディレクトリ
        split: 'train', 'val', または 'test'
        
    Returns:
        (X, y) numpy arrays
    """
    split_dir = data_dir / split
    
    if split == 'train':
        X = np.load(split_dir / 'X_train.npy')
        y = np.load(split_dir / 'y_train.npy')
    else:
        X = np.load(split_dir / f'X_{split}.npy')
        y = np.load(split_dir / f'y_{split}.npy')
    
    return X, y


def create_dataloaders(data_dir: Path, batch_size: int, num_workers: int = 4):
    """DataLoaderの作成
    
    Args:
        data_dir: データディレクトリ
        batch_size: バッチサイズ
        num_workers: ワーカー数
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)
    
    # データ読み込み
    logger.info("Loading datasets...")
    X_train, y_train = load_dataset(data_dir, 'train')
    X_val, y_val = load_dataset(data_dir, 'val')
    X_test, y_test = load_dataset(data_dir, 'test')
    
    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val:   X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    # PyTorch Datasetの作成
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    
    # DataLoaderの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_model(input_dim: int, d_model: int = 256, num_layers: int = 6,
                num_heads: int = 8, dropout: float = 0.1):
    """モデルの作成
    
    Args:
        input_dim: 入力次元数
        d_model: モデルの隠れ次元数
        num_layers: Transformerレイヤー数
        num_heads: アテンションヘッド数
        dropout: ドロップアウト率
        
    Returns:
        CompleteMahjongModel
    """
    logger = logging.getLogger(__name__)
    
    # Backboneの作成
    backbone = TIT(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )
    
    # Headの作成
    head = SimplifiedDiscardOnlyHead(
        d_model=d_model,
        num_tiles=34
    )
    
    # 完全なモデル
    model = CompleteMahjongModel(backbone, head)
    
    # パラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='デモ用学習スクリプト（10,000ゲーム）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # データ設定
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='データディレクトリ（processed_demo など）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/demo',
        help='出力ディレクトリ'
    )
    
    # 学習設定
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='エポック数'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='バッチサイズ'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='学習率'
    )
    
    # モデル設定
    parser.add_argument(
        '--d-model',
        type=int,
        default=256,
        help='モデルの隠れ次元数'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=6,
        help='Transformerレイヤー数'
    )
    
    parser.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='アテンションヘッド数'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='ドロップアウト率'
    )
    
    # その他設定
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='デバイス'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoaderのワーカー数'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='乱数シード'
    )
    
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='チェックポイント保存の間隔（エポック）'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='学習を再開するチェックポイントのパス'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ログ設定
    log_file = output_dir / 'train.log'
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("デモ用学習スクリプト")
    logger.info("="*80)
    logger.info(f"データディレクトリ: {args.data_dir}")
    logger.info(f"出力ディレクトリ: {args.output_dir}")
    logger.info(f"エポック数: {args.epochs}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"学習率: {args.learning_rate}")
    logger.info("="*80)
    
    # データディレクトリの確認
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"データディレクトリが存在しません: {data_dir}")
        return 1
    
    # dataset_info.json を読み込んで入力次元を取得
    dataset_info_path = data_dir / 'dataset_info.json'
    if not dataset_info_path.exists():
        logger.error(f"dataset_info.json が見つかりません: {dataset_info_path}")
        return 1
    
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    input_dim = dataset_info['dataset_info']['feature_dimension']
    logger.info(f"入力次元数: {input_dim}")
    
    # 乱数シード設定
    set_seed(args.seed)
    
    # デバイス設定
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用デバイス: {device}")
    
    try:
        # DataLoaderの作成
        logger.info("\n" + "="*80)
        logger.info("Step 1: データ読み込み")
        logger.info("="*80)
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir, args.batch_size, args.num_workers
        )
        
        # モデルの作成
        logger.info("\n" + "="*80)
        logger.info("Step 2: モデル作成")
        logger.info("="*80)
        model = create_model(
            input_dim=input_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
        
        # オプティマイザの作成
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01
        )
        
        # スケジューラの作成
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # トレーナーの作成
        logger.info("\n" + "="*80)
        logger.info("Step 3: トレーナー初期化")
        logger.info("="*80)
        trainer = SupervisedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=str(output_dir / 'checkpoints'),
            log_dir=str(output_dir / 'logs'),
            scheduler=scheduler
        )
        
        # チェックポイントから再開
        start_epoch = 1
        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                logger.warning(f"チェックポイントが見つかりません: {args.resume}")
                logger.info("新規に学習を開始します")
            else:
                start_epoch = trainer.load_checkpoint(str(resume_path))
                logger.info(f"エポック {start_epoch} から学習を再開します")
        else:
            # latest.pth が存在する場合は自動的に再開
            latest_checkpoint = output_dir / 'checkpoints' / 'latest.pth'
            if latest_checkpoint.exists():
                logger.info(f"前回の学習を検出しました: {latest_checkpoint}")
                response = input("前回の学習から再開しますか？ (y/N): ")
                if response.lower() == 'y':
                    start_epoch = trainer.load_checkpoint(str(latest_checkpoint))
                    logger.info(f"エポック {start_epoch} から学習を再開します")
        
        # 学習開始
        logger.info("\n" + "="*80)
        logger.info("Step 4: 学習開始")
        logger.info("="*80)
        logger.info(f"エポック数: {args.epochs}")
        logger.info(f"開始エポック: {start_epoch}")
        logger.info(f"訓練バッチ数: {len(train_loader)}")
        logger.info(f"検証バッチ数: {len(val_loader)}")
        logger.info("="*80)
        
        trainer.train(
            num_epochs=args.epochs,
            save_every=args.save_every
        )
        
        # 最終評価
        logger.info("\n" + "="*80)
        logger.info("Step 5: 最終評価（テストデータ）")
        logger.info("="*80)
        
        model.eval()
        test_metrics = MetricsCalculator(num_classes=34)
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="Testing"):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                predictions = outputs['discard_logits'].argmax(dim=1)
                
                test_metrics.update(predictions.cpu(), batch_y.cpu())
        
        final_test_metrics = test_metrics.compute()
        
        logger.info("テストデータでの最終結果:")
        logger.info(f"  精度: {final_test_metrics['accuracy']:.4f}")
        logger.info(f"  Top-3精度: {final_test_metrics['top_3_accuracy']:.4f}")
        logger.info(f"  Top-5精度: {final_test_metrics['top_5_accuracy']:.4f}")
        
        # メトリクスを保存
        metrics_dir = output_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        with open(metrics_dir / 'test_metrics.json', 'w') as f:
            json.dump({k: float(v) if hasattr(v, 'item') else v 
                      for k, v in final_test_metrics.items()}, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("✅ 学習完了!")
        logger.info("="*80)
        logger.info(f"チェックポイント: {output_dir / 'checkpoints'}")
        logger.info(f"ログ: {log_file}")
        logger.info(f"メトリクス: {metrics_dir}")
        logger.info("="*80)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  学習が中断されました")
        return 130
    
    except Exception as e:
        logger.error(f"❌ エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

