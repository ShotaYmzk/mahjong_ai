#!/bin/bash
# Training script for Mahjong AI

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Configuration files
TRAIN_CONFIG="configs/train_config.yaml"
MODEL_CONFIG="configs/model_config.yaml"

# Training parameters (can be overridden)
NUM_EPOCHS=${1:-100}
BATCH_SIZE=${2:-32}
LEARNING_RATE=${3:-0.0001}

echo "======================================"
echo "Mahjong AI Training"
echo "======================================"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "======================================"

# Create output directories
mkdir -p outputs/checkpoints
mkdir -p outputs/logs
mkdir -p outputs/visualizations

# Run training
python3 <<EOF
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.preprocessing import MahjongDatasetBuilder
from src.model import TIT, SimplifiedDiscardOnlyHead, CompleteMahjongModel
from src.training import SupervisedTrainer, create_optimizer, create_scheduler
from src.utils import setup_logger, set_seed, get_device, load_config

# Setup
print("Setting up training environment...")
set_seed(42)
device = get_device(use_cuda=True)
logger = setup_logger('training', log_file='outputs/logs/train.log')

# Load configurations
print("Loading configurations...")
train_config = load_config('$TRAIN_CONFIG')
model_config = load_config('$MODEL_CONFIG')

# Check if processed data exists
processed_dir = Path('data/processed')
if not (processed_dir / 'X.npy').exists():
    print("Processed data not found. Building dataset...")
    builder = MahjongDatasetBuilder(
        data_dir='data/raw',
        output_dir='data/processed'
    )
    stats = builder.build_dataset(max_games=train_config['data'].get('max_games'))
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
else:
    print("Using existing processed data")

# Create dataset builder and loaders
print("Creating data loaders...")
builder = MahjongDatasetBuilder(
    data_dir='data/raw',
    output_dir='data/processed'
)

dataloaders = builder.create_dataloaders(
    batch_size=train_config['data']['batch_size'],
    num_workers=train_config['data']['num_workers']
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")

# Build model
print("Building model...")
model_type = train_config.get('model_type', 'tit')

if model_type == 'tit':
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
else:
    from src.model import SimplifiedMahjongTransformer
    backbone = SimplifiedMahjongTransformer(
        input_dim=model_config['input']['input_dim'],
        d_model=model_config['simplified_transformer']['d_model'],
        nhead=model_config['simplified_transformer']['nhead'],
        num_layers=model_config['simplified_transformer']['num_layers'],
        dim_feedforward=model_config['simplified_transformer']['dim_feedforward'],
        dropout=model_config['simplified_transformer']['dropout']
    )

# Use simplified discard-only head for supervised learning
head = SimplifiedDiscardOnlyHead(
    d_model=model_config['tit']['d_model'],
    num_tiles=model_config['input']['num_tile_types'],
    dropout=model_config['discard_head']['dropout']
)

model = CompleteMahjongModel(backbone, head)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Create optimizer and scheduler
print("Creating optimizer and scheduler...")
optimizer = create_optimizer(
    model,
    optimizer_name=train_config['training']['optimizer'],
    learning_rate=train_config['training']['learning_rate'],
    weight_decay=train_config['training']['weight_decay']
)

scheduler = create_scheduler(
    optimizer,
    scheduler_name=train_config['training']['scheduler'],
    **train_config['training'].get('scheduler_params', {})
)

# Create trainer
print("Initializing trainer...")
trainer = SupervisedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    checkpoint_dir=train_config['logging']['checkpoint_dir'],
    log_dir=train_config['logging']['log_dir']
)

# Train
print("Starting training...")
print("=" * 50)

trainer.train(
    num_epochs=train_config['training']['num_epochs'],
    scheduler=scheduler
)

print("=" * 50)
print("Training completed!")
print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
print(f"Best validation loss: {trainer.best_val_loss:.4f}")
print(f"Checkpoints saved to: {train_config['logging']['checkpoint_dir']}")
print(f"Logs saved to: {train_config['logging']['log_dir']}")

EOF

echo "======================================"
echo "Training script completed!"
echo "======================================"

