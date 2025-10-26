# Mahjong AI with Transformer and XAI - 教師あり学習版

A deep learning-based Mahjong AI system using Transformer-in-Transformer (TIT) architecture with explainable AI (XAI) capabilities.

## Features

- **Transformer-in-Transformer (TIT) Architecture**: Hierarchical transformer model for processing Mahjong game states
- **Discard History Tracking**: Track all players' discards from the beginning of each round
- **Shanten Calculation**: Integrated shanten number calculation for hand evaluation
- **Explainable AI (XAI)**: Attention visualization and gradient-based attribution methods
- **Comprehensive Data Pipeline**: XML parsing, feature encoding with discard history, and dataset building
- **Supervised Learning**: Train directly from expert gameplay data with cross-entropy loss

## Project Structure

```
mahjong_ai/
├── data/
│   ├── raw/                    # XML game logs
│   ├── processed/              # Preprocessed numpy arrays
│   ├── dataset_stats.json      # Dataset statistics
│   └── split/                  # train/val/test splits
│
├── src/
│   ├── preprocessing/
│   │   ├── parse_xml.py        # XML→parsed game objects
│   │   ├── feature_encoder.py  # 1x34 tile encoding
│   │   └── dataset_builder.py  # TensorDataset generation
│   │
│   ├── model/
│   │   ├── transformer_tit.py  # TIT architecture
│   │   ├── hierarchical_head.py # Multi-head output
│   │   └── xai_hooks.py        # Attention/activation hooks
│   │
│   ├── training/
│   │   └── train_supervised.py # Supervised learning only
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy, SP, HR metrics
│   │   └── visualize_attention.py # Attention visualization
│   │
│   └── utils/
│       ├── logger.py
│       ├── config_loader.py
│       └── seed.py
│
├── configs/
│   ├── train_config.yaml       # Training hyperparameters
│   └── model_config.yaml       # Model architecture
│
├── notebooks/
│   ├── data_explore.ipynb
│   ├── model_debug.ipynb
│   └── attention_analysis.ipynb
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── visualizations/
│
├── requirements.txt
├── README.md
└── run_train.sh
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mahjong_ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Prepare Data

Place your Tenhou XML game logs in `data/raw/`:

```bash
# Example: Download or copy XML files
cp /path/to/xml/files/*.xml data/raw/
```

### 2. Build Dataset

```python
from src.preprocessing import MahjongDatasetBuilder

builder = MahjongDatasetBuilder(
    data_dir='data/raw',
    output_dir='data/processed'
)

# Build and split dataset
builder.build_dataset(max_games=1000)
datasets = builder.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### 3. Train Model

#### Supervised Learning

```bash
# Using the training script
bash run_train.sh
```

Or in Python:

```python
from src.model import TIT, SimplifiedDiscardOnlyHead, CompleteMahjongModel
from src.training import SupervisedTrainer, create_optimizer, create_scheduler
from src.utils import setup_logger, set_seed, get_device

# Setup
set_seed(42)
device = get_device()
logger = setup_logger('training', log_file='outputs/logs/train.log')

# Build model
backbone = TIT(input_dim=340, d_model=256)
head = SimplifiedDiscardOnlyHead(d_model=256, num_tiles=34)
model = CompleteMahjongModel(backbone, head)

# Create optimizer and scheduler
optimizer = create_optimizer(model, optimizer_name='adamw', learning_rate=1e-4)
scheduler = create_scheduler(optimizer, scheduler_name='plateau')

# Train
trainer = SupervisedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device
)

trainer.train(num_epochs=150, scheduler=scheduler)
```

### 4. Evaluate Model

```python
from src.evaluation import MetricsCalculator, AttentionVisualizer

# Compute metrics
metrics_calc = MetricsCalculator(num_classes=34)
metrics_calc.update(predictions, targets)
metrics = metrics_calc.compute()

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
print(f"Success Probability: {metrics['success_probability']:.4f}")

# Visualize attention
visualizer = AttentionVisualizer(output_dir='outputs/visualizations')
visualizer.plot_attention_heatmap(attention_weights, save_name='attention_map.png')
```

### 5. Shanten Calculation

```python
from src.utils import calculate_shanten, analyze_hand_details, print_hand_analysis

# Calculate shanten for a hand
tiles_34 = [0] * 34  # Your hand encoding
shanten = calculate_shanten(tiles_34)
print(f"Shanten: {shanten}")  # -1=win, 0=tenpai, 1+=N-shanten

# Detailed hand analysis
analysis = analyze_hand_details("123m456p789s1122z")
print_hand_analysis(analysis)
```

### 6. XAI Analysis

```python
from src.model import XAIHooks, AttentionAnalyzer
from src.preprocessing import GameStateManager

# Track discard history
manager = GameStateManager()
manager.start_new_round(round_num=0, dealer=0)
manager.add_discard(player_id=0, tile=120, turn=1)

# Get game state with history
state = manager.get_current_state()
discard_seq = state.encode_discard_history_sequence(max_length=64)

# Register hooks for attention visualization
xai_hooks = XAIHooks(model)
xai_hooks.register_all_attention_hooks()

# Forward pass
outputs, attention_weights = model(input_tensor)
attention = xai_hooks.get_attention_weights()

# Visualize
from src.evaluation import AttentionVisualizer
visualizer = AttentionVisualizer()
visualizer.plot_attention_heatmap(
    attention['outer_transformer.layers.0'],
    save_name='attention_map.png'
)
```

## Configuration

Edit configuration files in `configs/`:

- `train_config.yaml`: Training hyperparameters (batch size, learning rate, discard history, etc.)
- `model_config.yaml`: Model architecture (hidden dims, num layers, discard history settings, etc.)

## Model Architecture

### Transformer-in-Transformer (TIT)

1. **Input**: Game state encoded as 340-dim vector (34 tiles × 10 feature groups)
2. **Tile Embedding**: Project each tile group to d_model dimensions
3. **Inner Transformer**: Process tiles within each group
4. **Group Aggregation**: Aggregate group representations
5. **Outer Transformer**: Process relationships between groups
6. **Global Pooling**: Aggregate to single vector
7. **Hierarchical Head**: Predict action type, claim type, and tile selection

### Feature Encoding

Uses 1x34 tile encoding (based on Tjong et al.):
- Each tile type (0-33) represented by count (0-4)
- 10 feature groups:
  - Own hand
  - Own melds
  - Opponent 1-3 discards
  - Opponent 1-3 melds
  - Dora indicators

## Metrics

- **Accuracy**: Exact match with expert action
- **Top-k Accuracy**: Expert action in top-k predictions
- **Success Probability (SP)**: Average probability of correct tile
- **Hit Rate (HR)**: Percentage of times correct tile in top-k
- **Win Rate**: Percentage of games won (in actual play)

## Citations

If you use this code, please cite:

```bibtex
@misc{mahjong_ai_tit,
  title={Mahjong AI with Transformer-in-Transformer and Explainable AI},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/yourusername/mahjong-ai}}
}
```

## License

MIT License

## Acknowledgments

- Tenhou for game logs
- Transformer architecture based on "Attention Is All You Need" (Vaswani et al., 2017)
- TIT architecture inspired by "Tokens-to-Token ViT" (Yuan et al., 2021)
- Mahjong AI research community

## TODO

- [ ] Implement complete shanten calculator
- [ ] Add meld decoding for chi/pon/kan
- [ ] Implement full game environment for RL
- [ ] Add multi-GPU training support
- [ ] Create web interface for playing against AI
- [ ] Add more sophisticated defense logic

## Contact

For questions or issues, please open an issue on GitHub.

