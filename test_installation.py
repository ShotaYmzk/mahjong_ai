"""Test script to verify Mahjong AI installation."""

import sys
from pathlib import Path

print("=" * 60)
print("Mahjong AI Installation Test")
print("=" * 60)

# Test imports
print("\n1. Testing module imports...")
try:
    from src.preprocessing import TenhouXMLParser, TileEncoder, MahjongDatasetBuilder
    print("   ✓ Preprocessing modules imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import preprocessing modules: {e}")
    sys.exit(1)

try:
    from src.model import TIT, SimplifiedMahjongTransformer, HierarchicalHead, XAIHooks
    print("   ✓ Model modules imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import model modules: {e}")
    sys.exit(1)

try:
    from src.training import SupervisedTrainer, RLTrainer, FanBackwardReward
    print("   ✓ Training modules imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import training modules: {e}")
    sys.exit(1)

try:
    from src.evaluation import MetricsCalculator, AttentionVisualizer
    print("   ✓ Evaluation modules imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import evaluation modules: {e}")
    sys.exit(1)

try:
    from src.utils import setup_logger, load_config, set_seed, get_device
    print("   ✓ Utility modules imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import utility modules: {e}")
    sys.exit(1)

# Test dependencies
print("\n2. Testing dependencies...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"   ✗ PyTorch not available: {e}")

try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except Exception as e:
    print(f"   ✗ NumPy not available: {e}")

try:
    import yaml
    print(f"   ✓ PyYAML available")
except Exception as e:
    print(f"   ✗ PyYAML not available: {e}")

try:
    import matplotlib
    print(f"   ✓ Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"   ✗ Matplotlib not available: {e}")

# Test XML parsing
print("\n3. Testing XML parser...")
try:
    parser = TenhouXMLParser()
    test_xml = Path("2009080100gm-00e1-0000-63d644dd.xml")
    
    if test_xml.exists():
        game = parser.parse_file(str(test_xml))
        if game:
            print(f"   ✓ Successfully parsed test XML")
            print(f"     - Game ID: {game.game_id}")
            print(f"     - Rounds: {len(game.rounds)}")
            print(f"     - Actions: {len(game.actions)}")
        else:
            print(f"   ✗ Failed to parse test XML")
    else:
        print(f"   ⚠ Test XML file not found (skipping)")
except Exception as e:
    print(f"   ✗ XML parser test failed: {e}")

# Test encoder
print("\n4. Testing tile encoder...")
try:
    encoder = TileEncoder()
    test_hand = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    encoding = encoder.encode_hand(test_hand)
    print(f"   ✓ Tile encoder working")
    print(f"     - Encoding shape: {encoding.shape}")
    print(f"     - Valid hand: {encoder.is_valid_hand(encoding)}")
except Exception as e:
    print(f"   ✗ Tile encoder test failed: {e}")

# Test model creation
print("\n5. Testing model creation...")
try:
    import torch
    model = TIT(input_dim=340, d_model=128, num_inner_layers=1, num_outer_layers=2)
    test_input = torch.randn(2, 340)
    features, attention = model(test_input)
    print(f"   ✓ TIT model created and tested")
    print(f"     - Feature shape: {features.shape}")
    print(f"     - Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ✗ Model creation test failed: {e}")

# Test configuration loading
print("\n6. Testing configuration loading...")
try:
    train_config_path = Path("configs/train_config.yaml")
    model_config_path = Path("configs/model_config.yaml")
    
    if train_config_path.exists():
        train_config = load_config(str(train_config_path))
        print(f"   ✓ Train config loaded: {len(train_config)} sections")
    else:
        print(f"   ⚠ Train config not found")
    
    if model_config_path.exists():
        model_config = load_config(str(model_config_path))
        print(f"   ✓ Model config loaded: {len(model_config)} sections")
    else:
        print(f"   ⚠ Model config not found")
except Exception as e:
    print(f"   ✗ Config loading test failed: {e}")

# Test directory structure
print("\n7. Checking directory structure...")
required_dirs = [
    "data/raw",
    "data/processed",
    "data/split",
    "src/preprocessing",
    "src/model",
    "src/training",
    "src/evaluation",
    "src/utils",
    "configs",
    "notebooks",
    "outputs/checkpoints",
    "outputs/logs",
    "outputs/visualizations"
]

for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"   ✓ {dir_path}")
    else:
        print(f"   ⚠ {dir_path} (missing)")

print("\n" + "=" * 60)
print("Installation test completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Place XML game logs in data/raw/")
print("3. Run training: bash run_train.sh")
print("=" * 60)

