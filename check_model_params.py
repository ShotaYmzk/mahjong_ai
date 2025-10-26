"""モデルのパラメータ数を確認するスクリプト"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
from src.model import TIT, SimplifiedDiscardOnlyHead, CompleteMahjongModel
from src.utils import load_config

print("=" * 60)
print("モデルパラメータ数確認")
print("=" * 60)

# 設定ファイルを読み込み
model_config = load_config('configs/model_config.yaml')

# TITモデルを作成
backbone = TIT(
    input_dim=model_config['input']['input_dim'],
    d_model=model_config['tit']['d_model'],
    nhead_inner=model_config['tit']['nhead_inner'],
    nhead_outer=model_config['tit']['nhead_outer'],
    dim_feedforward=model_config['tit']['dim_feedforward'],
    dropout=model_config['tit']['dropout'],
    num_inner_layers=model_config['tit']['num_inner_layers'],
    num_outer_layers=model_config['tit']['num_outer_layers'],
    num_tile_groups=model_config['input']['num_tile_groups'],
    tile_group_size=model_config['input']['tile_group_size']
)

head = SimplifiedDiscardOnlyHead(
    d_model=model_config['tit']['d_model'],
    num_tiles=model_config['input']['num_tile_types'],
    dropout=model_config['tit']['dropout']
)

model = CompleteMahjongModel(backbone, head)

# パラメータ数を計算
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
backbone_params = sum(p.numel() for p in backbone.parameters())
head_params = sum(p.numel() for p in head.parameters())

print(f"\n【モデル構成】")
print(f"  Inner Transformer層数: {model_config['tit']['num_inner_layers']}")
print(f"  Outer Transformer層数: {model_config['tit']['num_outer_layers']}")
print(f"  d_model: {model_config['tit']['d_model']}")
print(f"  nhead (Inner): {model_config['tit']['nhead_inner']}")
print(f"  nhead (Outer): {model_config['tit']['nhead_outer']}")
print(f"  dim_feedforward: {model_config['tit']['dim_feedforward']}")

print(f"\n【パラメータ数】")
print(f"  Backbone (TIT):     {backbone_params:>12,} params")
print(f"  Head (Discard):     {head_params:>12,} params")
print(f"  ─────────────────────────────────")
print(f"  Total:              {total_params:>12,} params")
print(f"  Trainable:          {trainable_params:>12,} params")
print(f"  Target:             {15_000_000:>12,} params (15M)")
print(f"  Difference:         {total_params - 15_000_000:>+12,} params")

# パーセンテージ
percentage = (total_params / 15_000_000) * 100
print(f"  Target達成率:       {percentage:>11.1f}%")

# メモリ使用量の推定
param_memory_mb = (total_params * 4) / (1024 ** 2)  # float32
print(f"\n【推定メモリ使用量】")
print(f"  パラメータ (float32): {param_memory_mb:.1f} MB")
print(f"  勾配 (float32):       {param_memory_mb:.1f} MB")
print(f"  オプティマイザ状態:    {param_memory_mb * 2:.1f} MB (AdamW)")
print(f"  合計 (推定):          {param_memory_mb * 4:.1f} MB")

# バッチサイズ512での推定
train_config = load_config('configs/train_config.yaml')
batch_size = train_config['data']['batch_size']
input_dim = model_config['input']['input_dim']
activation_memory = (batch_size * input_dim * 4) / (1024 ** 2)
print(f"\n【Batch size {batch_size}での推定メモリ】")
print(f"  入力テンソル:         {activation_memory:.1f} MB")
print(f"  中間活性化 (推定):     {activation_memory * 10:.1f} MB")
print(f"  総VRAM使用量 (推定):   {param_memory_mb * 4 + activation_memory * 10:.1f} MB")

# テスト実行
print(f"\n【動作テスト】")
try:
    test_input = torch.randn(batch_size, input_dim)
    with torch.no_grad():
        outputs, attention = model(test_input)
    print(f"  ✓ Forward pass成功")
    print(f"    入力shape: {test_input.shape}")
    if isinstance(outputs, dict):
        print(f"    出力shape: {outputs['discard'].shape if 'discard' in outputs else outputs.shape}")
    else:
        print(f"    出力shape: {outputs.shape}")
except Exception as e:
    print(f"  ✗ Forward pass失敗: {e}")

print("\n" + "=" * 60)
print("確認完了")
print("=" * 60)


