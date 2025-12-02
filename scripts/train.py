#!/usr/bin/env python
"""
Training script for melanoma classification models.

This script provides a CLI interface for training models with YAML configurations,
following the pattern from the GitHub repository.

Usage:
    python -m scripts.train --config configs/baseline.yaml --output-suffix baseline_v1
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Set OpenMP configuration before importing scientific libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import pandas as pd
from src.data import get_train_val_dataloaders, DATA_PATH
from src.model import SimpleCNN
from src.train import ModelTrainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_dir(base_dir: str, suffix: str) -> Path:
    """Create output directory structure."""
    output_dir = Path(base_dir) / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "tensorboard").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    return output_dir


def save_config_copy(config: dict, output_dir: Path):
    """Save a copy of the configuration used for training."""
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")


def compute_pos_weight(train_csv_path: str) -> float:
    """Compute pos_weight for handling class imbalance."""
    train_df = pd.read_csv(train_csv_path)
    neg = int((train_df["target"] == 0).sum())
    pos = int((train_df["target"] == 1).sum())
    
    if pos == 0:
        return 1.0
    
    pos_weight = neg / pos
    print(f"Class distribution - Negative: {neg}, Positive: {pos}")
    print(f"Computed pos_weight: {pos_weight:.3f}")
    
    return pos_weight


def train(config: dict, output_dir: Path):
    """Main training function."""
    
    # Extract configuration
    train_config = config['training']
    data_config = config['data']
    model_config = config['model']
    
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    epochs = train_config['epochs']
    lr = train_config['learning_rate']
    
    # Load data
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    train_loader, val_loader = get_train_val_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model
    print("\n" + "="*50)
    print(f"Creating model: {model_config['name']}")
    print("="*50)
    model = SimpleCNN()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Compute pos_weight if needed
    pos_weight = None
    if train_config['loss'].get('pos_weight') == 'auto':
        train_csv = data_config['train_csv']
        pos_weight = compute_pos_weight(train_csv)
    
    # Create trainer
    print("\n" + "="*50)
    print("Initializing trainer...")
    print("="*50)
    trainer = ModelTrainer(model, device=device, pos_weight=pos_weight, lr=lr)
    
    # Train model
    print("\n" + "="*50)
    print(f"Starting training for {epochs} epochs...")
    print("="*50)
    trainer.fit(train_loader, val_loader, epochs=epochs)
    
    # Save checkpoints
    print("\n" + "="*50)
    print("Saving model checkpoints...")
    print("="*50)
    
    best_ckpt_path = output_dir / "best.ckpt"
    last_ckpt_path = output_dir / "last.ckpt"
    
    # Save best and last checkpoints
    trainer.save_model(str(best_ckpt_path))
    trainer.save_model(str(last_ckpt_path))
    
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print(f"Last checkpoint saved to: {last_ckpt_path}")
    
    # Print final results
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Final Train Balanced Accuracy: {trainer.history['train_acc'][-1]:.4f}")
    print(f"Final Val Balanced Accuracy: {trainer.history['val_acc'][-1]:.4f}")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"To view TensorBoard logs (if implemented):")
    print(f"  tensorboard --logdir {output_dir / 'tensorboard'}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train melanoma classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline model
  python -m scripts.train --config configs/baseline.yaml --output-suffix baseline_v1
  
  # Train with custom output directory
  python -m scripts.train --config configs/baseline.yaml --output-suffix exp1 --output-dir results
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file (e.g., configs/baseline.yaml)'
    )
    
    parser.add_argument(
        '--output-suffix',
        type=str,
        required=True,
        help='Suffix for output directory (e.g., baseline_v1)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Base output directory (default: outputs)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("="*50)
    print(f"Loading configuration from: {args.config}")
    print("="*50)
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = setup_output_dir(args.output_dir, args.output_suffix)
    print(f"Output directory: {output_dir}")
    
    # Save configuration copy
    save_config_copy(config, output_dir)
    
    # Run training
    train(config, output_dir)


if __name__ == "__main__":
    main()
