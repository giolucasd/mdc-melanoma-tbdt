# Training Examples

This document shows how to train models using the new standardized approach.

## Quick Start - Training with GPU

### Using the CLI Script (Recommended - GitHub Standard)

Train the pure baseline (no augmentation):

```bash
# Pure CNN baseline without augmentation
python -m scripts.train --config configs/baseline_cnn_no_augmentation.yaml --output-suffix baseline_cnn_no_aug_v1
```

This will:
1. Load configuration from `configs/baseline_cnn_no_augmentation.yaml`
2. Automatically compute `pos_weight` from the training set
3. Train for 30 epochs (5 warmup) with GPU acceleration
4. Use only original data (no augmentation) for pure baseline
5. Save checkpoints to `outputs/baseline_cnn_no_aug_v1/`
6. Save training history for later analysis
7. All hyperparameters are centralized in the config file

### Using Interactive Mode (Alternative)

For quick experiments or interactive training:

```bash
python main.py
# Choose option 2: Train Model (GPU-accelerated)
```

## Configuration File

The baseline configuration is in `configs/baseline_cnn_no_augmentation.yaml`:

### `baseline_cnn_no_augmentation.yaml` - Pure CNN Baseline
- **Pure baseline**: No data augmentation, only resize + normalize
- **Training**: 30 epochs (5 warmup), batch_size=32, lr=0.01
- **Purpose**: Establish baseline performance with original data only
- **All hyperparameters centralized**: optimizer settings, scheduler, loss, data loading

The config controls:
- **Model**: Architecture selection (currently SimpleCNN)
- **Training**: Batch size, epochs, learning rate, optimizer (Adam with betas, eps, weight_decay), scheduler (StepLR)
- **Data**: Paths, image size, preprocessing (no augmentation)
- **Loss**: BCEWithLogitsLoss with automatic pos_weight computation
- **Output**: Checkpoint and logging directories

## Output Structure

After training, you'll find:

```
outputs/baseline_v1/
├── config.yaml           # Copy of configuration used
├── best.ckpt            # Best model checkpoint
├── best_history.json    # Training curves for best checkpoint
├── last.ckpt            # Last epoch checkpoint
├── last_history.json    # Training curves for last checkpoint
└── tensorboard/         # TensorBoard logs (future)
```

## Loading and Analyzing Models

Use the interactive analysis tool to visualize results:

```bash
python main.py
# Choose option 1: Analyze Dataset
# Click "Load saved model" button
# Select a .ckpt file from outputs/
```

This will show:
- Model architecture and parameters
- Training curves (accuracy vs epoch)
- ROC and Precision-Recall curves
- Evaluation metrics

## Advanced: Custom Configurations

Create your own config file based on `configs/baseline.yaml`:

```bash
cp configs/baseline.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your changes
python -m scripts.train --config configs/my_experiment.yaml --output-suffix my_exp_v1
```

## Migration from Old Approach

**Old way (deprecated):**
```bash
python train_with_gpu.py  # File removed
```

**New way (standard):**
```bash
python -m scripts.train --config configs/baseline.yaml --output-suffix baseline_v1
```

The new approach provides:
- ✅ Reproducible configurations via YAML
- ✅ Organized output structure
- ✅ Better checkpoint management
- ✅ Compatible with GitHub repository standard
- ✅ Easy hyperparameter tracking
