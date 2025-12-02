# Configuration Files

This directory contains YAML configuration files for different baseline models and experiments.

## Available Configurations

### Baseline Model

**`baseline_cnn_no_augmentation.yaml`** - **Pure CNN Baseline**
   - SimpleCNN trained on original data only
   - No data augmentation (only resize + normalize)
   - Training: 30 epochs (5 warmup), batch_size=32, lr=0.01
   - Purpose: Establish baseline performance with pure model capacity on raw data
   - Model type identifier: `baseline_cnn_no_augmentation`
   - All hyperparameters centralized in config for reproducibility

## Usage

Train the baseline using:

```bash
# Pure baseline (no augmentation)
python -m scripts.train \
  --config configs/baseline_cnn_no_augmentation.yaml \
  --output-suffix baseline_cnn_no_aug_v1
```

## Creating New Configurations

To add more baseline models or experiments:

1. Copy an existing config file
2. Modify parameters (model, training, data)
3. Update the `model.type` field with a descriptive identifier
4. Document your config here
5. Train with: `python -m scripts.train --config configs/your_config.yaml --output-suffix your_exp`

## Future Baselines

Planned configurations for comparison:
- Transfer learning baselines (ResNet, EfficientNet, etc.)
- Different architectures (VGG, DenseNet, etc.)
- Ensemble methods
- Advanced augmentation strategies

Each new baseline should have a clear purpose and be documented here.
