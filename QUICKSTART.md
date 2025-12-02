# 🚀 Quick Start Guide - GPU Training

## What Changed?

Your project has been migrated to follow the **GitHub repository standard** for reproducible ML experiments.

### Before (Old Way)
```bash
python train_with_gpu.py  # ❌ Removed
python main.py → option 2  # ⚠️ Still works, but not recommended for experiments
```

### After (New Way - Recommended)
```bash
python -m scripts.train --config configs/baseline.yaml --output-suffix exp_name
```

---

## Step-by-Step: Train Your First Model

### 1️⃣ Download Data (if not done yet)
```bash
./scripts/download_data.sh
```

### 2️⃣ Train Baseline Model

**Windows (PowerShell):**
```powershell
# Quick way - using convenience script
.\train_baseline.ps1 my_first_experiment

# Or manual way
python -m scripts.train --config configs/baseline.yaml --output-suffix my_first_experiment
```

**Linux/Mac:**
```bash
python -m scripts.train --config configs/baseline.yaml --output-suffix my_first_experiment
```

### 3️⃣ Monitor Training

Watch the progress in your terminal. You'll see:
- Epoch progress bars
- Train/Val loss and balanced accuracy
- Automatic checkpoint saving

### 4️⃣ Analyze Results

After training completes:
```bash
python main.py
# Choose option 1: Analyze Dataset
# Click "Load saved model" button
# Browse to: outputs/my_first_experiment/best.ckpt
```

You'll see:
- ✅ Model architecture and parameters
- ✅ Training curves (accuracy vs epoch)
- ✅ ROC and Precision-Recall curves
- ✅ Performance metrics

---

## Configuration File Explained

The `configs/baseline.yaml` controls everything:

```yaml
training:
  batch_size: 32        # Images per batch
  num_workers: 4        # Data loading threads
  epochs: 20            # Training epochs
  learning_rate: 0.01   # Initial learning rate
  
  loss:
    pos_weight: auto    # Automatically computed from dataset
  
  optimizer:
    type: Adam
  
  scheduler:
    type: StepLR
    step_size: 3        # Reduce LR every 3 epochs
    gamma: 0.1          # Multiply LR by 0.1
```

---

## Output Structure

After training `my_first_experiment`, you'll have:

```
outputs/my_first_experiment/
├── config.yaml              # Exact config used (for reproducibility)
├── best.ckpt               # Best model (by val_loss)
├── best_history.json       # Training curves for best model
├── last.ckpt               # Final epoch model
├── last_history.json       # Training curves for final model
└── tensorboard/            # TensorBoard logs (future)
```

---

## Advanced Usage

### Run Multiple Experiments

```powershell
# Experiment 1: Default settings
.\train_baseline.ps1 exp1_baseline

# Experiment 2: Modified config
# 1. Copy and edit config:
cp configs/baseline.yaml configs/exp2.yaml
# Edit exp2.yaml (e.g., change batch_size to 64)

# 2. Train with new config:
python -m scripts.train --config configs/exp2.yaml --output-suffix exp2_larger_batch
```

### Compare Results

All experiments are saved separately in `outputs/`:
```
outputs/
├── exp1_baseline/
├── exp2_larger_batch/
└── exp3_different_lr/
```

Load each one in the analysis tool to compare!

---

## GPU Training Verified ✅

Your setup already includes:
- ✅ GPU detection (CUDA support)
- ✅ OpenMP configuration
- ✅ Class imbalance handling (pos_weight)
- ✅ Balanced accuracy metric
- ✅ Learning rate scheduling

All these features are now configured via YAML and tracked automatically!

---

## Troubleshooting

**Problem:** `No module named 'yaml'`
```powershell
uv pip install --system pyyaml
```

**Problem:** "data/train.csv not found"
```bash
./scripts/download_data.sh
```

**Problem:** Training is slow
- Check GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce `num_workers` in config if using HDD
- Reduce `batch_size` if GPU memory is full

---

## Next Steps

1. ✅ Train baseline: `.\train_baseline.ps1 baseline_v1`
2. 📊 Analyze results: `python main.py` → option 1 → Load model
3. 🔬 Run experiments: Modify `configs/baseline.yaml` and train again
4. 📈 Compare models: Load different checkpoints from `outputs/`

Happy training! 🎉
