# Melanoma Classification <!-- omit from toc -->

![GitHub repo size](https://img.shields.io/github/repo-size/giolucasd/mdc-melanoma-tbdt)
![GitHub contributors](https://img.shields.io/github/contributors/giolucasd/mdc-melanoma-tbdt)
![GitHub stars](https://img.shields.io/github/stars/giolucasd/mdc-melanoma-tbdt?style=social)
![GitHub forks](https://img.shields.io/github/forks/giolucasd/mdc-melanoma-tbdt?style=social)

Final project for Mineração de Dados Complexos course from group The Big Data Theory.

- [1. Prerequisites](#1-prerequisites)
- [2. Installing `mdc-melanoma-tbdt`](#2-installing-mdc-melanoma-tbdt)
- [3. Using `mdc-melanoma-tbdt`](#3-using-mdc-melanoma-tbdt)
  - [3.1. Downloading the data](#31-downloading-the-data)
  - [3.2. Training baseline](#32-training-baseline)
  - [3.3. Interactive analysis](#33-interactive-analysis)

## 1. Prerequisites

Before you begin, ensure you have met the following requirements:

* You have **[uv](https://github.com/astral-sh/uv)** installed (for dependency management and reproducibility).
* You have **Python 3.13+** installed.

Recommendations:
* Use **Linux**! The project was tested on Ubuntu 24.04.
* A CUDA-enabled GPU is strongly recommended for efficient pre-training and fine-tuning.

## 2. Installing `mdc-melanoma-tbdt`

Clone this repository and install dependencies using **uv**:

```bash
git clone https://github.com/giolucasd/mdc-melanoma-tbdt.git
cd mdc-melanoma-tbdt
uv sync
```

To include development dependencies (for reproducibility or debugging):

```bash
uv sync --all-extras
```

After installing the dependencies, activate the virtual environment created by **uv**:

```bash
source .venv/bin/activate
```

And install the project itself with:

```bash
uv pip install -e .
```

## 3. Using `mdc-melanoma-tbdt`

### 3.1. Downloading the data

First, be sure you are in the root directory.

Then go to your kaggle account, access your profile, settings and generate an API token.
Copy the token export command, paste it in your terminal to login into kaggle.

Now, go to the competition page (https://www.kaggle.com/competitions/classificacao-de-melanoma) and join the competition.

Finally, download the dataset into the expected structure with:

```bash
./scripts/download_data.sh
```

This script will automatically download the zip file into the "data/" directory, extract it and clean the temporary files generated.

### 3.2. Training baseline

After downloading the data, you can train the baseline CNN using the CLI script provided in `scripts/train.py`.

This script wraps the full training pipeline:
- Setup reproducibility and deterministic behavior
- Loads a YAML configuration (`configs/*.yaml`)
- Builds the baseline CNN (`src.model`)
- Instantiates the ModelTrainer (`src.train`)
- Saves checkpoints and training logs
- Stores a copy of the config in the output directory for reproducibility

> **NOTE**: This is the recommended approach for reproducible experiments.

**Configuration:**
All hyperparameters are centralized in `configs/baseline_cnn_no_augmentation.yaml`:
- Pure SimpleCNN (no data augmentation)
- Training: 30 epochs (5 warmup), batch_size=32, lr=0.01
- All training parameters organized for reproducibility

**Linux/Mac:**
```bash
python -m scripts.train \
  --config configs/baseline_cnn_no_augmentation.yaml \
  --output-suffix baseline_cnn_no_aug_v1
```

**Windows (PowerShell):**
```powershell
# Option 1: Direct command
python -m scripts.train --config configs/baseline_cnn_no_augmentation.yaml --output-suffix baseline_cnn_no_aug_v1

# Option 2: Use convenience script
.\train_baseline.ps1 baseline_v1
```

This will generate an output directory:

```
outputs/baseline_v1/
│
├── config.yaml               # copy of the config used
├── best.ckpt                 # best checkpoint (monitored by val_loss)
├── last.ckpt                 # last checkpoint
├── best_history.json         # training history for best checkpoint
├── last_history.json         # training history for last checkpoint
└── tensorboard/              # TensorBoard logs (if implemented)
```

To launch TensorBoard to monitor losses and balanced accuracy (when implemented):

```bash
tensorboard --logdir outputs/baseline_v1/tensorboard
```

Note that you can create multiple config files and reuse the same CLI. This will be even more helpful when using configurable models.

### 3.3. Interactive analysis

For interactive dataset analysis and model evaluation, use the main script:

```bash
python main.py
```

This provides two options:
1. **Analyze Dataset** - View dataset statistics, class distribution, and sample images
2. **Train Model** - Train a model interactively with GPU acceleration

When analyzing a trained model (Option 1 → Load saved model button), you'll see:
- Full model architecture and parameters
- Training curves (balanced accuracy per epoch)
- ROC and Precision-Recall curves
- Evaluation metrics

## Project Structure

```
mdc-melanoma-tbdt/
├── configs/              # YAML configuration files
│   └── baseline.yaml     # Baseline CNN config
├── data/                 # Dataset (gitignored)
│   ├── train.csv
│   ├── val.csv
│   └── images/
├── models/               # Saved model checkpoints (gitignored)
├── notebooks/            # Jupyter notebooks for exploration
│   └── data-exploring.ipynb
├── outputs/              # Training outputs (gitignored)
├── scripts/              # CLI scripts
│   ├── download_data.sh  # Download dataset
│   └── train.py          # Training script
├── src/                  # Source code
│   ├── __init__.py
│   ├── data.py           # Dataset and dataloaders
│   ├── model.py          # Model architectures
│   └── train.py          # Training logic
├── transforms/           # Data transformations
│   └── __init__.py
├── .gitignore
├── explore.md            # Data exploration notes
├── LICENSE
├── main.py               # Interactive main script
├── pyproject.toml        # Project dependencies
├── README.md
└── uv.lock              # Lockfile for reproducibility
```
