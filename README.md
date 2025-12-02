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

Now, go to the competition page (https://www.kaggle.com/competitions/classificacao-de-melanoma) and join the conpetition.

Finally, download the dataset into the expected structure with:

```bash
./scripts/download_data.sh
```

This script will automatically download the zip file into the "data/" directory, extract it and clean the temporary files generated.

### 3.2. Training baseline

After downloading the data, you can train the baseline CNN using the CLI script provided in `scripts/baseline_train`.

This script wraps the full training pipeline:
- setup reproducibility and deterministic behavior
- loads a YAML configuration (configs/*.yaml)
- builds the baseline CNN (src.models.baseline)
- instantiates the LightningModule (src.training)
- configures TensorBoard logging
- saves checkpoints and training logs
- stores a copy of the config in the output directory for reproducibility

> NOTE: other scripts will also do that, but we won't keep repeting it.

```bash
python -m scripts.train \
  --config configs/baseline.yaml \
  --output-suffix baseline_v1
```

This will generate an output directory:

```bash
outputs/baseline_v1/
│
├── config.yaml               # copy of the config used
├── best.ckpt                 # best checkpoint (monitored by val_loss)
├── last.ckpt                 # last checkpoint
└── tensorboard/              # TensorBoard logs
```

To launch TensorBoard to monitor losses and balanced accuracy:

```bash
tensorboard --logdir outputs/baseline_v1/tensorboard
```

Note that you can create multiple config files and reuse the same CLI. This will be even more helpful when using configurable models.
