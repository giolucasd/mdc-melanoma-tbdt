# Melanoma Classification <!-- omit from toc -->

![GitHub repo size](https://img.shields.io/github/repo-size/giolucasd/mdc-melanoma-tbdt)
![GitHub contributors](https://img.shields.io/github/contributors/giolucasd/mdc-melanoma-tbdt)
![GitHub stars](https://img.shields.io/github/stars/giolucasd/mdc-melanoma-tbdt?style=social)
![GitHub forks](https://img.shields.io/github/forks/giolucasd/mdc-melanoma-tbdt?style=social)

Final project for Mineração de Dados Complexos course.

- [1. Prerequisites](#1-prerequisites)
- [2. Installing `mdc-melanoma-tbdt`](#2-installing-mdc-melanoma-tbdt)
- [3. Using `mdc-melanoma-tbdt`](#3-using-mdc-melanoma-tbdt)

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

## 3. Using `mdc-melanoma-tbdt`

Be sure you are in the root directory and download the dataset:

```bash
./scripts/download_data.sh
```
