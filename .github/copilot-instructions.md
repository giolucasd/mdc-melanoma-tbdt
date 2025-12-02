# Copilot/AI Agent Instructions for `mdc-melanoma-tbdt`

## Project Overview
- **Purpose:** Melanoma skin cancer image classification for the Mineração de Dados Complexos course.
- **Main components:**
  - `src/data.py`: Data loading, augmentation, and PyTorch `Dataset`/`DataLoader` logic for the challenge dataset.
  - `main.py`: Entry point (currently a stub).
  - `notebooks/`: Data exploration and visualization (see `data-exploring.ipynb`).
  - `scripts/download_data.sh`: Downloads and unpacks the Kaggle competition data.

## Data & Structure
- **Expected data layout:**
  - `data/`
    - `images/`
      - `train/`, `val/`, `test/`
    - `train.csv`, `val.csv`, `sample_submission.csv`
- **Data download:** Use `./scripts/download_data.sh` (requires Kaggle API credentials and competition join).

## Development Workflow
- **Dependency management:** Use [`uv`](https://github.com/astral-sh/uv) for reproducible installs:
  - `uv sync` (main dependencies)
  - `uv sync --all-extras` (with dev dependencies)
  - Activate with `source .venv/bin/activate` (Linux) or equivalent for your OS.
- **Python version:** 3.13+
- **Recommended OS:** Linux (tested on Ubuntu 24.04); CUDA GPU recommended for training.

## Coding Patterns & Conventions
- **Data pipeline:**
  - Use `get_train_val_dataloaders` and `get_test_dataloader` from `src/data.py` for standardized data loading.
  - All image normalization uses ImageNet statistics (`IMAGENET_MEAN`, `IMAGENET_STD`).
  - Data augmentation is applied only to training data.
- **Notebooks:**
  - Add `src/` to `sys.path` for imports.
  - Use provided utility functions for visualization (see `data-exploring.ipynb`).
- **PyTorch:**
  - All data loading and transforms are handled in `src/data.py`.
  - Use the `MelanomaDataset` class for custom dataset logic.

## Integration & Extensibility
- **Add new models or training logic** in new modules under `src/`.
- **Keep data access and augmentation logic in `src/data.py`** for consistency.
- **Scripts** should be placed in `scripts/` and be executable from the project root.

## Example: Loading Data
```python
from src.data import get_train_val_dataloaders
train_loader, val_loader = get_train_val_dataloaders(batch_size=32, num_workers=4)
```

## Additional Notes
- Do not hardcode paths; use `pathlib.Path` and project-relative paths.
- Follow the structure and conventions in `src/data.py` for any new data-related code.
- For questions about project setup, see `README.md`.
