import argparse
import shutil
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.utils import setup_reproducibility, shut_down_warnings
from src.data import get_train_val_dataloaders
from src.models.baseline import MelanomaBaselineCNN
from src.training import MelanomaLitModule

setup_reproducibility(seed=27)
shut_down_warnings()


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Melanoma Classification Training CLI")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        required=True,
        help="Folder name suffix inside outputs/",
    )

    args = parser.parse_args()

    # ----------------------------------------------------
    # Load config
    # ----------------------------------------------------
    config_path = Path(args.config)
    config = load_config(config_path)

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    # ----------------------------------------------------
    # Prepare output directory
    # ----------------------------------------------------
    out_dir = Path("outputs") / args.output_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, out_dir / "config.yaml")

    # ----------------------------------------------------
    # Data
    # ----------------------------------------------------
    train_loader, val_loader = get_train_val_dataloaders(
        batch_size=data_cfg.get("batch_size", 256),
        num_workers=data_cfg.get("num_workers", 4),
    )

    # ----------------------------------------------------
    # Model + Lightning module
    # ----------------------------------------------------
    model = MelanomaBaselineCNN(num_classes=1)
    lit_model = MelanomaLitModule(model, training_cfg)

    # ----------------------------------------------------
    # Loggers & Checkpoints
    # ----------------------------------------------------
    logger = TensorBoardLogger(save_dir=out_dir, name="tensorboard")

    ckpt_best = ModelCheckpoint(
        dirpath=out_dir,
        filename="best",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    ckpt_last = ModelCheckpoint(
        dirpath=out_dir,
        filename="last",
        save_top_k=1,
        every_n_epochs=1,
    )

    # ----------------------------------------------------
    # Train
    # ----------------------------------------------------
    trainer = Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator="auto",
        devices=1,
        callbacks=[ckpt_best, ckpt_last],
        logger=logger,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
