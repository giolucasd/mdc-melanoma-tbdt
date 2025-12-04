from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_PATH = Path(__file__).parent.parent / "data"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class MelanomaDataset(Dataset):
    """
    Melanoma skin cancer classification dataset for MDC course.

    Expects a CSV with columns for path and target.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Path | str,
        transform=None,
        path_column: str = "path",
        target_column: str = "target",
        test_mode: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.transform = transform

        self.path_column = path_column
        self.target_column = target_column

        self.test_mode = test_mode

        assert pd.api.types.is_integer_dtype(self.df[self.target_column])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.images_dir / row[self.path_column]
        label = int(row[self.target_column])

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.test_mode:
            return img, row[self.path_column]

        return img, label


def build_transforms(cfg):
    aug_list = []

    # --- CROP ---
    crop_cfg = cfg.get("crop", {})
    if crop_cfg.get("enabled", False):
        aug_list.append(
            transforms.RandomResizedCrop(
                crop_cfg.get("size", 224),
                scale=tuple(crop_cfg.get("scale", [0.8, 1.0])),
                ratio=tuple(crop_cfg.get("ratio", [0.9, 1.1])),
            )
        )

    # --- FLIP ---
    flip_cfg = cfg.get("flip", {})
    if flip_cfg.get("enabled", False):
        if flip_cfg.get("horizontal", True):
            aug_list.append(transforms.RandomHorizontalFlip())
        if flip_cfg.get("vertical", False):
            aug_list.append(transforms.RandomVerticalFlip())

    # --- ROTATE ---
    rot_cfg = cfg.get("rotate", {})
    if rot_cfg.get("enabled", False):
        aug_list.append(
            transforms.RandomRotation(rot_cfg.get("degrees", 20))
        )

    # --- AFFINE ---
    aff_cfg = cfg.get("affine", {})
    if aff_cfg.get("enabled", False):
        aug_list.append(
            transforms.RandomAffine(
                degrees=aff_cfg.get("degrees", 0),
                translate=tuple(aff_cfg.get("translate", [0.05, 0.05])),
                scale=tuple(aff_cfg.get("scale", [0.95, 1.05])),
                shear=aff_cfg.get("shear", 5),
            )
        )

    # --- COLOR ---
    col_cfg = cfg.get("color", {})
    if col_cfg.get("enabled", False):
        aug_list.append(
            transforms.ColorJitter(
                brightness=col_cfg.get("brightness", 0.1),
                contrast=col_cfg.get("contrast", 0.1),
                saturation=col_cfg.get("saturation", 0.05),
                hue=col_cfg.get("hue", 0.02),
            )
        )

    # Always convert to tensor
    aug_list.append(transforms.ToTensor())

    # --- NORMALIZE ---
    norm_cfg = cfg.get("normalize", {})
    aug_list.append(
        transforms.Normalize(
            mean=norm_cfg.get("mean", IMAGENET_MEAN),
            std=norm_cfg.get("std", IMAGENET_STD),
        )
    )

    # --- CUTOUT ---
    cut_cfg = cfg.get("cutout", {})
    if cut_cfg.get("enabled", False):
        aug_list.append(
            transforms.RandomErasing(
                p=cut_cfg.get("p", 0.25),
                scale=tuple(cut_cfg.get("scale", [0.02, 0.10])),
                ratio=tuple(cut_cfg.get("ratio", [0.3, 3.3])),
            )
        )

    return transforms.Compose(aug_list)


def get_train_test_transforms(aug_cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms for train and test.

    In train images, we explore data augmentation strategies.
    The augmentations are though to match the ISIC dataset characteristics.
    In test and validation images we only resize and normalize.

    Normalization is conducted according to ImageNet statistics.
    This choice allows us to explore transfer learning.
    """
    train_transforms = build_transforms(aug_cfg)

    norm_cfg = aug_cfg.get("normalize", {})
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm_cfg.get("mean", IMAGENET_MEAN), 
                std=norm_cfg.get("std", IMAGENET_STD)),
        ]
    )

    return train_transforms, test_transforms


def get_train_val_dataloaders(
    aug_cfg: dict,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders.

    Expects the MDC melanoma classification challenge data structure:
        data/
        ├─ images/
        │   └─ images/
        │       ├─ train/
        │       └─ val/
        ├─ train.csv
        └─ val.csv
    """
    train_csv = DATA_PATH / "train.csv"
    val_csv = DATA_PATH / "val.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_transforms, test_transforms = get_train_test_transforms(aug_cfg=aug_cfg)

    root = DATA_PATH

    train_ds = MelanomaDataset(train_df, root, transform=train_transforms)
    val_ds = MelanomaDataset(val_df, root, transform=test_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(
    aug_cfg: dict,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get test dataloader.

    Expects the MDC melanoma classification challenge data structure:
        data/
        ├─ images/
        │   └─ images/
        │       └─ test/
        └─ sample_submission.csv
    """
    test_csv = DATA_PATH / "sample_submission.csv"

    test_df = pd.read_csv(test_csv)

    _, test_transforms = get_train_test_transforms(aug_cfg=aug_cfg)

    root = DATA_PATH

    test_ds = MelanomaDataset(
        test_df,
        root,
        transform=test_transforms,
        path_column="ID",
        target_column="TARGET",
        test_mode=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
