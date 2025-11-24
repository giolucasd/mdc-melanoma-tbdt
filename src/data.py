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
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.transform = transform

        self.path_column = path_column
        self.target_column = target_column

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

        return img, label


def get_train_test_transforms():
    """
    Get transforms for train and test.

    In train images,we explore data augmentation strategies.
    In test and validation images we only resize and normalize.

    Normalization is conducted according to ImageNet statistics.
    This choice allows us to explore transfer learning.
    """
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transforms, test_transforms


def get_train_val_dataloaders(
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

    train_transforms, test_transforms = get_train_test_transforms()

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

    _, test_transforms = get_train_test_transforms()

    root = DATA_PATH

    test_ds = MelanomaDataset(
        test_df,
        root,
        transform=test_transforms,
        path_column="ID",
        target_column="TARGET",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
