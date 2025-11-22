from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_PATH = Path(__file__).parent.parent / "data"


class ISICDataset(Dataset):
    """
    ISIC 2019 Classification training dataset.

    Images resized to 224x224 for Melanoma Skin Cancer Detection.
    Use the scripts/download_data.sh script for compatibility.
    Made available in kaggle:
    https://www.kaggle.com/datasets/nischaydnk/isic-2019-jpg-224x224-resized/data
    """

    def __init__(self, df: pd.DataFrame, root_dir: Path | str, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not pd.api.types.is_integer_dtype(self.df["label"]):
            classes = sorted(self.df["label"].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.df["label"] = self.df["label"].map(self.class_to_idx)
        else:
            self.class_to_idx = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["isic_id"] + ".jpg"
        img_path = self.root_dir / filename

        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_train_test_transforms():
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, test_transforms


def get_train_val_test_split(df: pd.DataFrame, val_size: float, test_size: float):
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        shuffle=True,
    )

    return train_df, val_df, test_df


def get_train_val_test_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    metadata_path = DATA_PATH / "train-metadata.csv"
    img_dir = DATA_PATH / "train-image"

    df = pd.read_csv(metadata_path)

    train_df, val_df, test_df = get_train_val_test_split(df, val_size, test_size)

    train_transforms, test_transforms = get_train_test_transforms()

    train_ds = ISICDataset(train_df, img_dir, transform=train_transforms)
    val_ds = ISICDataset(val_df, img_dir, transform=test_transforms)
    test_ds = ISICDataset(test_df, img_dir, transform=test_transforms)

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
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
