import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.models.blocks import ConvBlock


class MelanomaBaselineCNN(BaseModel):
    """A reasonable baseline CNN for ISIC binary classification."""

    def __init__(self, config: dict = None):
        super().__init__()

        number_of_classes = config.get("num_classes", 1)

        self.features = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, kernel_size=5),
            ConvBlock(in_channels=32, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
