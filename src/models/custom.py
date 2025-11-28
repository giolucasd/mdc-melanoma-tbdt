import torch.nn as nn

from src.models.base import BaseModel
from src.models.blocks import ResidualBlock


class CustomResNet(BaseModel):
    """
    A lightweight custom ResNet optimized for small datasets (e.g., melanoma classification).
    - Uses Basic Residual Blocks (ResNet-18 style)
    - Configurable stem kernel size (3x3, 5x5, 7x7)
    - Much lighter than full ResNet
    """

    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__(config)
        stem_out = config.get("stem_out", 32)
        stem_kernel_size = config.get("stem_kernel_size", 7)
        block_channels = config.get("block_channels", (32, 64, 128, 256))
        blocks_per_stage = config.get("blocks_per_stage", (2, 2, 2, 1))
        number_of_classes = config.get("num_classes", 1)

        padding = stem_kernel_size // 2

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                stem_out,
                kernel_size=stem_kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
        )

        in_channels = stem_out
        stages = []

        for out_channels, num_blocks in zip(block_channels, blocks_per_stage):
            blocks = []
            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 and in_channels != out_channels else 1
                blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.ModuleList(stages)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, number_of_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)
