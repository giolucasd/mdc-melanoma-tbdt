from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A simple convolutional block.

    It has two conv layers, each followed by BatchNorm and GELU.
    It finishes with a MaxPool layer to reduce spatial dimensions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=stride),
        )

    def forward(self, x):
        return self.block(x)
