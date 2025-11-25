from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


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


class ResidualBlock(nn.Module):
    """
    Basic residual block.

    Optionally adjusts channel dimensions and spatial size using a 1x1 conv.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.gelu(out)
