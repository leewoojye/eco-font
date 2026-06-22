from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class EcoMaskUNet(nn.Module):
    """Local-feature U-Net for glyph-to-eco-mask prediction.

    The input channels are glyph, distance transform, skeleton, target saving,
    x-coordinate, and y-coordinate. This is the local-preservation piece that
    makes the model closer to LF-style font generation than plain segmentation.
    """

    def __init__(
        self,
        input_channels: int = 6,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        channels = [base_channels * (2**i) for i in range(depth)]
        self.input_block = ConvBlock(input_channels, channels[0], dropout)
        self.downs = nn.ModuleList(
            [Down(channels[i], channels[i + 1], dropout) for i in range(depth - 1)]
        )
        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.ups.append(Up(channels[i + 1], channels[i], channels[i], dropout))
        self.out = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.input_block(x)]
        current = skips[0]
        for down in self.downs:
            current = down(current)
            skips.append(current)
        current = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            current = up(current, skip)
        return self.out(current)


def build_model(config: dict) -> EcoMaskUNet:
    return EcoMaskUNet(
        input_channels=int(config.get("input_channels", 6)),
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 4)),
        dropout=float(config.get("dropout", 0.05)),
    )
