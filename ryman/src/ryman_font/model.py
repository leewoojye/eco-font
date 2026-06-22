from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class RymanNet(nn.Module):
    def __init__(self, input_channels: int = 7, base_channels: int = 16, depth: int = 3, dropout: float = 0.04) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(depth)]
        self.in_block = Block(input_channels, channels[0], dropout)
        self.downs = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), Block(channels[i], channels[i + 1], dropout)) for i in range(depth - 1)]
        )
        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.ups.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2),
                        "block": Block(channels[i] * 2, channels[i], dropout),
                    }
                )
            )
        self.out = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.in_block(x)]
        current = skips[0]
        for down in self.downs:
            current = down(current)
            skips.append(current)
        current = skips[-1]
        for up_block, skip in zip(self.ups, reversed(skips[:-1])):
            current = up_block["up"](current)
            if current.shape[-2:] != skip.shape[-2:]:
                current = F.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = up_block["block"](torch.cat([current, skip], dim=1))
        return self.out(current)


def build_model(config: dict) -> RymanNet:
    return RymanNet(
        input_channels=int(config.get("input_channels", 7)),
        base_channels=int(config.get("base_channels", 16)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.04)),
    )
