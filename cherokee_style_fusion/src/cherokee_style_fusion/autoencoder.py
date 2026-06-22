from __future__ import annotations

import torch
from torch import nn


class TinyStyleFusionAutoEncoder(nn.Module):
    """Optional baseline for future learned fusion experiments.

    Input channels are expected to be:
    0. content glyph
    1. style glyph
    2. target ink-saving scalar map

    The deterministic fusion pipeline is the default path because Cherokee
    public font data is small. This model is included so the folder has a
    trainable baseline without changing the main experiment design.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4, channels * 2),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
