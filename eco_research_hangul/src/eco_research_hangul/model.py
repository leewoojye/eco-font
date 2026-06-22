from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(1, half - 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = TimeResBlock(out_channels, out_channels, time_dim, dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x), time_emb)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = TimeResBlock(out_channels + skip_channels, out_channels, time_dim, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1), time_emb)


class ConditionalGlyphDDPM(nn.Module):
    def __init__(
        self,
        condition_channels: int = 4,
        base_channels: int = 32,
        depth: int = 3,
        dropout: float = 0.05,
        time_dim: int | None = None,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.condition_channels = condition_channels
        self.time_dim = time_dim or base_channels * 4
        channels = [base_channels * (2**i) for i in range(depth)]
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.input_block = TimeResBlock(1 + condition_channels, channels[0], self.time_dim, dropout)
        self.downs = nn.ModuleList(
            [DownBlock(channels[i], channels[i + 1], self.time_dim, dropout) for i in range(depth - 1)]
        )
        self.mid = TimeResBlock(channels[-1], channels[-1], self.time_dim, dropout)
        self.ups = nn.ModuleList(
            [
                UpBlock(channels[i + 1], channels[i], channels[i], self.time_dim, dropout)
                for i in reversed(range(depth - 1))
            ]
        )
        self.out_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.out = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, noisy_target: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(sinusoidal_embedding(timesteps, self.time_dim))
        x = torch.cat([noisy_target, condition], dim=1)
        current = self.input_block(x, time_emb)
        skips = [current]
        for down in self.downs:
            current = down(current, time_emb)
            skips.append(current)
        current = self.mid(skips[-1], time_emb)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            current = up(current, skip, time_emb)
        return self.out(F.silu(self.out_norm(current)))


def build_model(config: dict) -> ConditionalGlyphDDPM:
    return ConditionalGlyphDDPM(
        condition_channels=int(config.get("condition_channels", 4)),
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.05)),
        time_dim=int(config["time_dim"]) if config.get("time_dim") else None,
    )

