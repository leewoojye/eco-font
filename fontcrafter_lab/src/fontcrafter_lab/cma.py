from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ContextAwareMaskAdapter(nn.Module):
    """Disclosed FontCrafter CMA block.

    The paper describes CMA as a lightweight adapter inserted after MM-DiT
    blocks. It concatenates downsampled glyph mask features with block features,
    reduces channels to 64 with a linear layer, applies GELU, and projects back
    to the block feature dimension.

    This module is provided for checkpoint-compatible experiments. The official
    learned weights are not bundled here because the authors have not released
    them in the checked public materials.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(feature_dim + 1, hidden_dim)
        self.up = nn.Linear(hidden_dim, feature_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must be [batch, tokens, channels]")
        if mask.ndim == 4:
            pooled = F.adaptive_avg_pool2d(mask.float(), (1, features.shape[1])).flatten(2).transpose(1, 2)
        elif mask.ndim == 3:
            pooled = mask.float().unsqueeze(-1)
        else:
            raise ValueError("mask must be [batch, tokens] or [batch, 1, h, w]")
        if pooled.shape[1] != features.shape[1]:
            pooled = F.interpolate(pooled.transpose(1, 2), size=features.shape[1], mode="linear").transpose(1, 2)
        control = self.up(F.gelu(self.down(torch.cat([features, pooled[..., :1]], dim=-1))))
        return features + control
