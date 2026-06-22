from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def edge_protection_loss(logits: torch.Tensor, glyph: torch.Tensor) -> torch.Tensor:
    """Penalize predicted cuts on thin/outer glyph edges."""
    probs = torch.sigmoid(logits)
    pooled = F.avg_pool2d(glyph, kernel_size=5, stride=1, padding=2)
    edge_band = ((glyph > 0.05) & (pooled < 0.95)).float()
    return (probs * edge_band).mean()


class EcoMaskLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        target_saving_weight: float = 0.35,
        edge_protection_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.target_saving_weight = target_saving_weight
        self.edge_protection_weight = edge_protection_weight

    def forward(
        self,
        logits: torch.Tensor,
        target_mask: torch.Tensor,
        glyph: torch.Tensor,
        target_saving: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        bce = F.binary_cross_entropy_with_logits(logits, target_mask)
        dice = dice_loss_from_logits(logits, target_mask)
        probs = torch.sigmoid(logits) * (glyph > 0.05).float()
        glyph_area = glyph.flatten(1).sum(dim=1).clamp_min(1e-6)
        predicted_saving = probs.flatten(1).sum(dim=1) / glyph_area
        saving_loss = F.l1_loss(predicted_saving, target_saving)
        edge_loss = edge_protection_loss(logits, glyph)
        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.target_saving_weight * saving_loss
            + self.edge_protection_weight * edge_loss
        )
        parts = {
            "loss": float(total.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "dice": float(dice.detach().cpu()),
            "saving_l1": float(saving_loss.detach().cpu()),
            "edge": float(edge_loss.detach().cpu()),
            "predicted_saving": float(predicted_saving.mean().detach().cpu()),
        }
        return total, parts
