from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


def total_variation(prob: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).mean()
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).mean()
    return dx + dy


class RymanLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 0.8,
        ink_weight: float = 0.65,
        skeleton_weight: float = 0.0,
        smooth_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ink_weight = ink_weight
        self.skeleton_weight = skeleton_weight
        self.smooth_weight = smooth_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, glyph: torch.Tensor, target_saving: torch.Tensor, skeleton: torch.Tensor) -> tuple[torch.Tensor, dict]:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = dice_loss(logits, target)
        prob = torch.sigmoid(logits)
        glyph_area = glyph.flatten(1).sum(dim=1).clamp_min(1e-6)
        keep_ratio = prob.flatten(1).sum(dim=1) / glyph_area
        predicted_saving = 1.0 - keep_ratio
        ink = F.l1_loss(predicted_saving, target_saving)
        skeleton_penalty = ((1.0 - prob) * skeleton).mean()
        smooth = total_variation(prob)
        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.ink_weight * ink
            + self.skeleton_weight * skeleton_penalty
            + self.smooth_weight * smooth
        )
        return total, {
            "loss": float(total.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "dice": float(dice.detach().cpu()),
            "ink_l1": float(ink.detach().cpu()),
            "predicted_saving": float(predicted_saving.mean().detach().cpu()),
        }
