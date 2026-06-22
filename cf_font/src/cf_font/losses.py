from __future__ import annotations

import torch
from torch.nn import functional as F


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (prob * target).flatten(1).sum(dim=1)
    denom = prob.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def total_variation(prob: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).mean()
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).mean()
    return dx + dy


def _projection_indices(height: int, width: int, device: torch.device) -> list[tuple[torch.Tensor, int]]:
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return [
        (yy.reshape(-1), height),
        (xx.reshape(-1), width),
        ((yy + xx).reshape(-1), height + width - 1),
        ((yy - xx + width - 1).reshape(-1), height + width - 1),
    ]


def _project_distribution(image: torch.Tensor, index: torch.Tensor, bins: int, eps: float) -> torch.Tensor:
    values = image[:, 0].reshape(image.size(0), -1)
    hist = torch.zeros((image.size(0), bins), dtype=image.dtype, device=image.device)
    hist.scatter_add_(1, index[None, :].expand(image.size(0), -1), values)
    return (hist + eps) / (hist.sum(dim=1, keepdim=True) + eps * bins)


def projected_character_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    mode: str = "wasserstein",
) -> torch.Tensor:
    """Projected Character Loss from CF-Font using 1D marginal distributions."""
    if pred.shape != target.shape:
        raise ValueError(f"PCL shape mismatch: {pred.shape} vs {target.shape}")
    height, width = pred.shape[-2:]
    losses = []
    for index, bins in _projection_indices(height, width, pred.device):
        p = _project_distribution(pred.clamp_min(0.0), index, bins, eps)
        q = _project_distribution(target.clamp_min(0.0), index, bins, eps)
        if mode == "kl":
            losses.append((q * (q.log() - p.log())).sum(dim=1).mean())
        elif mode == "wasserstein":
            losses.append(torch.abs(torch.cumsum(p, dim=1) - torch.cumsum(q, dim=1)).mean())
        else:
            raise ValueError(f"Unknown PCL mode: {mode}")
    return torch.stack(losses).mean()


def generator_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    glyph: torch.Tensor,
    target_saving: torch.Tensor,
    skeleton: torch.Tensor,
    pcl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    keep_prob = torch.sigmoid(logits) * (glyph > 0.05).float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(keep_prob, target)
    pcl = projected_character_loss(keep_prob, target)
    glyph_area = glyph.flatten(1).sum(dim=1).clamp_min(1e-6)
    keep_ratio = keep_prob.flatten(1).sum(dim=1) / glyph_area
    pred_saving = 1.0 - keep_ratio
    ink = F.l1_loss(pred_saving, target_saving)
    skel = ((1.0 - keep_prob) * skeleton).mean()
    smooth = total_variation(keep_prob)
    loss = bce + 0.75 * dice + float(pcl_weight) * pcl + 0.75 * ink + 0.22 * skel + 0.035 * smooth
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "pcl": float(pcl.detach().cpu()),
        "ink_l1": float(ink.detach().cpu()),
        "pred_saving": float(pred_saving.mean().detach().cpu()),
    }
