from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocalExpertBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_styles: int, num_experts: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Embedding(num_styles, num_experts)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(style_id), dim=1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (outputs * weights[:, :, None, None, None]).sum(dim=1)


class HybridEcoNet(nn.Module):
    def __init__(self, input_channels: int = 8, base_channels: int = 24, num_styles: int = 4, num_experts: int = 4) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = LocalExpertBlock(input_channels, c, num_styles, num_experts)
        self.enc2 = LocalExpertBlock(c, c * 2, num_styles, num_experts)
        self.enc3 = LocalExpertBlock(c * 2, c * 4, num_styles, num_experts)
        self.mid = LocalExpertBlock(c * 4, c * 6, num_styles, num_experts)
        self.up3 = nn.ConvTranspose2d(c * 6, c * 4, kernel_size=2, stride=2)
        self.dec3 = LocalExpertBlock(c * 8, c * 4, num_styles, num_experts)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = LocalExpertBlock(c * 4, c * 2, num_styles, num_experts)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = LocalExpertBlock(c * 2, c, num_styles, num_experts)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x, style_id)
        e2 = self.enc2(F.max_pool2d(e1, 2), style_id)
        e3 = self.enc3(F.max_pool2d(e2, 2), style_id)
        mid = self.mid(F.max_pool2d(e3, 2), style_id)
        d3 = _pad_to(self.up3(mid), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), style_id)
        d2 = _pad_to(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), style_id)
        d1 = _pad_to(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), style_id)
        return self.head(d1)


def _pad_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    dy = ref.size(2) - x.size(2)
    dx = ref.size(3) - x.size(3)
    if dy == 0 and dx == 0:
        return x
    return F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).flatten(1).sum(dim=1)
    denom = probs.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def total_variation(prob: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).mean()
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).mean()
    return dx + dy


def generator_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    glyph: torch.Tensor,
    target_saving: torch.Tensor,
    skeleton: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target)
    prob = torch.sigmoid(logits) * (glyph > 0.05).float()
    glyph_area = glyph.flatten(1).sum(dim=1).clamp_min(1e-6)
    keep_ratio = prob.flatten(1).sum(dim=1) / glyph_area
    pred_saving = 1.0 - keep_ratio
    ink = F.l1_loss(pred_saving, target_saving)
    skel = ((1.0 - prob) * skeleton).mean()
    smooth = total_variation(prob)
    loss = bce + 0.8 * dice + 0.75 * ink + 0.25 * skel + 0.04 * smooth
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "ink_l1": float(ink.detach().cpu()),
        "pred_saving": float(pred_saving.mean().detach().cpu()),
    }
