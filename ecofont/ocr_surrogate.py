"""Local OCR-surrogate recognizer for OCR-guided eco-mask search."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .font_io import filter_supported_chars
from .render import render_glyph
from .text_presets import characters_for_language
from .train import resolve_device


class GlyphOCRNet(nn.Module):
    """Small CNN classifier used as a local OCR confidence model."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))


def _random_eco_cut(image: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = image.shape
    yy, xx = np.mgrid[:h, :w]
    mask = np.ones_like(image, dtype=np.float32)

    if rng.random() < 0.45:
        spacing = rng.uniform(7.0, 14.0)
        radius = rng.uniform(0.8, 2.2)
        ox = rng.uniform(0.0, spacing)
        oy = rng.uniform(0.0, spacing)
        gx = np.mod(xx - ox + spacing / 2.0, spacing) - spacing / 2.0
        gy = np.mod(yy - oy + spacing / 2.0, spacing) - spacing / 2.0
        mask[(gx * gx + gy * gy) <= radius * radius] = 0.0

    if rng.random() < 0.45:
        theta = np.deg2rad(rng.choice([0.0, 45.0, 90.0, 135.0]))
        spacing = rng.uniform(7.0, 14.0)
        width = rng.uniform(0.8, 1.8)
        coord = xx * np.cos(theta) + yy * np.sin(theta)
        phase = np.mod(coord + rng.uniform(0.0, spacing), spacing)
        mask[(phase < width) | (phase > spacing - width)] = 0.0

    if rng.random() < 0.25:
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = (image > 0.2).astype(np.uint8)
        edge = binary - cv2.erode(binary, kernel, iterations=1)
        mask[edge > 0] = rng.choice([0.0, 1.0])

    return np.clip(image * mask, 0.0, 1.0)


def augment_foreground(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply OCR-style synthetic variation to one glyph foreground."""
    h, w = image.shape
    angle = rng.uniform(-7.0, 7.0)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-3.0, 3.0)
    ty = rng.uniform(-3.0, 3.0)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    aug = cv2.warpAffine(
        image.astype(np.float32),
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    if rng.random() < 0.5:
        aug = _random_eco_cut(aug, rng)
    if rng.random() < 0.25:
        kernel = np.ones((2, 2), dtype=np.uint8)
        binary = (aug > 0.2).astype(np.uint8)
        if rng.random() < 0.5:
            aug = cv2.erode(binary, kernel, iterations=1).astype(np.float32)
        else:
            aug = cv2.dilate(binary, kernel, iterations=1).astype(np.float32)
    if rng.random() < 0.2:
        aug = cv2.GaussianBlur(aug, (3, 3), 0)

    noise = np.asarray(rng.choices([-0.03, 0.0, 0.03], weights=[1, 12, 1], k=h * w), dtype=np.float32)
    aug = aug + noise.reshape(h, w)
    return np.clip(aug, 0.0, 1.0).astype(np.float32)


class SyntheticGlyphOCRDataset(Dataset):
    def __init__(self, bases: list[np.ndarray], samples_per_class: int, seed: int, augment: bool) -> None:
        self.bases = bases
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return len(self.bases) * self.samples_per_class

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = index % len(self.bases)
        image = self.bases[label]
        if self.augment:
            rng = random.Random(self.seed + index * 7919)
            image = augment_foreground(image, rng)
        x = torch.from_numpy(image[None, :, :].astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)
        return x, y


@dataclass(frozen=True)
class OCRTrainConfig:
    font: Path
    output: Path
    language: str = "chr"
    text: str | None = None
    image_size: int = 96
    samples_per_char: int = 32
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-3
    device: str = "auto"
    seed: int = 7


def train_ocr_surrogate(config: OCRTrainConfig) -> dict:
    """Train and save a local glyph recognizer."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    requested = characters_for_language(config.language, config.text)
    chars, missing = filter_supported_chars(config.font, requested)
    if len(chars) < 2:
        raise ValueError("OCR surrogate needs at least two supported characters")

    bases = [render_glyph(config.font, ch, image_size=config.image_size).foreground for ch in chars]
    train_ds = SyntheticGlyphOCRDataset(bases, config.samples_per_char, config.seed, augment=True)
    val_ds = SyntheticGlyphOCRDataset(bases, 2, config.seed + 100_000, augment=False)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    device = resolve_device(config.device)
    model = GlyphOCRNet(num_classes=len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    history: list[dict[str, float]] = []
    best_acc = -1.0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in tqdm(train_loader, desc=f"ocr epoch {epoch}/{config.epochs}", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        val_acc = correct / max(1, total)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": float(val_acc)}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    config.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "chars": chars,
        "missing": missing,
        "image_size": config.image_size,
        "font": str(config.font),
        "history": history,
        "best_val_acc": best_acc,
    }
    torch.save(checkpoint, config.output)
    summary = {
        "checkpoint": str(config.output),
        "char_count": len(chars),
        "missing_count": len(missing),
        "best_val_acc": best_acc,
        "device": str(device),
    }
    (config.output.parent / "ocr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


class OCREvaluator:
    """Batch scorer for OCR-guided search."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        data = torch.load(checkpoint, map_location=self.device)
        self.chars: list[str] = data["chars"]
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.image_size: int = int(data["image_size"])
        self.model = GlyphOCRNet(num_classes=len(self.chars)).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

    def score_batch(self, images: list[np.ndarray], chars: list[str], batch_size: int = 64) -> list[dict[str, float | str | bool]]:
        if len(images) != len(chars):
            raise ValueError("images and chars must have equal length")
        results: list[dict[str, float | str | bool]] = []
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch_chars = chars[start : start + batch_size]
            x = torch.from_numpy(np.stack([img[None, :, :].astype(np.float32) for img in batch_images])).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1).cpu().numpy()
            for prob, ch in zip(probs, batch_chars, strict=True):
                target_idx = self.char_to_idx.get(ch)
                pred_idx = int(np.argmax(prob))
                target_prob = float(prob[target_idx]) if target_idx is not None else 0.0
                results.append(
                    {
                        "ocr_confidence": target_prob,
                        "ocr_pred": self.chars[pred_idx],
                        "ocr_pred_confidence": float(prob[pred_idx]),
                        "ocr_correct": bool(target_idx == pred_idx),
                        "ocr_loss": float(-np.log(max(target_prob, 1e-8))),
                    }
                )
        return results
