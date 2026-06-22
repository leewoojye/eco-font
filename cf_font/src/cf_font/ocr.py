from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import CFEcoDataset
from .priors import ECO_STYLES, make_target
from .train import device_from_name


def _torch_load(path: str | Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class OCRNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1))


def _augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = image.shape
    angle = rng.uniform(-7, 7)
    scale = rng.uniform(0.90, 1.08)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += [rng.uniform(-3, 3), rng.uniform(-3, 3)]
    out = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if rng.random() < 0.62:
        style = rng.choice(ECO_STYLES)
        target_saving = rng.uniform(0.35, 0.70)
        out, _score = make_target(out.astype(np.float32), style, target_saving)
    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out.astype(np.float32), (3, 3), 0)
    if rng.random() < 0.20:
        kernel = np.ones((2, 2), dtype=np.uint8)
        binary = (out > 0.15).astype(np.uint8)
        out = (cv2.dilate(binary, kernel, iterations=1) if rng.random() < 0.5 else cv2.erode(binary, kernel, iterations=1)).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class OCRDataset(Dataset):
    def __init__(self, glyphs: np.ndarray, samples_per_char: int, seed: int, augment: bool) -> None:
        self.glyphs = glyphs
        self.samples_per_char = int(samples_per_char)
        self.seed = int(seed)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return self.glyphs.shape[1] * self.samples_per_char

    def __getitem__(self, index: int):
        label = index % self.glyphs.shape[1]
        rng = random.Random(self.seed + index * 3571)
        font_index = rng.randrange(self.glyphs.shape[0])
        image = self.glyphs[font_index, label]
        if self.augment:
            image = _augment(image, rng)
        return torch.from_numpy(image[None].astype(np.float32)), torch.tensor(label, dtype=torch.long)


@dataclass(frozen=True)
class OCRConfig:
    dataset: Path
    out: Path
    samples_per_char: int = 32
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    device: str = "auto"
    seed: int = 29


def train_ocr(config: OCRConfig) -> dict[str, Any]:
    ds = CFEcoDataset(config.dataset)
    train_ds = OCRDataset(ds.glyphs, config.samples_per_char, config.seed, augment=True)
    val_ds = OCRDataset(ds.glyphs, 2, config.seed + 99_000, augment=False)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    device = device_from_name(config.device)
    model = OCRNet(len(ds.chars)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history = []
    best_acc = -1.0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"ocr {epoch}/{config.epochs}", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device)).argmax(dim=1).cpu()
                correct += int((pred == y).sum())
                total += int(y.numel())
        acc = correct / max(1, total)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": float(acc)}
        print(json.dumps(row, ensure_ascii=False))
        history.append(row)
        if acc >= best_acc:
            best_acc = acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    config.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "chars": ds.chars,
            "fonts": ds.fonts,
            "image_size": ds.image_size,
            "history": history,
            "best_val_acc": best_acc,
        },
        config.out,
    )
    summary = {
        "checkpoint": str(config.out),
        "char_count": len(ds.chars),
        "font_count": len(ds.fonts),
        "best_val_acc": best_acc,
        "device": str(device),
    }
    (config.out.parent / "ocr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class OCREvaluator:
    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.device = device_from_name(device)
        data = _torch_load(checkpoint, map_location=self.device)
        self.chars: list[str] = data["chars"]
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.model = OCRNet(len(self.chars)).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

    def score(self, images: list[np.ndarray], chars: list[str]) -> list[dict[str, Any]]:
        x = torch.from_numpy(np.stack([img[None].astype(np.float32) for img in images])).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()
        rows = []
        for prob, ch in zip(probs, chars, strict=True):
            pred = int(np.argmax(prob))
            target = self.char_to_idx.get(ch)
            conf = float(prob[target]) if target is not None else 0.0
            rows.append(
                {
                    "ocr_text": self.chars[pred],
                    "ocr_confidence": conf,
                    "ocr_pred_confidence": float(prob[pred]),
                    "ocr_match": bool(pred == target),
                }
            )
        return rows
