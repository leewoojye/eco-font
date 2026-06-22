"""Model training loop."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import GlyphMaskDataset
from .model import EcoMaskUNet, mask_loss


def _collate(batch):
    xs, ys, rows = zip(*batch, strict=True)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), list(rows)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_model(
    dataset_dir: str | Path,
    output: str | Path,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    val_split: float = 0.15,
    device: str = "auto",
    base_channels: int = 32,
    seed: int = 42,
) -> dict:
    """Train the U-Net and save a checkpoint."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = GlyphMaskDataset(dataset_dir)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_size = int(round(len(indices) * val_split)) if len(indices) > 3 else 0
    val_indices = indices[:val_size]
    train_indices = indices[val_size:] or indices

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = (
        DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, collate_fn=_collate)
        if val_indices
        else None
    )

    device_obj = resolve_device(device)
    model = EcoMaskUNet(input_channels=4, base_channels=base_channels).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", unit="batch"):
            x = x.to(device_obj)
            y = y.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = mask_loss(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x = x.to(device_obj)
                    y = y.to(device_obj)
                    val_losses.append(float(mask_loss(model(x), y).cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else train_loss

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_loss <= best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "model_config": {"input_channels": 4, "base_channels": base_channels},
        "history": history,
        "dataset_dir": str(dataset_dir),
        "best_val_loss": best_val,
    }
    torch.save(checkpoint, output_path)

    metrics = {
        "checkpoint": str(output_path),
        "samples": len(dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "best_val_loss": best_val,
        "device": str(device_obj),
    }
    (output_path.parent / "training_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics
