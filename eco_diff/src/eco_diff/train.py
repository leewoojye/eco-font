from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_yaml
from .dataset import EcoMaskDataset, split_records
from .losses import EcoMaskLoss
from .models import build_model


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: EcoMaskLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train_mode)
    parts: list[dict[str, float]] = []
    iterator = tqdm(loader, leave=False, desc="train" if train_mode else "val")
    for batch in iterator:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        glyph = batch["glyph"].to(device=device, dtype=torch.float32)
        target_saving = batch["target_saving"].to(device=device, dtype=torch.float32)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss, loss_parts = criterion(logits, y, glyph, target_saving)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        parts.append(loss_parts)
        iterator.set_postfix(loss=f"{loss_parts['loss']:.4f}", save=f"{loss_parts['predicted_saving']:.3f}")
    return _mean_dict(parts)


def train_from_config(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 7))
    _seed_everything(seed)
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})

    manifest = Path(data_cfg["manifest"])
    output_dir = ensure_dir(training_cfg.get("output_dir", "runs/demo"))
    device = _device(str(training_cfg.get("device", "auto")))
    train_records, val_records = split_records(manifest, float(data_cfg.get("val_ratio", 0.15)), seed)
    train_dataset = EcoMaskDataset(manifest, train_records)
    val_dataset = EcoMaskDataset(manifest, val_records) if val_records else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(training_cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        if val_dataset is not None
        else None
    )

    model = build_model(model_cfg).to(device)
    criterion = EcoMaskLoss(**{k: float(v) for k, v in loss_cfg.items()})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    epochs = int(training_cfg.get("epochs", 20))
    amp = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    best_val = float("inf")
    history: list[dict] = []
    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer, amp=amp)
        val_metrics = _run_epoch(model, val_loader, criterion, device, optimizer=None, amp=amp) if val_loader else {}
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = val_metrics.get("loss", train_metrics["loss"])
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": model_cfg,
            "loss_config": loss_cfg,
            "config": config,
            "history": history,
        }
        torch.save(checkpoint, last_path)
        if score < best_val:
            best_val = score
            torch.save(checkpoint, best_path)
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={score:.4f} "
            f"best={best_val:.4f}"
        )

    return best_path if best_path.exists() else last_path
