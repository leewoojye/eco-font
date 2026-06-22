from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_yaml
from .data import RymanDataset, split_records
from .losses import RymanLoss
from .model import build_model


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(items: list[dict]) -> dict:
    if not items:
        return {}
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]}


def _epoch(model, loader, criterion, device, optimizer=None, amp=False) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train_mode)
    rows: list[dict] = []
    iterator = tqdm(loader, leave=False, desc="train" if train_mode else "val")
    for batch in iterator:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        glyph = batch["glyph"].to(device=device, dtype=torch.float32)
        saving = batch["target_saving"].to(device=device, dtype=torch.float32)
        skeleton = x[:, 2:3]
        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss, parts = criterion(logits, y, glyph, saving, skeleton)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        rows.append(parts)
        iterator.set_postfix(loss=f"{parts['loss']:.4f}", save=f"{parts['predicted_saving']:.3f}")
    return _mean(rows)


def train_from_config(config_path: str | Path) -> Path:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 23))
    _seed(seed)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    manifest = Path(data_cfg["manifest"])
    out_dir = ensure_dir(train_cfg.get("output_dir", "runs/hangul"))
    device = _device(str(train_cfg.get("device", "auto")))
    train_records, val_records = split_records(manifest, float(data_cfg.get("val_ratio", 0.12)), seed)
    train_ds = RymanDataset(manifest, train_records)
    val_ds = RymanDataset(manifest, val_records) if val_records else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=int(train_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        if val_ds
        else None
    )
    model = build_model(cfg.get("model", {})).to(device)
    criterion = RymanLoss(**{k: float(v) for k, v in cfg.get("loss", {}).items()})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 8e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    epochs = int(train_cfg.get("epochs", 18))
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    best = float("inf")
    history: list[dict] = []
    best_path = out_dir / "checkpoint_best.pt"
    last_path = out_dir / "checkpoint_last.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, criterion, device, optimizer=optimizer, amp=amp)
        val_metrics = _epoch(model, val_loader, criterion, device, optimizer=None, amp=amp) if val_loader else {}
        score = val_metrics.get("loss", train_metrics["loss"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": cfg.get("model", {}),
            "config": cfg,
            "history": history,
        }
        torch.save(ckpt, last_path)
        if score < best:
            best = score
            torch.save(ckpt, best_path)
        print(f"epoch {epoch:03d}/{epochs} train_loss={train_metrics['loss']:.4f} val_loss={score:.4f} best={best:.4f}")
    return best_path
