from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_yaml
from .dataset import EcoPairDataset, read_manifest, split_records
from .diffusion import DiffusionSchedule
from .model import build_model


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


def _run_epoch(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
    prediction_type: str = "epsilon",
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train_mode)
    losses: list[float] = []
    iterator = tqdm(loader, leave=False, desc="train" if train_mode else "val")
    for batch in iterator:
        x0 = batch["x0"].to(device=device, dtype=torch.float32)
        condition = batch["condition"].to(device=device, dtype=torch.float32)
        timesteps = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        noisy = schedule.q_sample(x0, timesteps, noise)
        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast("cuda", enabled=amp):
                predicted = model(noisy, condition, timesteps)
                if prediction_type == "x0":
                    loss = F.mse_loss(predicted, x0)
                elif prediction_type == "epsilon":
                    loss = F.mse_loss(predicted, noise)
                else:
                    raise ValueError(f"Unknown prediction_type: {prediction_type}")
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        value = float(loss.detach().cpu())
        losses.append(value)
        iterator.set_postfix(mse=f"{value:.4f}")
    return {"mse": float(np.mean(losses)) if losses else 0.0}


def train_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    base = config_path.parent.parent
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 7))
    _seed_everything(seed)
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]
    manifest = base / data_cfg["manifest"]
    records = read_manifest(manifest)
    train_records, val_records = split_records(records, float(data_cfg.get("val_ratio", 0.15)), seed)
    device = _device(str(training_cfg.get("device", "auto")))
    train_dataset = EcoPairDataset(manifest, train_records)
    val_dataset = EcoPairDataset(manifest, val_records) if val_records else None
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
    schedule = DiffusionSchedule(
        timesteps=int(diffusion_cfg.get("timesteps", 64)),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
        device=device,
    )
    prediction_type = str(diffusion_cfg.get("prediction_type", "epsilon"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    output_dir = ensure_dir(base / training_cfg.get("output_dir", "runs/smoke"))
    best_path = output_dir / "diffusion_best.pt"
    last_path = output_dir / "diffusion_last.pt"
    history_path = output_dir / "history.json"
    amp = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    best_score = float("inf")
    history: list[dict] = []
    epochs = int(training_cfg.get("epochs", 12))
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, schedule, train_loader, device, optimizer=optimizer, amp=amp, prediction_type=prediction_type)
        val_metrics = _run_epoch(model, schedule, val_loader, device, optimizer=None, amp=amp, prediction_type=prediction_type) if val_loader else {}
        score = val_metrics.get("mse", train_metrics["mse"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": model_cfg,
            "diffusion_config": diffusion_cfg,
            "config": cfg,
            "history": history,
        }
        torch.save(checkpoint, last_path)
        if score < best_score:
            best_score = score
            torch.save(checkpoint, best_path)
        print(f"epoch {epoch:03d}/{epochs} train_mse={train_metrics['mse']:.4f} val_mse={score:.4f} best={best_score:.4f}")
    print(f"best_checkpoint={best_path}")
    return best_path
