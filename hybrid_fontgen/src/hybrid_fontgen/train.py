from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import EcoStyleDataset
from .model import HybridEcoNet, generator_loss
from .priors import STYLES


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _collate(batch):
    xs, ys, style_ids, target_savings, glyphs, rows = zip(*batch, strict=True)
    return (
        torch.from_numpy(np.stack(xs)),
        torch.from_numpy(np.stack(ys)),
        torch.tensor(style_ids, dtype=torch.long),
        torch.tensor(target_savings, dtype=torch.float32),
        torch.from_numpy(np.stack(glyphs)[:, None, :, :]),
        list(rows),
    )


def train_generator(
    dataset: str | Path,
    out: str | Path,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
    base_channels: int = 24,
    seed: int = 11,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ds = EcoStyleDataset(dataset)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    val_count = max(1, int(round(len(indices) * 0.15))) if len(indices) > 8 else 0
    val_idx = indices[:val_count]
    train_idx = indices[val_count:] or indices

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, collate_fn=_collate) if val_idx else None

    device = device_from_name(device_name)
    model = HybridEcoNet(input_channels=8, base_channels=base_channels, num_styles=len(STYLES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict] = []
    best = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y, style_id, target_saving, glyph, _rows in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            style_id = style_id.to(device)
            target_saving = target_saving.to(device)
            glyph = glyph.to(device)
            skeleton = x[:, 2:3]
            opt.zero_grad(set_to_none=True)
            logits = model(x, style_id)
            loss, _parts = generator_loss(logits, y, glyph, target_saving, skeleton)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_loss = float(np.mean(losses))
        if val_loader is not None:
            model.eval()
            vals: list[float] = []
            with torch.no_grad():
                for x, y, style_id, target_saving, glyph, _rows in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    style_id = style_id.to(device)
                    target_saving = target_saving.to(device)
                    glyph = glyph.to(device)
                    loss, _parts = generator_loss(model(x, style_id), y, glyph, target_saving, x[:, 2:3])
                    vals.append(float(loss.cpu()))
            val_loss = float(np.mean(vals))

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        print(json.dumps(row, ensure_ascii=False))
        history.append(row)
        if val_loss <= best:
            best = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "model_config": {"input_channels": 8, "base_channels": base_channels, "num_styles": len(STYLES)},
            "styles": STYLES,
            "history": history,
            "dataset": str(dataset),
            "best_val_loss": best,
        },
        out,
    )
    summary = {
        "checkpoint": str(out),
        "samples": len(ds),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "best_val_loss": best,
        "device": str(device),
    }
    (out.parent / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
