from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import CFEcoDataset
from .losses import generator_loss
from .model import CFFontEcoNet


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "content": torch.from_numpy(np.stack([item["content"] for item in batch])),
        "glyph": torch.from_numpy(np.stack([item["glyph"] for item in batch])),
        "target": torch.from_numpy(np.stack([item["target"] for item in batch])),
        "hints": torch.from_numpy(np.stack([item["hints"] for item in batch])),
        "style_refs": torch.from_numpy(np.stack([item["style_refs"] for item in batch])),
        "font_index": torch.tensor([item["font_index"] for item in batch], dtype=torch.long),
        "char_index": torch.tensor([item["char_index"] for item in batch], dtype=torch.long),
        "style_id": torch.tensor([item["style_id"] for item in batch], dtype=torch.long),
        "target_saving": torch.tensor([item["target_saving"] for item in batch], dtype=torch.float32),
        "rows": [item["row"] for item in batch],
    }
    if "basis" in batch[0]:
        out["basis"] = torch.from_numpy(np.stack([item["basis"] for item in batch]))
        out["cfm_weights"] = torch.from_numpy(np.stack([item["cfm_weights"] for item in batch]))
    return out


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _split_indices(length: int, seed: int, val_ratio: float = 0.15) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(length * val_ratio))) if length > 12 else 0
    val_idx = indices[:val_count]
    train_idx = indices[val_count:] or indices
    return train_idx, val_idx


def _epoch(
    model: CFFontEcoNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_cfm: bool,
    pcl_weight: float,
    desc: str,
) -> dict[str, float]:
    model.train(optimizer is not None)
    losses: list[float] = []
    parts: dict[str, list[float]] = {}
    for raw in tqdm(loader, desc=desc, unit="batch"):
        batch = _move(raw, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if use_cfm:
            logits = model.forward_cf(
                batch["basis"],
                batch["cfm_weights"],
                batch["style_refs"],
                batch["hints"],
                batch["style_id"],
                batch["target_saving"],
            )
        else:
            logits = model.forward_base(
                batch["content"],
                batch["style_refs"],
                batch["hints"],
                batch["style_id"],
                batch["target_saving"],
            )
        loss, detail = generator_loss(
            logits,
            batch["target"],
            batch["glyph"],
            batch["target_saving"],
            batch["hints"][:, 1:2],
            pcl_weight=pcl_weight,
        )
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for key, value in detail.items():
            parts.setdefault(key, []).append(float(value))
    summary = {key: float(np.mean(values)) for key, values in parts.items()}
    summary["loss"] = float(np.mean(losses)) if losses else 0.0
    return summary


@torch.no_grad()
def collect_font_embeddings(model: CFFontEcoNet, dataset: CFEcoDataset, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    for font_index in range(len(dataset.fonts)):
        refs = dataset.style_refs_for_font(font_index)
        images = torch.from_numpy(refs).to(device)
        embeddings = model.content_embeddings(images).detach().cpu().numpy()
        rows.append(embeddings.reshape(-1))
    return np.stack(rows).astype(np.float32)


def _softmax_np(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(values)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-12, None)


def _pairwise_l1(features: np.ndarray) -> np.ndarray:
    diff = np.abs(features[:, None, :] - features[None, :, :])
    return diff.mean(axis=2).astype(np.float32)


def _kmedoids(distance: np.ndarray, k: int, seed: int, max_iter: int = 50) -> list[int]:
    n = distance.shape[0]
    k = max(1, min(int(k), n))
    rng = random.Random(seed)
    medoids = [rng.randrange(n)]
    while len(medoids) < k:
        nearest = np.min(distance[:, medoids], axis=1)
        nearest[medoids] = -1.0
        medoids.append(int(np.argmax(nearest)))
    medoids = sorted(set(medoids))
    while len(medoids) < k:
        candidate = rng.randrange(n)
        if candidate not in medoids:
            medoids.append(candidate)

    for _ in range(max_iter):
        labels = np.argmin(distance[:, medoids], axis=1)
        new_medoids = []
        for cluster_id in range(k):
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                new_medoids.append(medoids[cluster_id])
                continue
            intra = distance[np.ix_(members, members)].sum(axis=1)
            new_medoids.append(int(members[int(np.argmin(intra))]))
        if new_medoids == medoids:
            break
        medoids = new_medoids
    return [int(item) for item in medoids]


def select_basis_fonts(font_embeddings: np.ndarray, basis_count: int, seed: int) -> list[int]:
    raw_distance = _pairwise_l1(font_embeddings)
    signatures = _softmax_np(-raw_distance, axis=1)
    signature_distance = _pairwise_l1(signatures)
    return _kmedoids(signature_distance, basis_count, seed)


def calculate_cfm_weights(font_embeddings: np.ndarray, basis_ids: list[int], temperature: float) -> np.ndarray:
    basis = font_embeddings[basis_ids]
    distances = np.abs(font_embeddings[:, None, :] - basis[None, :, :]).mean(axis=2)
    return _softmax_np(-distances / max(float(temperature), 1e-6), axis=1).astype(np.float32)


def train_cf_font(
    dataset: str | Path,
    out: str | Path,
    base_epochs: int = 2,
    cf_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    base_channels: int = 24,
    style_dim: int = 96,
    basis_count: int = 4,
    cfm_temperature: float = 0.18,
    pcl_weight: float = 1.0,
    device_name: str = "auto",
    seed: int = 23,
    ref_count: int | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ds = CFEcoDataset(dataset, ref_count=ref_count)
    train_idx, val_idx = _split_indices(len(ds), seed)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, collate_fn=_collate) if val_idx else None

    device = device_from_name(device_name)
    model = CFFontEcoNet(
        base_channels=base_channels,
        style_dim=style_dim,
        num_eco_styles=len(ds.summary["styles"]),
        cfm_temperature=cfm_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, base_epochs + 1):
        train_row = _epoch(model, train_loader, device, optimizer, use_cfm=False, pcl_weight=pcl_weight, desc=f"base {epoch}/{base_epochs}")
        val_row = train_row
        if val_loader is not None:
            with torch.no_grad():
                val_row = _epoch(model, val_loader, device, None, use_cfm=False, pcl_weight=pcl_weight, desc=f"base-val {epoch}/{base_epochs}")
        row = {"stage": "base", "epoch": epoch, "train": train_row, "val": val_row}
        print(json.dumps(row, ensure_ascii=False))
        history.append(row)
        if val_row["loss"] <= best_val:
            best_val = val_row["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    font_embeddings = collect_font_embeddings(model, ds, device)
    basis_ids = select_basis_fonts(font_embeddings, basis_count=basis_count, seed=seed)
    cfm_weights = calculate_cfm_weights(font_embeddings, basis_ids=basis_ids, temperature=cfm_temperature)
    ds.set_cfm(basis_ids, cfm_weights)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, collate_fn=_collate) if val_idx else None

    for param in model.content_encoder.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate * 0.75, weight_decay=1e-4)

    for epoch in range(1, cf_epochs + 1):
        train_row = _epoch(model, train_loader, device, optimizer, use_cfm=True, pcl_weight=pcl_weight, desc=f"cf {epoch}/{cf_epochs}")
        val_row = train_row
        if val_loader is not None:
            with torch.no_grad():
                val_row = _epoch(model, val_loader, device, None, use_cfm=True, pcl_weight=pcl_weight, desc=f"cf-val {epoch}/{cf_epochs}")
        row = {"stage": "cfm", "epoch": epoch, "train": train_row, "val": val_row}
        print(json.dumps(row, ensure_ascii=False))
        history.append(row)
        if val_row["loss"] <= best_val:
            best_val = val_row["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    basis_rows = [ds.fonts[idx] for idx in basis_ids]
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "model_config": model.model_config,
            "history": history,
            "dataset": str(dataset),
            "dataset_summary": ds.summary,
            "basis_font_indices": basis_ids,
            "basis_fonts": basis_rows,
            "cfm_weights": cfm_weights,
            "font_embeddings": font_embeddings,
            "ref_chars": ds.ref_chars,
            "best_val_loss": best_val,
        },
        out,
    )
    basis_summary = {
        "basis_font_indices": basis_ids,
        "basis_fonts": basis_rows,
        "cfm_weights": cfm_weights.tolist(),
        "temperature": cfm_temperature,
    }
    (out.parent / "basis_summary.json").write_text(json.dumps(basis_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "checkpoint": str(out),
        "samples": len(ds),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "best_val_loss": best_val,
        "basis_font_indices": basis_ids,
        "basis_fonts": basis_rows,
        "device": str(device),
    }
    (out.parent / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
