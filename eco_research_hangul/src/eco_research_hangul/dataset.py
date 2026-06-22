from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .render import coordinate_maps, load_gray


def read_manifest(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = int(round(len(items) * float(val_ratio)))
    if len(items) > 1:
        n_val = max(1, min(n_val, len(items) - 1))
    return items[n_val:], items[:n_val]


def condition_from_source(source: np.ndarray, target_saving: float) -> np.ndarray:
    xx, yy = coordinate_maps(source.shape[0])
    saving = np.full_like(source, float(target_saving), dtype=np.float32)
    return np.stack([source.astype(np.float32), saving, xx, yy], axis=0).astype(np.float32)


class EcoPairDataset(Dataset):
    def __init__(self, manifest: str | Path, records: list[dict] | None = None) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.records = records if records is not None else read_manifest(self.manifest)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        source = load_gray(self.root / row["source"]).astype(np.float32)
        target = load_gray(self.root / row["target"]).astype(np.float32)
        condition = condition_from_source(source, float(row["target_saving"]))
        x0 = target[None] * 2.0 - 1.0
        return {
            "x0": torch.from_numpy(x0.astype(np.float32)),
            "condition": torch.from_numpy(condition),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "source": torch.from_numpy(source[None].astype(np.float32)),
            "target_saving": torch.tensor(float(row["target_saving"]), dtype=torch.float32),
            "char": row["char"],
        }

