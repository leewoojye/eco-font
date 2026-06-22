from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .font_render import load_gray, model_input_channels


class EcoGlyphDiffusionDataset(Dataset):
    def __init__(self, manifest_path: str | Path, records: list[dict] | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        if records is None:
            with self.manifest_path.open("r", encoding="utf-8") as f:
                self.records = [json.loads(line) for line in f if line.strip()]
        else:
            self.records = records
        if not self.records:
            raise ValueError(f"No records found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        glyph = load_gray(self.root / record["original"])
        eco = load_gray(self.root / record["eco"])
        target_saving = float(record["target_saving"])
        condition = model_input_channels(glyph, target_saving)
        # Diffusion operates on [-1, 1].
        x0 = eco[None, :, :].astype(np.float32) * 2.0 - 1.0
        return {
            "condition": torch.from_numpy(condition),
            "x0": torch.from_numpy(x0),
            "glyph": torch.from_numpy(glyph[None, :, :].astype(np.float32)),
            "eco": torch.from_numpy(eco[None, :, :].astype(np.float32)),
            "target_saving": torch.tensor(target_saving, dtype=torch.float32),
            "char": record["char"],
        }
