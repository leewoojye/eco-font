from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import ensure_dir
from .font_render import has_visible_glyph, load_gray, model_input_channels, render_glyph, save_gray
from .metrics import apply_cut_mask
from .rules import select_best_candidate


@dataclass(frozen=True)
class DatasetBuildSummary:
    records: int
    skipped: int
    out_dir: str
    manifest: str


def _safe_char_id(ch: str) -> str:
    return f"u{ord(ch):04x}"


def _iter_fonts(fonts: Iterable[str | Path], fonts_dir: str | Path | None = None) -> list[Path]:
    found = [Path(p) for p in fonts]
    if fonts_dir:
        root = Path(fonts_dir)
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            found.extend(sorted(root.rglob(pattern)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        resolved = str(path.expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("No fonts provided")
    return unique


def build_dataset(
    fonts: Iterable[str | Path],
    chars: list[str],
    out_dir: str | Path,
    target_savings: list[float],
    fonts_dir: str | Path | None = None,
    image_size: int = 96,
    font_size: int = 76,
    padding: int = 4,
    max_records: int | None = None,
) -> DatasetBuildSummary:
    out = ensure_dir(out_dir)
    image_dir = ensure_dir(out / "images")
    mask_dir = ensure_dir(out / "masks")
    eco_dir = ensure_dir(out / "eco")
    manifest_path = out / "manifest.jsonl"
    summary_path = out / "summary.json"
    font_paths = _iter_fonts(fonts, fonts_dir)

    records = 0
    skipped = 0
    metrics_accum: list[dict[str, float | int]] = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for font_idx, font_path in enumerate(font_paths):
            font_tag = f"font{font_idx:04d}"
            for ch in tqdm(chars, desc=f"render {font_path.name}", leave=False):
                rendered = render_glyph(font_path, ch, image_size=image_size, font_size=font_size, padding=padding)
                if not has_visible_glyph(rendered.image):
                    skipped += len(target_savings)
                    continue
                char_id = _safe_char_id(ch)
                for saving in target_savings:
                    candidate = select_best_candidate(rendered.image, saving)
                    eco = apply_cut_mask(rendered.image, candidate.cut_mask)
                    stem = f"{font_tag}_{char_id}_s{int(round(saving * 1000)):03d}"
                    image_rel = Path("images") / f"{stem}.png"
                    mask_rel = Path("masks") / f"{stem}.png"
                    eco_rel = Path("eco") / f"{stem}.png"
                    save_gray(image_dir / image_rel.name, rendered.image)
                    save_gray(mask_dir / mask_rel.name, candidate.cut_mask)
                    save_gray(eco_dir / eco_rel.name, eco)
                    metrics = candidate.metrics.to_dict() if candidate.metrics else {}
                    metrics_accum.append(metrics)
                    record = {
                        "font_path": str(font_path),
                        "font_index": font_idx,
                        "char": ch,
                        "char_id": char_id,
                        "target_saving": float(saving),
                        "image_size": image_size,
                        "font_size": font_size,
                        "original": str(image_rel),
                        "mask": str(mask_rel),
                        "eco": str(eco_rel),
                        "rule": candidate.name,
                        "rule_params": candidate.params,
                        "metrics": metrics,
                    }
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records += 1
                    if max_records is not None and records >= max_records:
                        break
                if max_records is not None and records >= max_records:
                    break
            if max_records is not None and records >= max_records:
                break

    aggregate: dict[str, float] = {}
    if metrics_accum:
        numeric_keys = [k for k, v in metrics_accum[0].items() if isinstance(v, (int, float))]
        for key in numeric_keys:
            values = [float(m[key]) for m in metrics_accum if key in m]
            if values:
                aggregate[f"mean_{key}"] = float(np.mean(values))
    summary = {
        "records": records,
        "skipped": skipped,
        "fonts": [str(p) for p in font_paths],
        "chars": len(chars),
        "target_savings": target_savings,
        "aggregate": aggregate,
        "manifest": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return DatasetBuildSummary(records, skipped, str(out), str(manifest_path))


class EcoMaskDataset(Dataset):
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        record = self.records[index]
        glyph = load_gray(self.root / record["original"])
        mask = load_gray(self.root / record["mask"])
        target_saving = float(record["target_saving"])
        x = model_input_channels(glyph, target_saving)
        y = mask[None, :, :].astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "glyph": torch.from_numpy(glyph[None, :, :].astype(np.float32)),
            "target_saving": torch.tensor(target_saving, dtype=torch.float32),
            "char": record["char"],
        }


def split_records(manifest_path: str | Path, val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(records) * val_ratio))) if len(records) > 1 else 0
    val_indices = set(indices[:val_count].tolist())
    train_records = [record for i, record in enumerate(records) if i not in val_indices]
    val_records = [record for i, record in enumerate(records) if i in val_indices]
    if not train_records:
        train_records, val_records = records, []
    return train_records, val_records
