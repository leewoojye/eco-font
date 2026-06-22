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
from .metrics import evaluate
from .pseudo import input_channels, make_ryman_target
from .render import has_visible_glyph, load_gray, render_glyph, save_gray


@dataclass(frozen=True)
class BuildSummary:
    records: int
    skipped: int
    manifest: str


def _safe_char_id(ch: str) -> str:
    return f"u{ord(ch):04x}"


def _font_paths(fonts: Iterable[str | Path], fonts_dir: str | Path | None = None) -> list[Path]:
    paths = [Path(p) for p in fonts]
    if fonts_dir:
        root = Path(fonts_dir)
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            paths.extend(sorted(root.rglob(pattern)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
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
    max_records: int | None = None,
    target_style: str = "contour",
) -> BuildSummary:
    out = ensure_dir(out_dir)
    original_dir = ensure_dir(out / "original")
    target_dir = ensure_dir(out / "target")
    score_dir = ensure_dir(out / "score")
    manifest_path = out / "manifest.jsonl"
    summary_path = out / "summary.json"
    paths = _font_paths(fonts, fonts_dir)
    records = 0
    skipped = 0
    accum: list[dict] = []
    with manifest_path.open("w", encoding="utf-8") as f:
        for font_idx, font_path in enumerate(paths):
            for ch in tqdm(chars, desc=f"build {font_path.name}", leave=False):
                rendered = render_glyph(font_path, ch, image_size=image_size, font_size=font_size)
                if not has_visible_glyph(rendered.image):
                    skipped += len(target_savings)
                    continue
                for saving in target_savings:
                    target, score = make_ryman_target(rendered.image, saving, style=target_style, char=ch)
                    stem = f"font{font_idx:03d}_{_safe_char_id(ch)}_s{int(round(saving * 1000)):03d}"
                    original_rel = Path("original") / f"{stem}.png"
                    target_rel = Path("target") / f"{stem}.png"
                    score_rel = Path("score") / f"{stem}.png"
                    save_gray(original_dir / original_rel.name, rendered.image)
                    save_gray(target_dir / target_rel.name, target)
                    save_gray(score_dir / score_rel.name, score)
                    metrics = evaluate(rendered.image, target, saving)
                    accum.append(metrics)
                    record = {
                        "font_path": str(font_path),
                        "font_index": font_idx,
                        "char": ch,
                        "char_id": _safe_char_id(ch),
                        "target_saving": float(saving),
                        "image_size": image_size,
                        "font_size": font_size,
                        "target_style": target_style,
                        "original": str(original_rel),
                        "target": str(target_rel),
                        "score": str(score_rel),
                        "metrics": metrics,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records += 1
                    if max_records is not None and records >= max_records:
                        break
                if max_records is not None and records >= max_records:
                    break
            if max_records is not None and records >= max_records:
                break
    summary = {
        "records": records,
        "skipped": skipped,
        "fonts": [str(p) for p in paths],
        "chars": len(chars),
        "target_savings": target_savings,
        "target_style": target_style,
        "manifest": str(manifest_path),
    }
    if accum:
        for key in ["ink_saving", "saving_gap", "skeleton_recall", "aesthetic_line_score"]:
            summary[f"mean_{key}"] = float(np.mean([float(m[key]) for m in accum]))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return BuildSummary(records=records, skipped=skipped, manifest=str(manifest_path))


def load_records(manifest_path: str | Path) -> list[dict]:
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_records(manifest_path: str | Path, val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    records = load_records(manifest_path)
    if not records:
        raise ValueError(f"No records in {manifest_path}")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(records) * val_ratio))) if len(records) > 1 else 0
    val_indices = set(indices[:val_count].tolist())
    train_records = [record for i, record in enumerate(records) if i not in val_indices]
    val_records = [record for i, record in enumerate(records) if i in val_indices]
    if not train_records:
        return records, []
    return train_records, val_records


class RymanDataset(Dataset):
    def __init__(self, manifest_path: str | Path, records: list[dict] | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.records = records if records is not None else load_records(manifest_path)
        if not self.records:
            raise ValueError(f"No records in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        glyph = load_gray(self.root / record["original"])
        target = load_gray(self.root / record["target"])
        saving = float(record["target_saving"])
        style = str(record.get("target_style", "contour"))
        return {
            "x": torch.from_numpy(input_channels(glyph, saving, style=style, char=record["char"])),
            "y": torch.from_numpy(target[None].astype(np.float32)),
            "glyph": torch.from_numpy(glyph[None].astype(np.float32)),
            "target_saving": torch.tensor(saving, dtype=torch.float32),
            "char": record["char"],
        }
