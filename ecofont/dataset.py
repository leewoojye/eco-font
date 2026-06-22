"""Dataset generation and PyTorch dataset wrappers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .font_io import filter_supported_chars
from .image_ops import features_for_glyph
from .metrics import average_metrics
from .render import render_glyph, safe_char_name
from .rules import RuleWeights, optimize_rule
from .text_presets import characters_for_language


@dataclass(frozen=True)
class DatasetBuildConfig:
    fonts: list[Path]
    output: Path
    language: str = "ko"
    text: str | None = None
    targets: tuple[float, ...] = (0.15, 0.25, 0.35)
    image_size: int = 128
    max_chars: int | None = None
    candidate_limit: int | None = None
    weights: RuleWeights = field(default_factory=RuleWeights)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def build_dataset(config: DatasetBuildConfig) -> dict:
    """Build pseudo-labeled glyph samples from one or more fonts."""
    output = config.output
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    chars = characters_for_language(config.language, config.text)
    if config.max_chars is not None:
        chars = chars[: config.max_chars]

    metadata: list[dict] = []
    skipped: list[dict] = []
    sample_idx = 0

    work_items: list[tuple[Path, str, float]] = []
    for font_path in config.fonts:
        supported, missing = filter_supported_chars(font_path, chars)
        for ch in missing:
            skipped.append(
                {
                    "font": str(font_path),
                    "char": ch,
                    "codepoint": f"U+{ord(ch):04X}",
                    "reason": "missing_from_font",
                }
            )
        for ch in supported:
            for target in config.targets:
                work_items.append((font_path, ch, target))

    for font_path, ch, target in tqdm(work_items, desc="building dataset", unit="sample"):
        rendered = render_glyph(font_path, ch, image_size=config.image_size)
        if float(rendered.foreground.sum()) < 5.0:
            skipped.append(
                {
                    "font": str(font_path),
                    "char": ch,
                    "codepoint": rendered.codepoint,
                    "reason": "empty_render",
                }
            )
            continue

        result = optimize_rule(
            rendered.foreground,
            target_saving=target,
            weights=config.weights,
            candidate_limit=config.candidate_limit,
        )
        x = features_for_glyph(rendered.foreground, target)
        y = result.remove_mask[None, :, :].astype(np.float32)

        font_stem = Path(font_path).stem.replace(" ", "_")
        sample_name = f"{sample_idx:06d}_{font_stem}_{safe_char_name(ch)}_{int(target * 100):02d}.npz"
        sample_path = sample_dir / sample_name
        np.savez_compressed(
            sample_path,
            x=x,
            y=y,
            original=rendered.foreground.astype(np.float32),
            eco=result.eco.astype(np.float32),
            remove_mask=result.remove_mask.astype(np.float32),
        )

        metadata.append(
            {
                "sample": str(sample_path.relative_to(output)),
                "font": str(font_path),
                "char": ch,
                "codepoint": rendered.codepoint,
                "target_saving": float(target),
                "loss": result.loss,
                "params": result.params_dict(),
                "metrics": result.metrics,
            }
        )
        sample_idx += 1

    _write_jsonl(output / "metadata.jsonl", metadata)
    _write_jsonl(output / "skipped.jsonl", skipped)

    metric_rows = [row["metrics"] for row in metadata]
    summary = {
        "config": asdict(config),
        "sample_count": len(metadata),
        "skipped_count": len(skipped),
        "average_metrics": average_metrics(metric_rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


class GlyphMaskDataset:
    """Lazy loader compatible with torch.utils.data.DataLoader."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        metadata_path = self.dataset_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        self.rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"No samples found in {metadata_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.dataset_dir / row["sample"]
        data = np.load(path)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        return x, y, row
