from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .metrics import evaluate, mean_metrics
from .priors import STYLES, input_channels, make_target
from .render import has_visible_glyph, render_glyph, save_gray, supported_chars, unique_chars


@dataclass(frozen=True)
class BuildConfig:
    font: Path
    chars: str
    out_dir: Path
    styles: tuple[str, ...] = tuple(STYLES)
    target_savings: tuple[float, ...] = (0.45, 0.60)
    image_size: int = 96
    font_size: int | None = None


def build_dataset(config: BuildConfig) -> dict:
    out = config.out_dir
    sample_dir = out / "samples"
    preview_dir = out / "preview"
    sample_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    chars, missing = supported_chars(config.font, unique_chars(config.chars))

    rows: list[dict] = []
    idx = 0
    for ch in tqdm(chars, desc="build", unit="glyph"):
        rendered = render_glyph(config.font, ch, image_size=config.image_size, font_size=config.font_size)
        if not has_visible_glyph(rendered.image):
            continue
        for style in config.styles:
            if style not in STYLES:
                raise ValueError(f"Unknown style: {style}. Valid: {','.join(STYLES)}")
            for target_saving in config.target_savings:
                target, score = make_target(rendered.image, style, target_saving)
                x = input_channels(rendered.image, style, target_saving)
                y = target[None, :, :].astype(np.float32)
                sample_name = f"{idx:06d}_u{ord(ch):04x}_{style}_s{int(target_saving * 1000):03d}.npz"
                np.savez_compressed(
                    sample_dir / sample_name,
                    x=x,
                    y=y,
                    glyph=rendered.image.astype(np.float32),
                    target=target.astype(np.float32),
                    score=score.astype(np.float32),
                    style_id=np.array(STYLES.index(style), dtype=np.int64),
                    target_saving=np.array(float(target_saving), dtype=np.float32),
                )
                metrics = evaluate(rendered.image, target, target_saving)
                rows.append(
                    {
                        "sample": str(Path("samples") / sample_name),
                        "font": str(config.font),
                        "char": ch,
                        "char_id": f"u{ord(ch):04x}",
                        "style": style,
                        "style_id": STYLES.index(style),
                        "target_saving": float(target_saving),
                        "metrics": metrics,
                    }
                )
                if idx < 16:
                    save_gray(preview_dir / f"{idx:06d}_{style}_original.png", rendered.image)
                    save_gray(preview_dir / f"{idx:06d}_{style}_target.png", target)
                idx += 1

    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    summary = {
        "config": asdict(config),
        "styles": STYLES,
        "sample_count": len(rows),
        "supported_count": len(chars),
        "missing": [{"char": ch, "codepoint": f"U+{ord(ch):04X}"} for ch in missing],
        "average_metrics": mean_metrics([row["metrics"] for row in rows]),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


class EcoStyleDataset(Dataset):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"No rows in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        data = np.load(self.root / row["sample"])
        return (
            data["x"].astype(np.float32),
            data["y"].astype(np.float32),
            int(data["style_id"]),
            float(data["target_saving"]),
            data["glyph"].astype(np.float32),
            row,
        )
