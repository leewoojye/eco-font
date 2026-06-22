from __future__ import annotations

import glob
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .metrics import evaluate, mean_metrics
from .priors import ECO_STYLES, input_hint_channels, make_target
from .render import (
    font_display_name,
    has_visible_glyph,
    read_chars_file,
    render_glyph,
    save_gray,
    supported_chars,
    unique_chars,
)


@dataclass(frozen=True)
class BuildConfig:
    fonts: tuple[Path, ...]
    chars: str
    out_dir: Path
    styles: tuple[str, ...] = tuple(ECO_STYLES)
    target_savings: tuple[float, ...] = (0.45, 0.60)
    image_size: int = 96
    font_size: int | None = None
    limit_chars: int | None = None
    ref_count: int = 8


def parse_fonts(fonts: str | None, fonts_glob: str | None) -> tuple[Path, ...]:
    paths: list[Path] = []
    if fonts:
        paths.extend(Path(part.strip()) for part in fonts.split(",") if part.strip())
    if fonts_glob:
        paths.extend(Path(path) for path in glob.glob(fonts_glob))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(paths):
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise ValueError("No fonts were provided")
    return tuple(unique)


def chars_from_args(chars: str | None, chars_file: Path | None) -> str:
    if chars_file:
        return read_chars_file(chars_file)
    if chars:
        return chars
    raise ValueError("Either --chars or --chars-file is required")


def _font_id(index: int, path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or f"font_{index}"
    return f"f{index:02d}_{stem}"


def build_dataset(config: BuildConfig) -> dict[str, Any]:
    out = config.out_dir
    sample_dir = out / "samples"
    glyph_dir = out / "glyphs"
    preview_dir = out / "preview"
    sample_dir.mkdir(parents=True, exist_ok=True)
    glyph_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    requested_chars = unique_chars(config.chars)
    if config.limit_chars is not None:
        requested_chars = requested_chars[: config.limit_chars]
    if not requested_chars:
        raise ValueError("No characters to render")
    for style in config.styles:
        if style not in ECO_STYLES:
            raise ValueError(f"Unknown eco style {style}. Valid: {','.join(ECO_STYLES)}")

    font_rows = []
    supported_sets: list[set[str]] = []
    missing_by_font: dict[str, list[dict[str, str]]] = {}
    for idx, font_path in enumerate(config.fonts):
        ok, missing = supported_chars(font_path, requested_chars)
        fid = _font_id(idx, font_path)
        font_rows.append(
            {
                "id": fid,
                "index": idx,
                "path": str(font_path),
                "name": font_display_name(font_path),
            }
        )
        supported_sets.append(set(ok))
        missing_by_font[fid] = [{"char": ch, "codepoint": f"U+{ord(ch):04X}"} for ch in missing]
    shared_chars = [ch for ch in requested_chars if all(ch in supported for supported in supported_sets)]
    if config.limit_chars is not None:
        shared_chars = shared_chars[: config.limit_chars]
    if not shared_chars:
        raise ValueError("No shared supported chars across provided fonts")
    ref_chars = shared_chars[: max(1, min(config.ref_count, len(shared_chars)))]

    glyph_cache: dict[tuple[int, str], np.ndarray] = {}
    for font_row in tqdm(font_rows, desc="render-fonts", unit="font"):
        font_index = int(font_row["index"])
        font_path = Path(str(font_row["path"]))
        font_glyph_dir = glyph_dir / str(font_row["id"])
        font_glyph_dir.mkdir(parents=True, exist_ok=True)
        for ch in shared_chars:
            rendered = render_glyph(font_path, ch, image_size=config.image_size, font_size=config.font_size)
            image = rendered.image if has_visible_glyph(rendered.image) else np.zeros((config.image_size, config.image_size), dtype=np.float32)
            glyph_cache[(font_index, ch)] = image.astype(np.float32)
            np.save(font_glyph_dir / f"u{ord(ch):04x}.npy", image.astype(np.float32))

    rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, float]] = []
    idx = 0
    for font_row in tqdm(font_rows, desc="build-samples", unit="font"):
        font_index = int(font_row["index"])
        for ch in shared_chars:
            glyph = glyph_cache[(font_index, ch)]
            if not has_visible_glyph(glyph):
                continue
            for style in config.styles:
                for target_saving in config.target_savings:
                    target, score = make_target(glyph, style, target_saving)
                    hints = input_hint_channels(glyph, style, target_saving)
                    sample_name = f"{idx:06d}_{font_row['id']}_u{ord(ch):04x}_{style}_s{int(target_saving * 1000):03d}.npz"
                    np.savez_compressed(
                        sample_dir / sample_name,
                        glyph=glyph.astype(np.float32),
                        target=target.astype(np.float32),
                        score=score.astype(np.float32),
                        hints=hints.astype(np.float32),
                        font_index=np.array(font_index, dtype=np.int64),
                        char_index=np.array(shared_chars.index(ch), dtype=np.int64),
                        style_id=np.array(ECO_STYLES.index(style), dtype=np.int64),
                        target_saving=np.array(float(target_saving), dtype=np.float32),
                    )
                    metrics = evaluate(glyph, target, target_saving)
                    metrics_rows.append(metrics)
                    rows.append(
                        {
                            "sample": str(Path("samples") / sample_name),
                            "font_id": font_row["id"],
                            "font_index": font_index,
                            "char": ch,
                            "char_id": f"u{ord(ch):04x}",
                            "char_index": shared_chars.index(ch),
                            "style": style,
                            "style_id": ECO_STYLES.index(style),
                            "target_saving": float(target_saving),
                            "metrics": metrics,
                        }
                    )
                    if idx < 20:
                        save_gray(preview_dir / f"{idx:06d}_original.png", glyph)
                        save_gray(preview_dir / f"{idx:06d}_target.png", target)
                    idx += 1

    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "config": asdict(config),
        "fonts": font_rows,
        "styles": list(ECO_STYLES),
        "chars": shared_chars,
        "ref_chars": ref_chars,
        "sample_count": len(rows),
        "font_count": len(font_rows),
        "char_count": len(shared_chars),
        "missing_by_font": missing_by_font,
        "average_metrics": mean_metrics(metrics_rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


class CFEcoDataset(Dataset):
    def __init__(self, root: str | Path, ref_count: int | None = None) -> None:
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        summary_path = self.root / "summary.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"No rows in {manifest}")
        self.fonts = self.summary["fonts"]
        self.chars = self.summary["chars"]
        self.ref_chars = self.summary.get("ref_chars", self.chars[:8])
        if ref_count is not None:
            self.ref_chars = self.ref_chars[:ref_count]
        self.ref_char_indices = [self.chars.index(ch) for ch in self.ref_chars]
        self.image_size = int(self.summary["config"]["image_size"])
        self.glyphs = self._load_glyphs()
        self.basis_font_ids: list[int] | None = None
        self.cfm_weights: np.ndarray | None = None

    def _load_glyphs(self) -> np.ndarray:
        glyphs = np.zeros((len(self.fonts), len(self.chars), self.image_size, self.image_size), dtype=np.float32)
        for font in self.fonts:
            font_index = int(font["index"])
            for char_index, ch in enumerate(self.chars):
                path = self.root / "glyphs" / str(font["id"]) / f"u{ord(ch):04x}.npy"
                glyphs[font_index, char_index] = np.load(path).astype(np.float32)
        return glyphs

    def set_cfm(self, basis_font_ids: list[int], cfm_weights: np.ndarray) -> None:
        self.basis_font_ids = [int(item) for item in basis_font_ids]
        self.cfm_weights = cfm_weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def style_refs_for_font(self, font_index: int) -> np.ndarray:
        return self.glyphs[int(font_index), self.ref_char_indices][:, None, :, :].astype(np.float32)

    def basis_stack_for_char(self, char_index: int) -> np.ndarray:
        if self.basis_font_ids is None:
            raise RuntimeError("CFM basis fonts are not configured")
        return self.glyphs[self.basis_font_ids, int(char_index)][:, None, :, :].astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        data = np.load(self.root / row["sample"])
        font_index = int(data["font_index"])
        char_index = int(data["char_index"])
        item: dict[str, Any] = {
            "content": data["glyph"][None, :, :].astype(np.float32),
            "glyph": data["glyph"][None, :, :].astype(np.float32),
            "target": data["target"][None, :, :].astype(np.float32),
            "hints": data["hints"].astype(np.float32),
            "style_refs": self.style_refs_for_font(font_index),
            "font_index": font_index,
            "char_index": char_index,
            "style_id": int(data["style_id"]),
            "target_saving": float(data["target_saving"]),
            "row": row,
        }
        if self.basis_font_ids is not None:
            item["basis"] = self.basis_stack_for_char(char_index)
            if self.cfm_weights is None:
                raise RuntimeError("CFM weights are not configured")
            item["cfm_weights"] = self.cfm_weights[font_index].astype(np.float32)
        return item
