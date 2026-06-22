from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .config import ensure_dir, load_yaml, read_chars
from .metrics import ink_saving
from .render import has_visible_glyph, render_glyph, save_gray


def build_dataset_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    base = config_path.parent.parent
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    out_dir = ensure_dir(base / data_cfg["out_dir"])
    manifest = base / data_cfg["manifest"]
    original_dir = ensure_dir(out_dir / "source")
    target_dir = ensure_dir(out_dir / "target")
    chars = read_chars(charset_file=base / data_cfg["charset_file"])
    image_size = int(data_cfg.get("image_size", 96))
    font_size = int(data_cfg.get("font_size", 76))
    rows: list[dict] = []
    skipped: list[dict] = []
    for pair_idx, pair in enumerate(data_cfg["font_pairs"]):
        source_font = Path(pair["source"])
        target_font = Path(pair["target"])
        for ch in tqdm(chars, desc=f"build {pair['name']}", leave=False):
            source = render_glyph(source_font, ch, image_size=image_size, font_size=font_size)
            target = render_glyph(target_font, ch, image_size=image_size, font_size=font_size)
            if not has_visible_glyph(source.image) or not has_visible_glyph(target.image):
                skipped.append({"pair": pair["name"], "char": ch, "reason": "missing or invisible glyph"})
                continue
            char_id = f"u{ord(ch):04x}"
            stem = f"pair{pair_idx:02d}_{char_id}"
            source_rel = Path("source") / f"{stem}.png"
            target_rel = Path("target") / f"{stem}.png"
            save_gray(out_dir / source_rel, source.image)
            save_gray(out_dir / target_rel, target.image)
            rows.append(
                {
                    "pair": pair["name"],
                    "source_font": str(source_font),
                    "target_font": str(target_font),
                    "char": ch,
                    "char_id": char_id,
                    "image_size": image_size,
                    "font_size": font_size,
                    "source": str(source_rel),
                    "target": str(target_rel),
                    "target_saving": ink_saving(source.image, target.image),
                }
            )
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "skipped.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skipped),
        encoding="utf-8",
    )
    summary = {"records": len(rows), "skipped": len(skipped), "manifest": str(manifest)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"records={len(rows)} skipped={len(skipped)}")
    print(f"manifest={manifest}")
    return manifest

