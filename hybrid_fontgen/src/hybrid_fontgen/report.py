from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _to_img(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def contact_sheet(rows: list[tuple[str, np.ndarray, np.ndarray]], output: str | Path) -> Path:
    if not rows:
        raise ValueError("No rows for contact sheet")
    cell = rows[0][1].shape[0]
    pad = 8
    label_h = 24
    sheet = Image.new("RGB", (pad * 3 + cell * 2, pad + len(rows) * (cell + label_h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, original, eco) in enumerate(rows):
        top = pad + idx * (cell + label_h + pad)
        safe = label.encode("ascii", errors="ignore").decode("ascii") or f"glyph-{idx}"
        draw.text((pad, top), safe, fill=(0, 0, 0))
        sheet.paste(_to_img(original), (pad, top + label_h))
        sheet.paste(_to_img(eco), (pad * 2 + cell, top + label_h))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def candidate_sheet(rows: list[dict], output: str | Path) -> Path:
    if not rows:
        raise ValueError("No rows for candidate sheet")
    cell = rows[0]["original"].shape[0]
    pad = 8
    label_h = 42
    max_candidates = max(len(row["candidates"]) for row in rows)
    cols = 1 + max_candidates
    sheet = Image.new("RGB", (pad + cols * (cell + pad), pad + len(rows) * (cell + label_h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(rows):
        top = pad + row_idx * (cell + label_h + pad)
        cells = [("original", row["original"], False)] + [
            (candidate["label"], candidate["image"], bool(candidate.get("selected"))) for candidate in row["candidates"]
        ]
        for col_idx, (label, image, selected) in enumerate(cells):
            left = pad + col_idx * (cell + pad)
            safe = label.encode("ascii", errors="ignore").decode("ascii") or f"cell-{row_idx}-{col_idx}"
            draw.text((left, top), safe[:24], fill=(0, 0, 0))
            img_top = top + label_h
            sheet.paste(_to_img(image), (left, img_top))
            if selected:
                draw.rectangle((left, img_top, left + cell - 1, img_top + cell - 1), outline=(0, 160, 80), width=3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output
