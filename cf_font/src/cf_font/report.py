from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_pil(image: np.ndarray, size: int) -> Image.Image:
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def contact_sheet(rows: list[tuple[str, np.ndarray, np.ndarray]], out: str | Path, cell: int = 104) -> None:
    if not rows:
        return
    label_h = 24
    cols = min(4, len(rows))
    row_h = cell + label_h
    width = cols * cell * 2
    height = ((len(rows) + cols - 1) // cols) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (label, original, eco) in enumerate(rows):
        col = idx % cols
        row = idx // cols
        x = col * cell * 2
        y = row * row_h
        draw.text((x + 4, y + 4), label[:28], fill=(0, 0, 0), font=font)
        sheet.paste(_to_pil(original, cell), (x, y + label_h))
        sheet.paste(_to_pil(eco, cell), (x + cell, y + label_h))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def candidate_sheet(rows: list[dict], out: str | Path, cell: int = 92) -> None:
    if not rows:
        return
    max_candidates = max(len(row["candidates"]) for row in rows)
    label_h = 36
    width = (max_candidates + 1) * cell
    height = len(rows) * (cell + label_h)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_idx, row in enumerate(rows):
        y = row_idx * (cell + label_h)
        draw.text((4, y + 4), str(row["label"])[:24], fill=(0, 0, 0), font=font)
        sheet.paste(_to_pil(row["original"], cell), (0, y + label_h))
        for cand_idx, candidate in enumerate(row["candidates"]):
            x = (cand_idx + 1) * cell
            outline = (20, 120, 20) if candidate.get("selected") else (180, 180, 180)
            draw.rectangle((x, y + label_h, x + cell - 1, y + label_h + cell - 1), outline=outline, width=2)
            draw.text((x + 3, y + 4), str(candidate["label"])[:18], fill=(0, 0, 0), font=font)
            sheet.paste(_to_pil(candidate["image"], cell), (x, y + label_h))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
