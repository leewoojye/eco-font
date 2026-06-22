from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class GlyphMask:
    char: str
    codepoint: str
    mask: Image.Image
    bbox: tuple[int, int, int, int] | None


def unique_chars(chars: str) -> list[str]:
    seen: dict[str, None] = {}
    for ch in chars:
        if not ch.isspace():
            seen.setdefault(ch, None)
    return list(seen.keys())


def render_glyph_mask(
    font_path: str | Path,
    char: str,
    size: int = 512,
    font_size: int | None = None,
    padding: int = 32,
    oversample: int = 3,
) -> GlyphMask:
    scale = max(1, int(oversample))
    canvas_size = int(size) * scale
    font_size = font_size or int(size * 0.76)
    font = ImageFont.truetype(str(font_path), font_size * scale)
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return GlyphMask(char, f"U+{ord(char):04X}", Image.new("L", (size, size), 0), None)
    pad = padding * scale
    x = (canvas_size - w) // 2 - bbox[0]
    y = (canvas_size - h) // 2 - bbox[1]
    x = int(np.clip(x, pad - bbox[0], canvas_size - pad - w - bbox[0]))
    y = int(np.clip(y, pad - bbox[1], canvas_size - pad - h - bbox[1]))
    draw.text((x, y), char, fill=255, font=font)
    if scale > 1:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.uint8).copy()
    arr[arr < 6] = 0
    ys, xs = np.where(arr > 12)
    out_bbox = None
    if len(xs) > 0:
        out_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return GlyphMask(char, f"U+{ord(char):04X}", Image.fromarray(arr, mode="L"), out_bbox)


def boundary_mask(mask: Image.Image, width: int = 9) -> Image.Image:
    import cv2

    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    binary = (arr > 16).astype(np.uint8)
    if binary.max() == 0:
        return Image.new("L", mask.size, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    dilated = cv2.dilate(binary, kernel)
    eroded = cv2.erode(binary, kernel)
    edge = ((dilated - eroded) > 0).astype(np.uint8) * 255
    return Image.fromarray(edge, mode="L")


def save_mask_triplet(root: Path, glyph: GlyphMask) -> None:
    root.mkdir(parents=True, exist_ok=True)
    glyph.mask.save(root / f"{glyph.codepoint}_mask.png")
    preview = Image.new("RGB", glyph.mask.size, "black")
    preview.paste(Image.new("RGB", glyph.mask.size, "white"), mask=glyph.mask)
    preview.save(root / f"{glyph.codepoint}_white_on_black.png")
