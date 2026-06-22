"""TTF/OTF glyph rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RenderedGlyph:
    char: str
    codepoint: str
    foreground: np.ndarray


def _fit_font(font_path: str | Path, char: str, image_size: int, margin: int) -> tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]]:
    max_font_size = max(8, image_size - 2 * margin)
    min_font_size = 6
    scratch = Image.new("L", (image_size, image_size), 255)
    draw = ImageDraw.Draw(scratch)

    last_font = ImageFont.truetype(str(font_path), min_font_size)
    last_bbox = draw.textbbox((0, 0), char, font=last_font)

    for font_size in range(max_font_size, min_font_size - 1, -2):
        font = ImageFont.truetype(str(font_path), font_size)
        bbox = draw.textbbox((0, 0), char, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        last_font, last_bbox = font, bbox
        if width <= image_size - 2 * margin and height <= image_size - 2 * margin:
            return font, bbox

    return last_font, last_bbox


def render_glyph(
    font_path: str | Path,
    char: str,
    image_size: int = 128,
    margin: int | None = None,
) -> RenderedGlyph:
    """Render one glyph as foreground ink in [0, 1]."""
    if len(char) != 1:
        raise ValueError("render_glyph expects exactly one Unicode character")

    margin = margin if margin is not None else max(8, image_size // 12)
    scale = 3
    canvas_size = image_size * scale
    scaled_margin = margin * scale

    font, bbox = _fit_font(font_path, char, canvas_size, scaled_margin)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    x = (canvas_size - width) / 2 - bbox[0]
    y = (canvas_size - height) / 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32)
    foreground = 1.0 - (arr / 255.0)
    foreground = np.clip(foreground, 0.0, 1.0)
    return RenderedGlyph(char=char, codepoint=f"U+{ord(char):04X}", foreground=foreground)


def safe_char_name(char: str) -> str:
    """Filesystem-safe character identifier."""
    return f"U{ord(char):04X}"
