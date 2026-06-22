from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont


def _char_to_glyph_name(font: TTFont, ch: str) -> str | None:
    codepoint = ord(ch)
    for table in font["cmap"].tables:
        if codepoint in table.cmap:
            return table.cmap[codepoint]
    return None


def _bbox(font: TTFont, glyph_name: str) -> tuple[int, int, int, int] | None:
    glyph = font["glyf"][glyph_name]
    glyph.recalcBounds(font["glyf"])
    if not hasattr(glyph, "xMin") or glyph.xMax <= glyph.xMin or glyph.yMax <= glyph.yMin:
        return None
    return int(glyph.xMin), int(glyph.yMin), int(glyph.xMax), int(glyph.yMax)


def _contours(bitmap: np.ndarray) -> list[np.ndarray]:
    binary = (bitmap > 0.18).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < 2.0:
            continue
        eps = max(0.6, 0.004 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, eps, True)
        if len(approx) >= 3:
            out.append(approx[:, 0, :].astype(np.float32))
    return out


def _to_units(contour: np.ndarray, shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = shape
    x_min, y_min, x_max, y_max = bbox
    x_units = x_min + (contour[:, 0] / max(1.0, float(w - 1))) * (x_max - x_min)
    y_units = y_max - (contour[:, 1] / max(1.0, float(h - 1))) * (y_max - y_min)
    return [(int(round(x)), int(round(y))) for x, y in zip(x_units, y_units)]


def export_ttf(source_font: str | Path, char_to_bitmap: dict[str, np.ndarray], output_font: str | Path) -> Path:
    output = Path(output_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    font = TTFont(str(source_font))
    replaced = 0
    for ch, bitmap in char_to_bitmap.items():
        glyph_name = _char_to_glyph_name(font, ch)
        if glyph_name is None or glyph_name not in font["glyf"]:
            continue
        bbox = _bbox(font, glyph_name)
        if bbox is None:
            continue
        contours = _contours(bitmap)
        if not contours:
            continue
        pen = TTGlyphPen(font.getGlyphSet())
        for contour in contours:
            points = _to_units(contour, bitmap.shape, bbox)
            if len(points) < 3:
                continue
            pen.moveTo(points[0])
            for pt in points[1:]:
                pen.lineTo(pt)
            pen.closePath()
        glyph = pen.glyph()
        glyph.recalcBounds(font["glyf"])
        font["glyf"][glyph_name] = glyph
        replaced += 1
    if replaced == 0:
        raise ValueError("No glyphs replaced")
    font.save(str(output))
    return output
