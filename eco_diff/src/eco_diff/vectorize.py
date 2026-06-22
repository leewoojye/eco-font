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


def _glyph_bbox(font: TTFont, glyph_name: str) -> tuple[int, int, int, int] | None:
    glyph = font["glyf"][glyph_name]
    glyph.recalcBounds(font["glyf"])
    if not hasattr(glyph, "xMin"):
        return None
    if glyph.xMax <= glyph.xMin or glyph.yMax <= glyph.yMin:
        return None
    return int(glyph.xMin), int(glyph.yMin), int(glyph.xMax), int(glyph.yMax)


def _bitmap_to_contours(bitmap: np.ndarray) -> list[np.ndarray]:
    binary = (bitmap > 0.22).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    usable: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < 2.0:
            continue
        epsilon = max(0.75, 0.004 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:
            usable.append(approx[:, 0, :].astype(np.float32))
    return usable


def _contour_to_font_units(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = image_shape
    x_min, y_min, x_max, y_max = bbox
    xs = contour[:, 0]
    ys = contour[:, 1]
    x_units = x_min + (xs / max(1.0, float(w - 1))) * (x_max - x_min)
    y_units = y_max - (ys / max(1.0, float(h - 1))) * (y_max - y_min)
    return [(int(round(x)), int(round(y))) for x, y in zip(x_units, y_units)]


def _glyph_from_bitmap(font: TTFont, glyph_name: str, bitmap: np.ndarray):
    bbox = _glyph_bbox(font, glyph_name)
    if bbox is None:
        return None
    contours = _bitmap_to_contours(bitmap)
    if not contours:
        return None
    pen = TTGlyphPen(font.getGlyphSet())
    for contour in contours:
        points = _contour_to_font_units(contour, bitmap.shape, bbox)
        if len(points) < 3:
            continue
        pen.moveTo(points[0])
        for pt in points[1:]:
            pen.lineTo(pt)
        pen.closePath()
    glyph = pen.glyph()
    glyph.recalcBounds(font["glyf"])
    return glyph


def export_ttf_from_bitmaps(
    source_font: str | Path,
    char_to_bitmap: dict[str, np.ndarray],
    output_font: str | Path,
) -> Path:
    """Replace selected glyphs with contour-traced eco bitmaps.

    This is a practical MVP exporter, not a typography-grade outline optimizer.
    It preserves cmap/hmtx/name/etc. from the source font and only swaps glyph
    outlines for characters that can be mapped.
    """
    output = Path(output_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    font = TTFont(str(source_font))
    glyf = font["glyf"]
    replaced = 0
    for ch, bitmap in char_to_bitmap.items():
        glyph_name = _char_to_glyph_name(font, ch)
        if glyph_name is None or glyph_name not in glyf:
            continue
        new_glyph = _glyph_from_bitmap(font, glyph_name, bitmap)
        if new_glyph is None:
            continue
        glyf[glyph_name] = new_glyph
        replaced += 1
    if replaced == 0:
        raise ValueError("No glyphs were replaced; check charset and source font coverage")
    font.save(str(output))
    return output
