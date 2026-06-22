from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RenderedGlyph:
    char: str
    image: np.ndarray
    bbox: tuple[int, int, int, int] | None


def unique_chars(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for ch in text:
        if not ch.isspace():
            seen.setdefault(ch, None)
    return list(seen.keys())


def supported_chars(font_path: str | Path, chars: list[str]) -> tuple[list[str], list[str]]:
    font = TTFont(str(font_path), lazy=True)
    cmap: set[int] = set()
    for table in font["cmap"].tables:
        cmap.update(table.cmap.keys())
    font.close()
    ok = [ch for ch in chars if ord(ch) in cmap]
    missing = [ch for ch in chars if ord(ch) not in cmap]
    return ok, missing


def render_glyph(
    font_path: str | Path,
    char: str,
    image_size: int = 96,
    font_size: int | None = None,
    oversample: int = 4,
    padding: int = 4,
) -> RenderedGlyph:
    scale = max(1, int(oversample))
    size = image_size * scale
    font_size = font_size or int(image_size * 0.80)
    font = ImageFont.truetype(str(font_path), font_size * scale)
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), char, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        return RenderedGlyph(char=char, image=np.zeros((image_size, image_size), dtype=np.float32), bbox=None)
    pad = padding * scale
    x = (size - width) // 2 - bbox[0]
    y = (size - height) // 2 - bbox[1]
    x = int(np.clip(x, pad - bbox[0], size - pad - width - bbox[0]))
    y = int(np.clip(y, pad - bbox[1], size - pad - height - bbox[1]))
    draw.text((x, y), char, font=font, fill=255)
    if scale > 1:
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr[arr < 0.01] = 0.0
    ys, xs = np.where(arr > 0.03)
    out_bbox = None
    if len(xs) > 0:
        out_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return RenderedGlyph(char=char, image=np.clip(arr, 0.0, 1.0).astype(np.float32), bbox=out_bbox)


def has_visible_glyph(image: np.ndarray, min_pixels: int = 16) -> bool:
    return int((image > 0.05).sum()) >= min_pixels


def distance_pixels(glyph: np.ndarray) -> np.ndarray:
    binary = (glyph > 0.08).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(glyph, dtype=np.float32)
    return cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)


def distance_transform(glyph: np.ndarray) -> np.ndarray:
    dist = distance_pixels(glyph)
    max_value = float(dist.max(initial=0.0))
    if max_value > 0:
        dist = dist / max_value
    return dist.astype(np.float32)


def skeleton_map(glyph: np.ndarray) -> np.ndarray:
    binary = (glyph > 0.12).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(glyph, dtype=np.float32)
    work = binary.copy()
    skel = np.zeros_like(work, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max(glyph.shape) * 2):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(work, opened))
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return skel.astype(np.float32)


def coordinate_maps(size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx.astype(np.float32), yy.astype(np.float32)


def save_gray(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def load_gray(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
