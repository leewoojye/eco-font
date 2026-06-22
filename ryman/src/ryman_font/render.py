from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RenderedGlyph:
    char: str
    image: np.ndarray
    bbox: tuple[int, int, int, int] | None
    font_path: str


def render_glyph(
    font_path: str | Path,
    char: str,
    image_size: int = 96,
    font_size: int = 76,
    padding: int = 4,
    oversample: int = 4,
) -> RenderedGlyph:
    scale = max(1, int(oversample))
    size = image_size * scale
    font = ImageFont.truetype(str(font_path), font_size * scale)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        arr = np.zeros((image_size, image_size), dtype=np.float32)
        return RenderedGlyph(char, arr, None, str(font_path))
    pad = padding * scale
    x = (size - width) // 2 - bbox[0]
    y = (size - height) // 2 - bbox[1]
    x = int(np.clip(x, pad - bbox[0], size - pad - width - bbox[0]))
    y = int(np.clip(y, pad - bbox[1], size - pad - height - bbox[1]))
    draw.text((x, y), char, font=font, fill=255)
    if scale > 1:
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[arr < 0.01] = 0.0
    ys, xs = np.where(arr > 0.03)
    out_bbox = None
    if len(xs) > 0:
        out_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return RenderedGlyph(char, arr.astype(np.float32), out_bbox, str(font_path))


def has_visible_glyph(image: np.ndarray, min_pixels: int = 16) -> bool:
    return int((image > 0.05).sum()) >= min_pixels


def distance_transform(glyph: np.ndarray) -> np.ndarray:
    binary = (glyph > 0.08).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(glyph, dtype=np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    max_value = float(dist.max())
    if max_value > 0:
        dist /= max_value
    return dist


def distance_pixels(glyph: np.ndarray) -> np.ndarray:
    binary = (glyph > 0.08).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(glyph, dtype=np.float32)
    return cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)


def skeleton_map(glyph: np.ndarray) -> np.ndarray:
    binary = (glyph > 0.12).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(glyph, dtype=np.float32)
    work = binary.copy()
    skel = np.zeros_like(work, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, residue)
        work = eroded
    return skel.astype(np.float32)


def coordinate_maps(size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def save_gray(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def load_gray(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
