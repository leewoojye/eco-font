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
    font_size: int
    image_size: int


def load_font(font_path: str | Path, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path), font_size)


def render_char(
    font_path: str | Path,
    char: str,
    image_size: int = 128,
    font_size: int = 104,
    padding: int = 7,
    oversample: int = 4,
) -> RenderedGlyph:
    if len(char) != 1:
        raise ValueError("render_char expects one Unicode character")

    scale = max(1, int(oversample))
    canvas = image_size * scale
    font = load_font(font_path, font_size * scale)
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        empty = np.zeros((image_size, image_size), dtype=np.float32)
        return RenderedGlyph(char, empty, None, str(font_path), font_size, image_size)

    pad = padding * scale
    x = (canvas - width) // 2 - bbox[0]
    y = (canvas - height) // 2 - bbox[1]
    x = int(np.clip(x, pad - bbox[0], canvas - pad - width - bbox[0]))
    y = int(np.clip(y, pad - bbox[1], canvas - pad - height - bbox[1]))
    draw.text((x, y), char, fill=255, font=font)
    if scale > 1:
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[arr < 0.01] = 0.0
    out_bbox = glyph_bbox(arr)
    return RenderedGlyph(char, arr, out_bbox, str(font_path), font_size, image_size)


def glyph_bbox(image: np.ndarray, threshold: float = 0.04) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(image > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def has_visible_glyph(image: np.ndarray, min_pixels: int = 20) -> bool:
    return int((image > 0.05).sum()) >= min_pixels


def crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return np.zeros((1, 1), dtype=np.float32)
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def fit_to_bbox(
    image: np.ndarray,
    target_bbox: tuple[int, int, int, int] | None,
    output_shape: tuple[int, int],
    threshold: float = 0.04,
) -> np.ndarray:
    if target_bbox is None:
        return np.zeros(output_shape, dtype=np.float32)
    source_bbox = glyph_bbox(image, threshold=threshold)
    if source_bbox is None:
        return np.zeros(output_shape, dtype=np.float32)
    crop = crop_to_bbox(image, source_bbox)
    x0, y0, x1, y1 = target_bbox
    target_w = max(1, x1 - x0)
    target_h = max(1, y1 - y0)
    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    out = np.zeros(output_shape, dtype=np.float32)
    out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], resized)
    return np.clip(out, 0.0, 1.0)


def save_gray(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def load_gray(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_rgb_tile(image: np.ndarray, size: int = 96) -> Image.Image:
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    tile = Image.fromarray(arr, mode="L").convert("RGB")
    if tile.size != (size, size):
        tile = tile.resize((size, size), Image.Resampling.LANCZOS)
    return tile


def load_ui_font(size: int = 12, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()
