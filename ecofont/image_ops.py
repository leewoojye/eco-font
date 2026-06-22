"""Image transforms used by the EcoFont pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def to_float01(image: np.ndarray) -> np.ndarray:
    """Convert an array to float32 in [0, 1]."""
    arr = image.astype(np.float32, copy=False)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def binarize(foreground: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Return a uint8 foreground mask."""
    return (to_float01(foreground) > threshold).astype(np.uint8)


def distance_transform(foreground: np.ndarray) -> np.ndarray:
    """Distance to glyph outline, zero outside the glyph."""
    mask = binarize(foreground)
    if mask.max(initial=0) == 0:
        return np.zeros_like(foreground, dtype=np.float32)
    return cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)


def normalized_distance_transform(foreground: np.ndarray) -> np.ndarray:
    """Distance transform normalized per glyph."""
    dist = distance_transform(foreground)
    max_dist = float(dist.max(initial=0.0))
    if max_dist <= 0:
        return dist
    return dist / max_dist


def skeletonize(foreground: np.ndarray) -> np.ndarray:
    """Morphological skeleton for binary glyph masks."""
    img = binarize(foreground).copy()
    if img.max(initial=0) == 0:
        return img.astype(np.float32)

    skeleton = np.zeros_like(img, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iter = int(math.ceil(max(img.shape) * 1.5))

    for _ in range(max_iter):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(img, opened))
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skeleton.astype(np.float32)


def features_for_glyph(foreground: np.ndarray, target_saving: float) -> np.ndarray:
    """Create model input channels for one glyph."""
    fg = to_float01(foreground)
    dist = normalized_distance_transform(fg)
    skel = skeletonize(fg)
    target = np.full_like(fg, float(target_saving), dtype=np.float32)
    return np.stack([fg, dist, skel, target], axis=0).astype(np.float32)


def foreground_to_pil(foreground: np.ndarray) -> Image.Image:
    """Convert black foreground-on-white background to a PIL grayscale image."""
    fg = to_float01(foreground)
    background = ((1.0 - fg) * 255.0).round().astype(np.uint8)
    return Image.fromarray(background, mode="L")


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Convert a removal mask to a PIL grayscale visualization."""
    arr = (to_float01(mask) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def save_foreground_png(path: str | Path, foreground: np.ndarray) -> None:
    """Save a glyph foreground array as black-on-white PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    foreground_to_pil(foreground).save(path)


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    """Save a removal mask visualization."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mask_to_pil(mask).save(path)


def make_contact_sheet(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], path: str | Path) -> None:
    """Save a contact sheet with original, eco, and mask columns."""
    if not rows:
        return

    cell_h, cell_w = rows[0][1].shape
    label_h = 20
    pad = 8
    cols = 3
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = len(rows) * (cell_h + label_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    headings = ["original", "eco", "remove"]
    for idx, (_, original, eco, mask) in enumerate(rows):
        top = pad + idx * (cell_h + label_h + pad)
        safe_label = rows[idx][0].encode("ascii", errors="ignore").decode("ascii") or f"glyph-{idx}"
        draw.text((pad, top), safe_label, fill=(0, 0, 0))
        for col, (heading, arr) in enumerate(zip(headings, [original, eco, mask], strict=True)):
            left = pad + col * (cell_w + pad)
            draw.text((left, top + 10), heading, fill=(80, 80, 80))
            if heading == "remove":
                img = mask_to_pil(arr).convert("RGB")
            else:
                img = foreground_to_pil(arr).convert("RGB")
            sheet.paste(img, (left, top + label_h))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def render_text_line(
    font_path: str | Path,
    text: str,
    font_size: int = 48,
    padding: int = 16,
) -> np.ndarray:
    """Render a text line and return foreground ink in [0, 1]."""
    font = ImageFont.truetype(str(font_path), font_size)
    scratch = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0] + 2 * padding)
    height = max(1, bbox[3] - bbox[1] + 2 * padding)
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=0)
    arr = np.asarray(image).astype(np.float32)
    return 1.0 - (arr / 255.0)
