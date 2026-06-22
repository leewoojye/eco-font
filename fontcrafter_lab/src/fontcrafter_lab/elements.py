from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _smooth_noise(size: int, seed: int, sigma: float = 10.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-6)


def make_element(kind: str = "blue_stone", size: int = 512, seed: int = 7) -> Image.Image:
    kind = kind.lower().strip()
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    n1 = _normalize(_smooth_noise(size, seed, sigma=size / 42))
    n2 = _normalize(_smooth_noise(size, seed + 31, sigma=size / 16))
    if kind in {"blue_stone", "stone", "marble"}:
        veins = np.sin((xx * 0.030 + yy * 0.018 + n2 * 7.0) * np.pi)
        veins = np.clip((veins + 1.0) * 0.5, 0.0, 1.0)
        base = np.stack(
            [
                38 + 42 * n2 + 80 * (veins > 0.86),
                76 + 54 * n1 + 96 * (veins > 0.88),
                112 + 92 * n2 + 80 * (veins > 0.90),
            ],
            axis=-1,
        )
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    if kind in {"ember", "flame", "fire"}:
        vertical = 1.0 - yy / max(size - 1, 1)
        waves = np.sin(xx * 0.055 + n1 * 8.0) * 0.35 + np.sin(xx * 0.022 + yy * 0.030) * 0.25
        heat = _normalize(vertical + waves + n2 * 0.45)
        red = 120 + 135 * heat
        green = 22 + 130 * np.clip(heat - 0.15, 0, 1)
        blue = 8 + 42 * np.clip(heat - 0.55, 0, 1)
        return Image.fromarray(np.clip(np.stack([red, green, blue], axis=-1), 0, 255).astype(np.uint8), mode="RGB")
    if kind in {"leaf", "green_leaf", "vein"}:
        ridges = np.abs(np.sin((xx * 0.028 - yy * 0.014 + n1 * 3.0) * np.pi))
        base = np.stack(
            [
                24 + 36 * n1,
                88 + 130 * n2 + 60 * (ridges > 0.92),
                42 + 60 * n1,
            ],
            axis=-1,
        )
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    if kind in {"flower", "object_flower"}:
        img = Image.new("RGB", (size, size), (8, 10, 18))
        draw = ImageDraw.Draw(img, "RGBA")
        rng = np.random.default_rng(seed)
        for _ in range(max(12, size // 18)):
            cx = int(rng.integers(0, size))
            cy = int(rng.integers(0, size))
            r = int(rng.integers(size // 26, size // 12))
            color = tuple(int(v) for v in rng.choice([[240, 72, 108], [255, 186, 72], [168, 92, 235], [245, 245, 238]]))
            for k in range(6):
                angle = 2 * np.pi * k / 6
                px = cx + int(np.cos(angle) * r * 0.55)
                py = cy + int(np.sin(angle) * r * 0.55)
                draw.ellipse((px - r, py - r // 2, px + r, py + r // 2), fill=(*color, 205))
            draw.ellipse((cx - r // 3, cy - r // 3, cx + r // 3, cy + r // 3), fill=(255, 225, 95, 235))
        return img
    raise ValueError(f"Unknown element kind: {kind}")


def write_default_elements(out_dir: str | Path, size: int = 512) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, kind in enumerate(["blue_stone", "ember", "leaf", "flower"]):
        img = make_element(kind, size=size, seed=13 + idx * 17)
        path = out / f"{kind}.png"
        img.save(path)
        paths.append(path)
    return paths
