from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .metrics import distance_transform
from .render import fit_to_bbox, glyph_bbox


@dataclass(frozen=True)
class FusionParams:
    alpha: float
    weight_delta: int
    width_scale: float
    slant: float
    eco_mode: str
    style_font: str


@dataclass(frozen=True)
class Candidate:
    name: str
    image: np.ndarray
    params: FusionParams
    style_family: str


def _morph(image: np.ndarray, delta: int) -> np.ndarray:
    if delta == 0:
        return image
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if delta > 0:
        for _ in range(delta):
            arr = cv2.dilate(arr, kernel, iterations=1)
    else:
        for _ in range(abs(delta)):
            arr = cv2.erode(arr, kernel, iterations=1)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    return (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _scale_width(image: np.ndarray, width_scale: float) -> np.ndarray:
    if abs(width_scale - 1.0) < 1e-3:
        return image
    h, w = image.shape
    new_w = max(1, int(round(w * width_scale)))
    resized = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_AREA)
    out = np.zeros_like(image)
    if new_w >= w:
        start = (new_w - w) // 2
        out = resized[:, start : start + w]
    else:
        start = (w - new_w) // 2
        out[:, start : start + new_w] = resized
    return out.astype(np.float32)


def _slant(image: np.ndarray, slant: float) -> np.ndarray:
    if abs(slant) < 1e-3:
        return image
    h, w = image.shape
    matrix = np.float32([[1.0, slant, -slant * h / 2.0], [0.0, 1.0, 0.0]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)


def _anti_alias(mask: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(np.clip(mask, 0.0, 1.0).astype(np.float32), (3, 3), 0).clip(0.0, 1.0)


def _inline_cut(image: np.ndarray, target_strength: float = 0.30) -> np.ndarray:
    dist = distance_transform(image)
    inside = image > 0.10
    cut = ((dist > 0.34) & inside).astype(np.float32)
    cut = cv2.GaussianBlur(cut, (3, 3), 0)
    return np.clip(image * (1.0 - target_strength * cut), 0.0, 1.0)


def _stencil_cut(image: np.ndarray, target_strength: float = 0.45) -> np.ndarray:
    h, w = image.shape
    inside = image > 0.12
    cut = np.zeros_like(image, dtype=np.float32)
    spacing = max(12, w // 6)
    thickness = max(2, w // 48)
    for x in range(spacing // 2, w, spacing):
        cv2.line(cut, (x, 0), (x, h - 1), 1.0, thickness=thickness, lineType=cv2.LINE_AA)
    cut *= inside.astype(np.float32)
    return np.clip(image * (1.0 - target_strength * cut), 0.0, 1.0)


def _edge_relief(image: np.ndarray, target_strength: float = 0.26) -> np.ndarray:
    dist = distance_transform(image)
    inside = image > 0.10
    relief = ((dist > 0.18) & (dist < 0.58) & inside).astype(np.float32)
    relief = cv2.GaussianBlur(relief, (5, 5), 0)
    return np.clip(image * (1.0 - target_strength * relief), 0.0, 1.0)


def apply_eco_mode(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "inline":
        return _inline_cut(image)
    if mode == "stencil":
        return _stencil_cut(image)
    if mode == "edge_relief":
        return _edge_relief(image)
    raise ValueError(f"Unknown eco_mode: {mode}")


def fuse_images(source: np.ndarray, style: np.ndarray, params: FusionParams) -> np.ndarray:
    target_bbox = glyph_bbox(source)
    aligned_style = fit_to_bbox(style, target_bbox, source.shape)
    source_weighted = np.clip(source, 0.0, 1.0)
    style_weighted = np.clip(aligned_style, 0.0, 1.0)
    blended = (1.0 - params.alpha) * source_weighted + params.alpha * style_weighted
    # Keep the candidate recognizable by retaining the shared high-confidence mass.
    support = np.maximum(source_weighted * 0.35, style_weighted * 0.45)
    blended = np.maximum(blended, support)
    blended = _scale_width(blended, params.width_scale)
    blended = _slant(blended, params.slant)
    blended = _morph(blended, params.weight_delta)
    blended = apply_eco_mode(blended, params.eco_mode)
    return _anti_alias(blended)


def generate_param_grid(config: dict, style_font: str) -> list[FusionParams]:
    grid = config["fusion_grid"]
    params: list[FusionParams] = []
    for alpha in grid["alpha"]:
        for weight_delta in grid["weight_delta"]:
            for width_scale in grid["width_scale"]:
                for slant in grid["slant"]:
                    for eco_mode in grid["eco_modes"]:
                        params.append(
                            FusionParams(
                                alpha=float(alpha),
                                weight_delta=int(weight_delta),
                                width_scale=float(width_scale),
                                slant=float(slant),
                                eco_mode=str(eco_mode),
                                style_font=style_font,
                            )
                        )
    return params
