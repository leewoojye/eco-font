from __future__ import annotations

import cv2
import numpy as np

from .render import coordinate_maps, distance_pixels, distance_transform, skeleton_map


ECO_STYLES = ["contour", "centerline", "edge", "diagonal"]


def normalize(image: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32, copy=False)
    max_value = float(out.max(initial=0.0))
    if max_value > 0:
        out = out / max_value
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def edge_prior(glyph: np.ndarray) -> np.ndarray:
    dist = distance_pixels(glyph)
    prior = np.exp(-dist / 1.45).astype(np.float32) * (glyph > 0.05)
    return normalize(prior)


def contour_prior(glyph: np.ndarray, target_saving: float) -> np.ndarray:
    dist = distance_pixels(glyph)
    spacing = float(np.interp(target_saving, [0.30, 0.75], [2.7, 5.1]))
    thickness = float(np.interp(target_saving, [0.30, 0.75], [0.80, 0.48]))
    phase = np.mod(dist + 0.35, spacing)
    close = np.minimum(phase, spacing - phase)
    bands = np.exp(-(close**2) / (2.0 * thickness**2)).astype(np.float32)
    bands *= (glyph > 0.05).astype(np.float32)
    return normalize(cv2.GaussianBlur(bands, (3, 3), 0))


def centerline_prior(glyph: np.ndarray, target_saving: float) -> np.ndarray:
    skel = skeleton_map(glyph)
    if skel.max(initial=0.0) == 0:
        return np.zeros_like(glyph, dtype=np.float32)
    dist = cv2.distanceTransform((1 - skel.astype(np.uint8)), cv2.DIST_L2, 5)
    sigma = float(np.interp(target_saving, [0.30, 0.75], [1.8, 0.9]))
    prior = np.exp(-(dist**2) / (2.0 * sigma**2)).astype(np.float32)
    prior *= (glyph > 0.05).astype(np.float32)
    return normalize(prior)


def diagonal_prior(glyph: np.ndarray, target_saving: float) -> np.ndarray:
    size = glyph.shape[0]
    dist = distance_pixels(glyph)
    core = np.clip((dist - 1.0) / 4.0, 0.0, 1.0)
    spacing = float(np.interp(target_saving, [0.30, 0.75], [9.0, 5.6]))
    coords = np.arange(size, dtype=np.float32)
    py, px = np.meshgrid(coords, coords, indexing="ij")
    diag_a = 0.5 + 0.5 * np.cos((px + py) * 2.0 * np.pi / spacing)
    diag_b = 0.5 + 0.5 * np.cos((px - py) * 2.0 * np.pi / (spacing * 1.25))
    rhythm = 0.65 * diag_a + 0.35 * diag_b
    center = centerline_prior(glyph, target_saving)
    prior = (0.55 * center + 0.45 * rhythm * core) * (glyph > 0.05)
    return normalize(cv2.GaussianBlur(prior.astype(np.float32), (3, 3), 0))


def style_prior(glyph: np.ndarray, style: str, target_saving: float) -> np.ndarray:
    if style == "contour":
        return contour_prior(glyph, target_saving)
    if style == "centerline":
        return centerline_prior(glyph, target_saving)
    if style == "edge":
        contour = contour_prior(glyph, target_saving)
        center = centerline_prior(glyph, target_saving)
        edge = edge_prior(glyph)
        return normalize(0.42 * contour + 0.38 * center + 0.20 * edge)
    if style == "diagonal":
        return diagonal_prior(glyph, target_saving)
    raise ValueError(f"Unknown style: {style}")


def project_to_budget(score: np.ndarray, glyph: np.ndarray, target_saving: float) -> np.ndarray:
    inside = glyph > 0.05
    if not inside.any():
        return np.zeros_like(glyph, dtype=np.float32)
    original_ink = float(glyph[inside].sum())
    keep_budget = max(1.0, original_ink * (1.0 - float(np.clip(target_saving, 0.0, 0.9))))
    ys, xs = np.where(inside)
    values = score[ys, xs]
    order = np.argsort(-values)
    out = np.zeros_like(glyph, dtype=np.float32)
    cumulative = np.cumsum(glyph[ys[order], xs[order]])
    keep_count = int(np.searchsorted(cumulative, keep_budget, side="left")) + 1
    keep_count = max(1, min(keep_count, len(order)))
    selected = order[:keep_count]
    out[ys[selected], xs[selected]] = glyph[ys[selected], xs[selected]]
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    out *= inside.astype(np.float32)
    return out.astype(np.float32)


def make_target(glyph: np.ndarray, style: str, target_saving: float) -> tuple[np.ndarray, np.ndarray]:
    score = style_prior(glyph, style, target_saving)
    target = project_to_budget(score, glyph, target_saving)
    return target, score


def input_hint_channels(glyph: np.ndarray, style: str, target_saving: float) -> np.ndarray:
    dist = distance_transform(glyph)
    skel = skeleton_map(glyph)
    prior = style_prior(glyph, style, target_saving)
    edge = edge_prior(glyph)
    target = np.full_like(glyph, float(target_saving), dtype=np.float32)
    xx, yy = coordinate_maps(glyph.shape[0])
    return np.stack([dist, skel, prior, edge, target, xx, yy], axis=0).astype(np.float32)
