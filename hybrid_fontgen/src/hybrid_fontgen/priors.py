from __future__ import annotations

import cv2
import numpy as np

from .render import coordinate_maps, distance_pixels, distance_transform, skeleton_map


STYLES = ["contour", "centerline", "edge", "diagonal"]
EXTRA_INFERENCE_STYLES = ["inline_void", "dot_void", "ribbon_void"]


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
    spacing = float(np.interp(target_saving, [0.35, 0.75], [2.8, 5.2]))
    thickness = float(np.interp(target_saving, [0.35, 0.75], [0.75, 0.48]))
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
    sigma = float(np.interp(target_saving, [0.35, 0.75], [1.7, 0.9]))
    prior = np.exp(-(dist**2) / (2.0 * sigma**2)).astype(np.float32)
    prior *= (glyph > 0.05).astype(np.float32)
    return normalize(prior)


def diagonal_prior(glyph: np.ndarray, target_saving: float) -> np.ndarray:
    size = glyph.shape[0]
    xx, yy = coordinate_maps(size)
    dist = distance_pixels(glyph)
    core = np.clip((dist - 1.0) / 4.0, 0.0, 1.0)
    spacing = float(np.interp(target_saving, [0.35, 0.75], [9.0, 5.6]))
    coords = np.arange(size, dtype=np.float32)
    py, px = np.meshgrid(coords, coords, indexing="ij")
    diag_a = 0.5 + 0.5 * np.cos((px + py) * 2.0 * np.pi / spacing)
    diag_b = 0.5 + 0.5 * np.cos((px - py) * 2.0 * np.pi / (spacing * 1.25))
    rhythm = 0.65 * diag_a + 0.35 * diag_b
    center = centerline_prior(glyph, target_saving)
    prior = (0.55 * center + 0.45 * rhythm * core) * (glyph > 0.05)
    return normalize(cv2.GaussianBlur(prior.astype(np.float32), (3, 3), 0))


def _interior_gate(glyph: np.ndarray) -> np.ndarray:
    dist = distance_pixels(glyph)
    return np.clip((dist - 1.4) / 3.4, 0.0, 1.0).astype(np.float32) * (glyph > 0.05)


def _void_pattern_strength(glyph: np.ndarray, style: str, target_saving: float) -> np.ndarray:
    size = glyph.shape[0]
    coords = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    if style == "dot_void":
        spacing = float(np.interp(target_saving, [0.35, 0.75], [13.0, 8.0]))
        radius = float(np.interp(target_saving, [0.35, 0.75], [2.1, 3.2]))
        gx = np.mod(xx + spacing * 0.33, spacing) - spacing / 2.0
        gy = np.mod(yy + spacing * 0.17, spacing) - spacing / 2.0
        radial = np.sqrt(gx * gx + gy * gy)
        return np.clip(1.0 - radial / max(radius, 1e-3), 0.0, 1.0).astype(np.float32)
    if style == "ribbon_void":
        dist = distance_pixels(glyph)
        spacing = float(np.interp(target_saving, [0.35, 0.75], [6.6, 4.4]))
        phase = np.mod(dist + 0.2, spacing) / spacing
        bands = np.maximum(
            np.exp(-((phase - 0.32) ** 2) / 0.010),
            np.exp(-((phase - 0.70) ** 2) / 0.013),
        )
        return np.clip(bands, 0.0, 1.0).astype(np.float32)
    spacing = float(np.interp(target_saving, [0.35, 0.75], [10.0, 6.0]))
    primary = 0.5 + 0.5 * np.cos((xx * 0.88 + yy * 0.42) * 2.0 * np.pi / spacing)
    secondary = 0.5 + 0.5 * np.cos((xx * -0.30 + yy * 1.00) * 2.0 * np.pi / (spacing * 1.35))
    return np.clip(0.72 * primary + 0.28 * secondary, 0.0, 1.0).astype(np.float32)


def _remove_until_budget(
    glyph: np.ndarray,
    keep: np.ndarray,
    candidates: np.ndarray,
    priority: np.ndarray,
    keep_budget: float,
) -> np.ndarray:
    current = float(glyph[keep].sum())
    if current <= keep_budget:
        return keep
    ys, xs = np.where(candidates & keep)
    if len(xs) == 0:
        return keep
    values = priority[ys, xs]
    order = np.argsort(-values)
    cumulative = np.cumsum(glyph[ys[order], xs[order]])
    remove_budget = current - keep_budget
    remove_count = int(np.searchsorted(cumulative, remove_budget, side="left")) + 1
    remove_count = max(0, min(remove_count, len(order)))
    if remove_count > 0:
        selected = order[:remove_count]
        keep[ys[selected], xs[selected]] = False
    return keep


def void_punch_target(glyph: np.ndarray, style: str, target_saving: float) -> tuple[np.ndarray, np.ndarray]:
    """Start from the original stroke and punch interior holes before thinning fallback."""
    inside = glyph > 0.05
    if not inside.any():
        return np.zeros_like(glyph, dtype=np.float32), np.zeros_like(glyph, dtype=np.float32)
    original_ink = float(glyph[inside].sum())
    keep_budget = max(1.0, original_ink * (1.0 - float(np.clip(target_saving, 0.0, 0.9))))
    dist = distance_pixels(glyph)
    pattern = _void_pattern_strength(glyph, style, target_saving)
    distance_score = normalize(dist)
    priority = (0.76 * pattern + 0.24 * distance_score).astype(np.float32)
    priority *= inside.astype(np.float32)
    skel = skeleton_map(glyph) > 0
    skel_full = cv2.dilate(skel.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > 0
    skel_guard = skel_full & (pattern < 0.58)
    edge_width = float(np.interp(target_saving, [0.35, 0.75], [1.35, 0.75]))
    edge_guard = dist <= edge_width
    protected = (edge_guard | skel_guard) & inside

    keep = inside.copy()
    patterned = inside & (~protected) & (pattern > 0.08)
    keep = _remove_until_budget(glyph, keep, patterned, priority, keep_budget)

    # If the requested saving is too high for holes alone, remove less salient inner pixels.
    current = float(glyph[keep].sum())
    if current > keep_budget:
        inner = inside & (~protected)
        keep = _remove_until_budget(glyph, keep, inner, 0.35 * priority + 0.65 * distance_score, keep_budget)
    current = float(glyph[keep].sum())
    if current > keep_budget:
        soft_edge = inside & (dist > 0.45) & (~skel_guard)
        keep = _remove_until_budget(glyph, keep, soft_edge, distance_score, keep_budget)

    out = glyph * keep.astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32), priority


def inline_void_prior(glyph: np.ndarray, target_saving: float) -> np.ndarray:
    """Preserve edges and skeleton while cutting white channels through stroke interiors."""
    dist = distance_pixels(glyph)
    inside = glyph > 0.05
    edge = edge_prior(glyph)
    center = centerline_prior(glyph, target_saving)
    contour = contour_prior(glyph, target_saving)
    gate = _interior_gate(glyph)
    spacing = float(np.interp(target_saving, [0.35, 0.75], [8.0, 5.0]))
    coords = np.arange(glyph.shape[0], dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    channel_phase = 0.5 + 0.5 * np.cos((xx * 0.95 + yy * 0.35) * 2.0 * np.pi / spacing)
    void_channels = (channel_phase > 0.68).astype(np.float32) * gate
    thick_core = np.clip((dist - 2.0) / 3.0, 0.0, 1.0)
    score = 0.30 * edge + 0.36 * center + 0.24 * contour + 0.10 * thick_core
    score *= (1.0 - 0.72 * void_channels)
    score *= inside.astype(np.float32)
    return normalize(cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0))


def dot_void_prior(glyph: np.ndarray, target_saving: float) -> np.ndarray:
    """Create dotted internal voids without removing the outer contour entirely."""
    inside = glyph > 0.05
    gate = _interior_gate(glyph)
    edge = edge_prior(glyph)
    center = centerline_prior(glyph, target_saving)
    contour = contour_prior(glyph, target_saving)
    spacing = float(np.interp(target_saving, [0.35, 0.75], [13.0, 8.0]))
    radius = float(np.interp(target_saving, [0.35, 0.75], [1.6, 2.5]))
    coords = np.arange(glyph.shape[0], dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    gx = np.mod(xx + spacing * 0.33, spacing) - spacing / 2.0
    gy = np.mod(yy + spacing * 0.17, spacing) - spacing / 2.0
    dots = ((gx * gx + gy * gy) <= radius * radius).astype(np.float32) * gate
    score = 0.38 * contour + 0.32 * center + 0.25 * edge + 0.05 * glyph
    score *= (1.0 - 0.82 * dots)
    score *= inside.astype(np.float32)
    return normalize(cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0))


def ribbon_void_prior(glyph: np.ndarray, target_saving: float) -> np.ndarray:
    """Keep parallel ribbon-like strokes and punch broader mid-stroke gaps."""
    inside = glyph > 0.05
    gate = _interior_gate(glyph)
    contour = contour_prior(glyph, target_saving)
    center = centerline_prior(glyph, target_saving)
    dist = distance_pixels(glyph)
    spacing = float(np.interp(target_saving, [0.35, 0.75], [6.5, 4.6]))
    phase = np.mod(dist + 0.2, spacing)
    interior_gaps = (phase < spacing * 0.34).astype(np.float32) * gate
    score = 0.48 * contour + 0.32 * center + 0.14 * edge_prior(glyph) + 0.06 * glyph
    score *= (1.0 - 0.78 * interior_gaps)
    score *= inside.astype(np.float32)
    return normalize(cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0))


def style_prior(glyph: np.ndarray, style: str, target_saving: float) -> np.ndarray:
    if style == "contour":
        return contour_prior(glyph, target_saving)
    if style == "centerline":
        return centerline_prior(glyph, target_saving)
    if style == "edge":
        contour = contour_prior(glyph, target_saving)
        center = centerline_prior(glyph, target_saving)
        edge = edge_prior(glyph)
        # Keep edge accents, but rely on center/bands for readability.
        return normalize(0.42 * contour + 0.38 * center + 0.20 * edge)
    if style == "diagonal":
        return diagonal_prior(glyph, target_saving)
    if style == "inline_void":
        return inline_void_prior(glyph, target_saving)
    if style == "dot_void":
        return dot_void_prior(glyph, target_saving)
    if style == "ribbon_void":
        return ribbon_void_prior(glyph, target_saving)
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
    out[ys[selected], xs[selected]] = 1.0
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    out *= inside.astype(np.float32)
    return out.astype(np.float32)


def make_target(glyph: np.ndarray, style: str, target_saving: float) -> tuple[np.ndarray, np.ndarray]:
    if style in EXTRA_INFERENCE_STYLES:
        return void_punch_target(glyph, style, target_saving)
    score = style_prior(glyph, style, target_saving)
    target = project_to_budget(score, glyph, target_saving)
    return target, score


def input_channels(glyph: np.ndarray, style: str, target_saving: float) -> np.ndarray:
    dist = distance_transform(glyph)
    skel = skeleton_map(glyph)
    prior = style_prior(glyph, style, target_saving)
    edge = edge_prior(glyph)
    target = np.full_like(glyph, float(target_saving), dtype=np.float32)
    xx, yy = coordinate_maps(glyph.shape[0])
    return np.stack([glyph, dist, skel, prior, edge, target, xx, yy], axis=0).astype(np.float32)
