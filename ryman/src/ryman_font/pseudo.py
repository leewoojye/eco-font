from __future__ import annotations

import cv2
import numpy as np

from .render import distance_pixels, distance_transform, skeleton_map


INITIALS = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
VOWELS = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
FINALS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
VERTICAL_VOWELS = {"ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅣ"}
HORIZONTAL_VOWELS = {"ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ"}


def decompose_hangul(ch: str) -> tuple[str, str, str] | None:
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return None
    index = code - 0xAC00
    initial = index // (21 * 28)
    medial = (index % (21 * 28)) // 28
    final = index % 28
    return INITIALS[initial], VOWELS[medial], FINALS[final]


def is_cherokee(ch: str | None) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (0x13A0 <= code <= 0x13FF) or (0xAB70 <= code <= 0xABBF)


def contour_band_prior(glyph: np.ndarray, spacing: float = 3.0, thickness: float = 0.72) -> np.ndarray:
    """Nested thin lines inside glyph strokes, inspired by Ryman Eco."""
    dist = distance_pixels(glyph)
    inside = glyph > 0.05
    if not inside.any():
        return np.zeros_like(glyph, dtype=np.float32)
    # Bands at repeated distances from the contour. Offset keeps the outer line
    # visible while leaving deliberate white channels between strokes.
    phase = np.mod(dist + 0.45, spacing)
    closeness = np.minimum(phase, spacing - phase)
    bands = np.exp(-(closeness**2) / (2.0 * thickness**2))
    bands *= inside.astype(np.float32)
    return np.clip(bands, 0.0, 1.0).astype(np.float32)


def skeleton_prior(glyph: np.ndarray) -> np.ndarray:
    skel = skeleton_map(glyph)
    if skel.max() == 0:
        return np.zeros_like(glyph, dtype=np.float32)
    dist_to_skel = cv2.distanceTransform((1 - skel.astype(np.uint8)), cv2.DIST_L2, 5)
    prior = np.exp(-dist_to_skel / 2.3).astype(np.float32)
    prior *= (glyph > 0.05).astype(np.float32)
    return prior


def centerline_prior(glyph: np.ndarray, sigma: float = 1.35) -> np.ndarray:
    """Thin monoline construction prior around the medial axis."""
    skel = skeleton_map(glyph)
    if skel.max() == 0:
        return np.zeros_like(glyph, dtype=np.float32)
    dist_to_skel = cv2.distanceTransform((1 - skel.astype(np.uint8)), cv2.DIST_L2, 5)
    prior = np.exp(-(dist_to_skel**2) / (2.0 * sigma**2)).astype(np.float32)
    prior *= (glyph > 0.05).astype(np.float32)
    return prior


def edge_prior(glyph: np.ndarray) -> np.ndarray:
    dist = distance_pixels(glyph)
    prior = np.exp(-dist / 1.6).astype(np.float32)
    prior *= (glyph > 0.05).astype(np.float32)
    return prior


def ryman_score_map(glyph: np.ndarray, spacing: float = 3.0) -> np.ndarray:
    bands = contour_band_prior(glyph, spacing=spacing)
    skel = skeleton_prior(glyph)
    edge = edge_prior(glyph)
    smooth = cv2.GaussianBlur(glyph.astype(np.float32), (5, 5), 0)
    score = 0.50 * bands + 0.25 * edge + 0.20 * skel + 0.05 * smooth
    score *= (glyph > 0.05).astype(np.float32)
    score = cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0)
    max_value = float(score.max())
    if max_value > 0:
        score /= max_value
    return score.astype(np.float32)


def distinct_score_map(glyph: np.ndarray, spacing: float = 4.2) -> np.ndarray:
    """A more independent Ryman-like design prior.

    This intentionally avoids treating the original outline as sacred. It
    prefers a thin architectural centerline plus interior echo lines, so the
    result reads as a redesigned display face instead of a hollowed source font.
    """
    inside = glyph > 0.05
    if not inside.any():
        return np.zeros_like(glyph, dtype=np.float32)

    dist = distance_pixels(glyph)
    center = centerline_prior(glyph, sigma=1.15)
    bands = contour_band_prior(glyph, spacing=spacing, thickness=0.52)
    edge = edge_prior(glyph)
    core_gate = np.clip((dist - 1.7) / 3.0, 0.0, 1.0).astype(np.float32)
    inner_bands = bands * (0.35 + 0.65 * core_gate)

    size = glyph.shape[0]
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    vertical_rhythm = (0.5 + 0.5 * np.cos((xx * 5.6 + yy * 1.15) * np.pi)).astype(np.float32)
    horizontal_rhythm = (0.5 + 0.5 * np.cos((yy * 4.4 - xx * 0.75) * np.pi)).astype(np.float32)
    rhythm = 0.60 * vertical_rhythm + 0.40 * horizontal_rhythm
    rhythm *= inside.astype(np.float32) * core_gate

    blurred = cv2.GaussianBlur(glyph.astype(np.float32), (7, 7), 0)
    score = 0.58 * center + 0.25 * inner_bands + 0.11 * rhythm + 0.06 * blurred * core_gate
    score *= inside.astype(np.float32)
    score *= (1.0 - 0.42 * edge)
    score = cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0)
    max_value = float(score.max())
    if max_value > 0:
        score /= max_value
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _p(box: tuple[float, float, float, float], x: float, y: float) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    return int(round(x0 + x * (x1 - x0))), int(round(y0 + y * (y1 - y0)))


def _line(canvas: np.ndarray, box: tuple[float, float, float, float], a: tuple[float, float], b: tuple[float, float], thickness: int) -> None:
    cv2.line(canvas, _p(box, *a), _p(box, *b), 255, thickness, cv2.LINE_AA)


def _ellipse(canvas: np.ndarray, box: tuple[float, float, float, float], center: tuple[float, float], axes: tuple[float, float], thickness: int) -> None:
    x0, y0, x1, y1 = box
    cx, cy = _p(box, *center)
    ax = max(2, int(round(axes[0] * (x1 - x0))))
    ay = max(2, int(round(axes[1] * (y1 - y0))))
    cv2.ellipse(canvas, (cx, cy), (ax, ay), 0, 0, 360, 255, thickness, cv2.LINE_AA)


def _canonical_fit(glyph: np.ndarray, margin_ratio: float = 0.12) -> np.ndarray:
    inside = glyph > 0.05
    if not inside.any():
        return np.zeros_like(glyph, dtype=np.float32)
    size = glyph.shape[0]
    margin = max(4, int(round(size * margin_ratio)))
    ys, xs = np.where(inside)
    crop = glyph[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    if h <= 0 or w <= 0:
        return np.zeros_like(glyph, dtype=np.float32)
    target = max(1, size - 2 * margin)
    scale = min(target / h, target / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(crop.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(glyph, dtype=np.float32)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def cherokee_canonical_score_map(glyph: np.ndarray) -> np.ndarray:
    """Font-normalized Cherokee construction prior.

    Cherokee has no Hangul-like decomposition in this experiment, so this prior
    redraws the rendered sign as a centered monoline construction. Unlike the
    distinct prior, the final projection is not restricted to the original
    outline, which makes this a script-specific canonical mode instead of a
    hollowed source-font mode.
    """
    base = _canonical_fit(glyph)
    if base.max() == 0:
        return np.zeros_like(glyph, dtype=np.float32)

    skel = skeleton_map(base)
    if skel.max() == 0:
        return distinct_score_map(glyph)
    skel_binary = (skel > 0).astype(np.uint8)
    dist_to_skel = cv2.distanceTransform(1 - skel_binary, cv2.DIST_L2, 5).astype(np.float32)
    center = np.exp(-(dist_to_skel**2) / (2.0 * 1.45**2)).astype(np.float32)

    dist_inside = distance_pixels(base)
    core_gate = np.clip((dist_inside - 1.0) / 3.0, 0.0, 1.0).astype(np.float32)
    inner_body = cv2.GaussianBlur(base.astype(np.float32), (5, 5), 0) * core_gate
    edge = edge_prior(base)

    size = glyph.shape[0]
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    rhythm = (0.5 + 0.5 * np.cos((xx * 4.9 - yy * 1.25) * np.pi)).astype(np.float32)
    rhythm *= cv2.dilate(skel_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)

    score = 0.74 * center + 0.16 * inner_body + 0.10 * rhythm
    score *= 1.0 - 0.38 * edge
    score = cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0)
    max_value = float(score.max())
    if max_value > 0:
        score /= max_value
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _draw_consonant(canvas: np.ndarray, jamo: str, box: tuple[float, float, float, float], thickness: int) -> None:
    if jamo in {"ㄱ", "ㄲ"}:
        _line(canvas, box, (0.18, 0.18), (0.82, 0.18), thickness)
        _line(canvas, box, (0.82, 0.18), (0.82, 0.78), thickness)
        if jamo == "ㄲ":
            _line(canvas, box, (0.35, 0.33), (0.72, 0.33), max(1, thickness - 1))
    elif jamo == "ㄴ":
        _line(canvas, box, (0.20, 0.18), (0.20, 0.78), thickness)
        _line(canvas, box, (0.20, 0.78), (0.82, 0.78), thickness)
    elif jamo in {"ㄷ", "ㄸ"}:
        _line(canvas, box, (0.20, 0.18), (0.20, 0.78), thickness)
        _line(canvas, box, (0.20, 0.18), (0.82, 0.18), thickness)
        _line(canvas, box, (0.20, 0.78), (0.82, 0.78), thickness)
        if jamo == "ㄸ":
            _line(canvas, box, (0.32, 0.36), (0.74, 0.36), max(1, thickness - 1))
    elif jamo == "ㄹ":
        _line(canvas, box, (0.18, 0.18), (0.82, 0.18), thickness)
        _line(canvas, box, (0.82, 0.18), (0.82, 0.46), thickness)
        _line(canvas, box, (0.82, 0.46), (0.25, 0.46), thickness)
        _line(canvas, box, (0.25, 0.46), (0.25, 0.78), thickness)
        _line(canvas, box, (0.25, 0.78), (0.84, 0.78), thickness)
    elif jamo == "ㅁ":
        _line(canvas, box, (0.20, 0.18), (0.82, 0.18), thickness)
        _line(canvas, box, (0.20, 0.18), (0.20, 0.78), thickness)
        _line(canvas, box, (0.82, 0.18), (0.82, 0.78), thickness)
        _line(canvas, box, (0.20, 0.78), (0.82, 0.78), thickness)
    elif jamo in {"ㅂ", "ㅃ"}:
        _line(canvas, box, (0.22, 0.18), (0.22, 0.78), thickness)
        _line(canvas, box, (0.78, 0.18), (0.78, 0.78), thickness)
        _line(canvas, box, (0.22, 0.18), (0.78, 0.18), thickness)
        _line(canvas, box, (0.22, 0.49), (0.78, 0.49), thickness)
        _line(canvas, box, (0.22, 0.78), (0.78, 0.78), thickness)
        if jamo == "ㅃ":
            _line(canvas, box, (0.38, 0.30), (0.38, 0.67), max(1, thickness - 1))
    elif jamo in {"ㅅ", "ㅆ"}:
        _line(canvas, box, (0.50, 0.18), (0.20, 0.78), thickness)
        _line(canvas, box, (0.50, 0.18), (0.82, 0.78), thickness)
        if jamo == "ㅆ":
            _line(canvas, box, (0.64, 0.20), (0.88, 0.72), max(1, thickness - 1))
    elif jamo == "ㅇ":
        _ellipse(canvas, box, (0.51, 0.50), (0.30, 0.31), thickness)
    elif jamo in {"ㅈ", "ㅉ"}:
        _line(canvas, box, (0.24, 0.20), (0.78, 0.20), thickness)
        _line(canvas, box, (0.51, 0.25), (0.22, 0.78), thickness)
        _line(canvas, box, (0.51, 0.25), (0.82, 0.78), thickness)
        if jamo == "ㅉ":
            _line(canvas, box, (0.66, 0.31), (0.88, 0.74), max(1, thickness - 1))
    elif jamo == "ㅊ":
        _line(canvas, box, (0.39, 0.12), (0.65, 0.12), thickness)
        _line(canvas, box, (0.24, 0.29), (0.78, 0.29), thickness)
        _line(canvas, box, (0.51, 0.34), (0.22, 0.78), thickness)
        _line(canvas, box, (0.51, 0.34), (0.82, 0.78), thickness)
    elif jamo == "ㅋ":
        _line(canvas, box, (0.18, 0.18), (0.82, 0.18), thickness)
        _line(canvas, box, (0.82, 0.18), (0.82, 0.78), thickness)
        _line(canvas, box, (0.35, 0.48), (0.82, 0.48), thickness)
    elif jamo == "ㅌ":
        _line(canvas, box, (0.20, 0.18), (0.20, 0.78), thickness)
        _line(canvas, box, (0.20, 0.18), (0.82, 0.18), thickness)
        _line(canvas, box, (0.20, 0.48), (0.72, 0.48), thickness)
        _line(canvas, box, (0.20, 0.78), (0.82, 0.78), thickness)
    elif jamo == "ㅍ":
        _line(canvas, box, (0.21, 0.18), (0.79, 0.18), thickness)
        _line(canvas, box, (0.21, 0.78), (0.79, 0.78), thickness)
        _line(canvas, box, (0.31, 0.22), (0.31, 0.74), thickness)
        _line(canvas, box, (0.69, 0.22), (0.69, 0.74), thickness)
    elif jamo == "ㅎ":
        _line(canvas, box, (0.38, 0.13), (0.66, 0.13), thickness)
        _line(canvas, box, (0.28, 0.29), (0.75, 0.29), thickness)
        _ellipse(canvas, box, (0.51, 0.62), (0.27, 0.23), thickness)
    else:
        _ellipse(canvas, box, (0.51, 0.50), (0.30, 0.31), thickness)


def _draw_vowel(canvas: np.ndarray, jamo: str, box: tuple[float, float, float, float], thickness: int) -> None:
    if jamo in {"ㅏ", "ㅑ"}:
        _line(canvas, box, (0.42, 0.12), (0.42, 0.88), thickness)
        _line(canvas, box, (0.42, 0.45), (0.82, 0.45), thickness)
        if jamo == "ㅑ":
            _line(canvas, box, (0.42, 0.64), (0.78, 0.64), thickness)
    elif jamo in {"ㅓ", "ㅕ"}:
        _line(canvas, box, (0.62, 0.12), (0.62, 0.88), thickness)
        _line(canvas, box, (0.20, 0.45), (0.62, 0.45), thickness)
        if jamo == "ㅕ":
            _line(canvas, box, (0.24, 0.64), (0.62, 0.64), thickness)
    elif jamo in {"ㅐ", "ㅒ"}:
        _line(canvas, box, (0.34, 0.12), (0.34, 0.88), thickness)
        _line(canvas, box, (0.70, 0.12), (0.70, 0.88), thickness)
        _line(canvas, box, (0.34, 0.45), (0.64, 0.45), thickness)
        if jamo == "ㅒ":
            _line(canvas, box, (0.34, 0.64), (0.64, 0.64), thickness)
    elif jamo in {"ㅔ", "ㅖ"}:
        _line(canvas, box, (0.28, 0.45), (0.54, 0.45), thickness)
        if jamo == "ㅖ":
            _line(canvas, box, (0.28, 0.64), (0.54, 0.64), thickness)
        _line(canvas, box, (0.54, 0.12), (0.54, 0.88), thickness)
        _line(canvas, box, (0.82, 0.12), (0.82, 0.88), thickness)
    elif jamo in {"ㅗ", "ㅛ"}:
        _line(canvas, box, (0.16, 0.70), (0.84, 0.70), thickness)
        _line(canvas, box, (0.50, 0.24), (0.50, 0.70), thickness)
        if jamo == "ㅛ":
            _line(canvas, box, (0.34, 0.36), (0.34, 0.70), thickness)
            _line(canvas, box, (0.66, 0.36), (0.66, 0.70), thickness)
    elif jamo in {"ㅜ", "ㅠ"}:
        _line(canvas, box, (0.16, 0.28), (0.84, 0.28), thickness)
        _line(canvas, box, (0.50, 0.28), (0.50, 0.76), thickness)
        if jamo == "ㅠ":
            _line(canvas, box, (0.34, 0.28), (0.34, 0.64), thickness)
            _line(canvas, box, (0.66, 0.28), (0.66, 0.64), thickness)
    elif jamo == "ㅡ":
        _line(canvas, box, (0.16, 0.55), (0.84, 0.55), thickness)
    elif jamo == "ㅣ":
        _line(canvas, box, (0.52, 0.12), (0.52, 0.88), thickness)
    else:
        _line(canvas, box, (0.18, 0.68), (0.82, 0.68), thickness)
        _line(canvas, box, (0.50, 0.20), (0.50, 0.68), thickness)
        _line(canvas, box, (0.62, 0.18), (0.62, 0.86), thickness)


def canonical_score_map(glyph: np.ndarray, ch: str | None) -> np.ndarray:
    if is_cherokee(ch):
        return cherokee_canonical_score_map(glyph)
    parts = decompose_hangul(ch) if ch else None
    if parts is None:
        return distinct_score_map(glyph)
    initial, vowel, final = parts
    size = glyph.shape[0]
    canvas = np.zeros((size, size), dtype=np.uint8)
    thickness = max(2, int(round(size / 34)))
    has_final = bool(final)
    if vowel in VERTICAL_VOWELS:
        top_h = 0.70 if has_final else 0.88
        _draw_consonant(canvas, initial, (7, 8, 53, top_h * size), thickness)
        _draw_vowel(canvas, vowel, (48, 7, 90, top_h * size + 2), thickness)
    elif vowel in HORIZONTAL_VOWELS:
        top_h = 0.58 if has_final else 0.66
        _draw_consonant(canvas, initial, (18, 7, 78, top_h * size), thickness)
        _draw_vowel(canvas, vowel, (13, top_h * size - 2, 83, (0.77 if has_final else 0.91) * size), thickness)
    else:
        top_h = 0.68 if has_final else 0.86
        _draw_consonant(canvas, initial, (7, 8, 50, top_h * size), thickness)
        _draw_vowel(canvas, vowel, (47, 8, 90, top_h * size), thickness)
    if has_final:
        _draw_consonant(canvas, final[0], (16, 61, 82, 91), max(2, thickness - 1))

    base = canvas.astype(np.float32) / 255.0
    score = cv2.GaussianBlur(base, (5, 5), 0)
    score = np.maximum(score, base * 0.92)
    score = cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0)
    max_value = float(score.max())
    if max_value > 0:
        score /= max_value
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def project_to_ink_budget(score: np.ndarray, glyph: np.ndarray, target_saving: float) -> np.ndarray:
    inside = glyph > 0.05
    if not inside.any():
        return np.zeros_like(glyph, dtype=np.float32)
    original_ink = float(glyph[inside].sum())
    keep_budget = max(1, int(round(original_ink * (1.0 - float(np.clip(target_saving, 0.0, 0.9))))))
    ys, xs = np.where(inside)
    values = score[ys, xs]
    order = np.argsort(-values)
    keep_budget = max(1, min(keep_budget, len(order)))
    selected = order[:keep_budget]
    out = np.zeros_like(glyph, dtype=np.float32)
    out[ys[selected], xs[selected]] = 1.0
    # Clean tiny isolated noise while preserving the thin-line texture.
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    out *= inside.astype(np.float32)
    return out.astype(np.float32)


def project_canvas_to_ink_budget(score: np.ndarray, reference_glyph: np.ndarray, target_saving: float) -> np.ndarray:
    original_ink = max(1.0, float(reference_glyph.sum()))
    keep_budget = max(1, int(round(original_ink * (1.0 - float(np.clip(target_saving, 0.0, 0.95))))))
    candidate = score > 0.002
    if not candidate.any():
        return np.zeros_like(score, dtype=np.float32)
    ys, xs = np.where(candidate)
    values = score[ys, xs]
    order = np.argsort(-values)
    keep_budget = max(1, min(keep_budget, len(order)))
    selected = order[:keep_budget]
    out = np.zeros_like(score, dtype=np.float32)
    out[ys[selected], xs[selected]] = 1.0
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return out.astype(np.float32)


def make_ryman_target(glyph: np.ndarray, target_saving: float, spacing: float | None = None, style: str = "contour", char: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if spacing is None:
        # Higher savings can afford wider white channels.
        spacing = float(np.interp(target_saving, [0.25, 0.60], [2.65, 4.15]))
    if style == "distinct":
        spacing = float(np.interp(target_saving, [0.45, 0.75], [3.75, 5.65]))
        score = distinct_score_map(glyph, spacing=spacing)
        target = project_to_ink_budget(score, glyph, target_saving)
    elif style == "canonical":
        score = canonical_score_map(glyph, char)
        target = project_canvas_to_ink_budget(score, glyph, target_saving)
    elif style == "contour":
        score = ryman_score_map(glyph, spacing=spacing)
        target = project_to_ink_budget(score, glyph, target_saving)
    else:
        raise ValueError(f"Unknown target style: {style}")
    return target, score


def input_channels(glyph: np.ndarray, target_saving: float, style: str = "contour", char: str | None = None) -> np.ndarray:
    dist = distance_transform(glyph)
    skel = skeleton_map(glyph)
    if style == "distinct":
        spacing = float(np.interp(target_saving, [0.45, 0.75], [3.75, 5.65]))
        bands = distinct_score_map(glyph, spacing=spacing)
        edge = centerline_prior(glyph, sigma=1.25)
    elif style == "canonical":
        spacing = float(np.interp(target_saving, [0.45, 0.75], [3.75, 5.65]))
        bands = canonical_score_map(glyph, char) if char else distinct_score_map(glyph, spacing=spacing)
        edge = centerline_prior(glyph, sigma=1.25)
    elif style == "contour":
        bands = contour_band_prior(glyph, spacing=float(np.interp(target_saving, [0.25, 0.60], [2.65, 4.15])))
        edge = edge_prior(glyph)
    else:
        raise ValueError(f"Unknown target style: {style}")
    target = np.full_like(glyph, float(target_saving), dtype=np.float32)
    coords = np.linspace(-1.0, 1.0, glyph.shape[0], dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return np.stack([glyph, dist, skel, bands, edge, target, xx], axis=0).astype(np.float32)
