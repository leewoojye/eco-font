from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .font_render import distance_transform, skeleton_map
from .metrics import CandidateMetrics, candidate_score, evaluate_candidate


@dataclass(frozen=True)
class RuleCandidate:
    name: str
    cut_mask: np.ndarray
    params: dict[str, float | int | str]
    metrics: CandidateMetrics | None = None

    def with_metrics(self, metrics: CandidateMetrics) -> "RuleCandidate":
        return RuleCandidate(self.name, self.cut_mask, self.params, metrics)


def _interior_mask(glyph: np.ndarray, margin: float = 0.18) -> np.ndarray:
    dist = distance_transform(glyph)
    binary = glyph > 0.08
    return (binary & (dist >= margin)).astype(np.uint8)


def _circle_mask(shape: tuple[int, int], cx: int, cy: int, radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 1.0, thickness=-1, lineType=cv2.LINE_AA)
    return mask


def dot_holes(
    glyph: np.ndarray,
    radius: int,
    spacing: int,
    phase_x: int = 0,
    phase_y: int = 0,
    margin: float = 0.2,
) -> np.ndarray:
    h, w = glyph.shape
    interior = _interior_mask(glyph, margin=margin)
    cut = np.zeros_like(glyph, dtype=np.float32)
    for y in range(phase_y + spacing // 2, h, spacing):
        for x in range(phase_x + spacing // 2, w, spacing):
            if interior[y, x] == 0:
                continue
            cut = np.maximum(cut, _circle_mask(glyph.shape, x, y, radius))
    cut *= interior.astype(np.float32)
    return np.clip(cut, 0.0, 1.0)


def line_holes(
    glyph: np.ndarray,
    thickness: int,
    spacing: int,
    angle: str = "horizontal",
    margin: float = 0.18,
) -> np.ndarray:
    h, w = glyph.shape
    interior = _interior_mask(glyph, margin=margin)
    cut = np.zeros_like(glyph, dtype=np.float32)
    if angle == "vertical":
        for x in range(spacing // 2, w, spacing):
            cv2.line(cut, (x, 0), (x, h - 1), 1.0, thickness=thickness, lineType=cv2.LINE_AA)
    elif angle == "diagonal":
        for offset in range(-h, w + h, spacing):
            cv2.line(cut, (offset, 0), (offset + h, h - 1), 1.0, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        for y in range(spacing // 2, h, spacing):
            cv2.line(cut, (0, y), (w - 1, y), 1.0, thickness=thickness, lineType=cv2.LINE_AA)
    cut *= interior.astype(np.float32)
    return np.clip(cut, 0.0, 1.0)


def skeleton_dots(
    glyph: np.ndarray,
    radius: int,
    every: int,
    margin: float = 0.16,
) -> np.ndarray:
    interior = _interior_mask(glyph, margin=margin)
    skel = skeleton_map(glyph) > 0
    ys, xs = np.where(skel & (interior > 0))
    cut = np.zeros_like(glyph, dtype=np.float32)
    if len(xs) == 0:
        return cut
    for i in range(0, len(xs), max(1, every)):
        cut = np.maximum(cut, _circle_mask(glyph.shape, int(xs[i]), int(ys[i]), radius))
    cut *= interior.astype(np.float32)
    return np.clip(cut, 0.0, 1.0)


def soft_center_cut(glyph: np.ndarray, strength: float = 0.65, margin: float = 0.12) -> np.ndarray:
    dist = distance_transform(glyph)
    interior = _interior_mask(glyph, margin=margin).astype(np.float32)
    cut = np.clip((dist - margin) / max(1e-6, 1.0 - margin), 0.0, 1.0)
    return np.clip(cut * strength * interior, 0.0, 1.0).astype(np.float32)


def generate_rule_candidates(glyph: np.ndarray, target_saving: float) -> list[RuleCandidate]:
    """Generate Ecofont-style pseudo-label candidates for one glyph."""
    target = float(np.clip(target_saving, 0.03, 0.65))
    candidates: list[RuleCandidate] = []

    # Dots mimic classic Ecofont holes. Search a small deterministic grid.
    radii = [1, 2, 3, 4]
    spacings = [6, 8, 10, 12, 14, 18, 22]
    margins = [0.14, 0.18, 0.22, 0.28]
    for radius in radii:
        for spacing in spacings:
            for margin in margins:
                for phase in [0, spacing // 3]:
                    mask = dot_holes(glyph, radius, spacing, phase_x=phase, phase_y=phase, margin=margin)
                    if mask.max() > 0:
                        candidates.append(
                            RuleCandidate(
                                "dot_holes",
                                mask,
                                {"radius": radius, "spacing": spacing, "margin": margin, "phase": phase},
                            )
                        )

    # Thin line cuts are useful for high target savings and ablations.
    for thickness in [1, 2]:
        for spacing in [7, 10, 14, 18]:
            for angle in ["horizontal", "vertical", "diagonal"]:
                mask = line_holes(glyph, thickness=thickness, spacing=spacing, angle=angle)
                if mask.max() > 0:
                    candidates.append(
                        RuleCandidate(
                            "line_holes",
                            mask,
                            {"thickness": thickness, "spacing": spacing, "angle": angle},
                        )
                    )

    for radius in [1, 2]:
        for every in [4, 7, 10, 14]:
            mask = skeleton_dots(glyph, radius=radius, every=every)
            if mask.max() > 0:
                candidates.append(RuleCandidate("skeleton_dots", mask, {"radius": radius, "every": every}))

    for strength in [0.25, 0.4, 0.55, 0.7]:
        mask = soft_center_cut(glyph, strength=strength)
        if mask.max() > 0:
            candidates.append(RuleCandidate("soft_center_cut", mask, {"strength": strength}))

    # Empty fallback keeps dataset building robust for very thin glyphs.
    candidates.append(RuleCandidate("identity", np.zeros_like(glyph, dtype=np.float32), {"target": target}))
    return candidates


def select_best_candidate(glyph: np.ndarray, target_saving: float) -> RuleCandidate:
    best: RuleCandidate | None = None
    best_score = float("inf")
    for candidate in generate_rule_candidates(glyph, target_saving):
        metrics = evaluate_candidate(glyph, candidate.cut_mask)
        scored = candidate.with_metrics(metrics)
        score = candidate_score(metrics, target_saving)
        if score < best_score:
            best = scored
            best_score = score
    if best is None:
        raise RuntimeError("No rule candidates generated")
    return best
