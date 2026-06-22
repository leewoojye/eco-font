from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidateScore:
    ink_saving: float
    readability_margin: float
    target_similarity: float
    aesthetic: float
    topology_penalty: float
    diversity_bonus: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ink_saving": self.ink_saving,
            "readability_margin": self.readability_margin,
            "target_similarity": self.target_similarity,
            "aesthetic": self.aesthetic,
            "topology_penalty": self.topology_penalty,
            "diversity_bonus": self.diversity_bonus,
            "total": self.total,
        }


def ink_area(image: np.ndarray) -> float:
    return float(np.clip(image, 0.0, 1.0).sum())


def ink_saving(source: np.ndarray, candidate: np.ndarray) -> float:
    original = ink_area(source)
    if original <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - ink_area(candidate) / original, -1.0, 1.0))


def binary(image: np.ndarray, threshold: float = 0.16) -> np.ndarray:
    return (image > threshold).astype(np.uint8)


def distance_transform(image: np.ndarray) -> np.ndarray:
    mask = binary(image, 0.12)
    if mask.max() == 0:
        return np.zeros_like(image, dtype=np.float32)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    if dist.max() > 0:
        dist /= float(dist.max())
    return dist


def component_count(image: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(binary(image), connectivity=8)
    return max(0, int(count) - 1)


def hole_count(image: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(binary(image), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int(sum(1 for item in hierarchy[0] if int(item[3]) >= 0))


def topology_penalty(source: np.ndarray, candidate: np.ndarray) -> float:
    comp_delta = abs(component_count(source) - component_count(candidate))
    hole_delta = abs(hole_count(source) - hole_count(candidate))
    return float(comp_delta + 0.35 * hole_delta)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.GaussianBlur(a.astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(b.astype(np.float32), (5, 5), 0)
    av = a - float(a.mean())
    bv = b - float(b.mean())
    denom = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip((av * bv).sum() / denom, -1.0, 1.0))


def readability_margin(candidate: np.ndarray, char: str, template_bank: dict[str, list[np.ndarray]]) -> tuple[float, float]:
    target_scores = [normalized_cross_correlation(candidate, template) for template in template_bank.get(char, [])]
    if not target_scores:
        return 0.0, 0.0
    target = max(target_scores)
    other = -1.0
    for other_char, templates in template_bank.items():
        if other_char == char:
            continue
        for template in templates:
            other = max(other, normalized_cross_correlation(candidate, template))
    return float(target - other), float(target)


def _bbox_stats(image: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(image > 0.05)
    h, w = image.shape
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = float(xs.mean()) / max(1.0, w - 1)
    cy = float(ys.mean()) / max(1.0, h - 1)
    bw = float(xs.max() - xs.min() + 1) / w
    bh = float(ys.max() - ys.min() + 1) / h
    return cx, cy, bw, bh


def aesthetic_score(image: np.ndarray) -> float:
    mask = binary(image, 0.12)
    if mask.max() == 0:
        return 0.0
    ink_ratio = float(mask.mean())
    cx, cy, bw, bh = _bbox_stats(image)
    balance = 1.0 - min(1.0, abs(cx - 0.5) * 2.4 + abs(cy - 0.52) * 2.0)
    occupancy = 1.0 - min(1.0, abs(bw - 0.52) * 1.3 + abs(bh - 0.68) * 0.9)
    ink_target = 1.0 - min(1.0, abs(ink_ratio - 0.18) / 0.20)

    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        smoothness = 0.0
    else:
        area = max(1.0, float(sum(cv2.contourArea(c) for c in contours)))
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
        ragged = perimeter / np.sqrt(area)
        smoothness = 1.0 - min(1.0, max(0.0, ragged - 8.0) / 18.0)

    dist = distance_transform(image)
    inside = dist[mask > 0]
    stroke = 0.0 if inside.size == 0 else float(np.clip(inside.mean() * 2.5, 0.0, 1.0))
    edge = cv2.Canny((image * 255).astype(np.uint8), 30, 150)
    edge_density = float(edge.mean() / 255.0)
    detail = 1.0 - min(1.0, abs(edge_density - 0.045) / 0.08)

    score = 0.23 * balance + 0.17 * occupancy + 0.18 * ink_target + 0.18 * smoothness + 0.14 * stroke + 0.10 * detail
    return float(np.clip(score, 0.0, 1.0))


def score_candidate(
    source: np.ndarray,
    candidate: np.ndarray,
    char: str,
    template_bank: dict[str, list[np.ndarray]],
    target_ink_saving: float,
    weights: dict[str, float],
    diversity_bonus: float = 0.0,
) -> CandidateScore:
    saving = ink_saving(source, candidate)
    margin, target = readability_margin(candidate, char, template_bank)
    aesthetic = aesthetic_score(candidate)
    topo = topology_penalty(source, candidate)
    ink_term = 1.0 - min(1.0, abs(saving - target_ink_saving) / max(0.08, target_ink_saving))
    readability_term = np.clip((margin + 0.25) / 0.65, 0.0, 1.0)
    topology_term = 1.0 / (1.0 + topo)
    total = (
        weights.get("ink", 1.0) * ink_term
        + weights.get("readability", 1.0) * float(readability_term)
        + weights.get("aesthetic", 1.0) * aesthetic
        + weights.get("topology", 1.0) * topology_term
        + weights.get("diversity", 0.0) * diversity_bonus
    )
    return CandidateScore(
        ink_saving=saving,
        readability_margin=float(margin),
        target_similarity=float(target),
        aesthetic=aesthetic,
        topology_penalty=topo,
        diversity_bonus=float(diversity_bonus),
        total=float(total),
    )
