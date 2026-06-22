from __future__ import annotations

import cv2
import numpy as np

from .render import skeleton_map


def ink_area(image: np.ndarray) -> float:
    return float(np.clip(image, 0.0, 1.0).sum())


def ink_saving(original: np.ndarray, eco: np.ndarray) -> float:
    orig = ink_area(original)
    if orig <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - ink_area(eco) / orig, 0.0, 1.0))


def component_count(image: np.ndarray) -> int:
    binary = (image > 0.15).astype(np.uint8)
    if binary.max() == 0:
        return 0
    num, _labels = cv2.connectedComponents(binary, connectivity=8)
    return max(0, int(num) - 1)


def skeleton_recall(original: np.ndarray, eco: np.ndarray) -> float:
    skel = skeleton_map(original) > 0
    if not skel.any():
        return 1.0
    eco_binary = (eco > 0.10).astype(np.uint8)
    eco_dilated = cv2.dilate(eco_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return float((eco_dilated[skel] > 0).mean())


def aesthetic_score(eco: np.ndarray) -> float:
    binary = (eco > 0.15).astype(np.uint8)
    if binary.max() == 0:
        return 0.0
    edges = cv2.Canny((binary * 255).astype(np.uint8), 50, 150)
    line_density = float(edges.sum() / 255.0) / max(1.0, float(binary.sum()))
    comp = component_count(eco)
    comp_penalty = max(0.0, (comp - 6) / 16.0)
    return float(np.clip(0.72 * min(line_density / 0.95, 1.0) + 0.28 * (1.0 - comp_penalty), 0.0, 1.0))


def style_novelty(original: np.ndarray, eco: np.ndarray) -> float:
    original_binary = original > 0.12
    eco_binary = eco > 0.12
    union = original_binary | eco_binary
    if not union.any():
        return 0.0
    xor_ratio = float((original_binary ^ eco_binary).sum() / union.sum())
    recall_departure = 1.0 - skeleton_recall(original, eco)
    comp_delta = min(float(abs(component_count(eco) - component_count(original))) / 6.0, 1.0)
    return float(np.clip(0.58 * xor_ratio + 0.30 * recall_departure + 0.12 * comp_delta, 0.0, 1.0))


def evaluate(original: np.ndarray, eco: np.ndarray, target_saving: float) -> dict[str, float]:
    saving = ink_saving(original, eco)
    return {
        "ink_saving": saving,
        "saving_gap": abs(saving - target_saving),
        "skeleton_recall": skeleton_recall(original, eco),
        "component_delta": float(abs(component_count(eco) - component_count(original))),
        "aesthetic_score": aesthetic_score(eco),
        "style_novelty": style_novelty(original, eco),
    }


def candidate_score(metrics: dict[str, float], ocr: dict | None = None, ocr_threshold: float = 0.70) -> float:
    if ocr is not None:
        conf = float(ocr.get("ocr_confidence", 0.0))
        if (not ocr.get("ocr_match")) or conf < ocr_threshold:
            return -1_000.0 + conf
    return (
        0.90 * float(metrics["ink_saving"])
        + 0.85 * float(metrics["skeleton_recall"])
        + 0.45 * float(metrics["aesthetic_score"])
        + 0.28 * float(metrics["style_novelty"])
        - 2.0 * float(metrics["saving_gap"])
        - 0.055 * max(0.0, float(metrics["component_delta"]) - 6.0)
    )


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k, value in row.items() if isinstance(value, (int, float))})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}
