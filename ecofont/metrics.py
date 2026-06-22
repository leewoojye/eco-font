"""Ink/readability tradeoff metrics."""

from __future__ import annotations

import cv2
import numpy as np

from .image_ops import binarize, to_float01


def ink_area(foreground: np.ndarray) -> float:
    """Total ink amount as summed foreground intensity."""
    return float(to_float01(foreground).sum())


def ink_ratio(original: np.ndarray, eco: np.ndarray) -> float:
    """Eco ink area divided by original ink area."""
    original_area = ink_area(original)
    if original_area <= 1e-6:
        return 1.0
    return float(ink_area(eco) / original_area)


def ink_saving(original: np.ndarray, eco: np.ndarray) -> float:
    """Fractional ink saving against original."""
    return float(np.clip(1.0 - ink_ratio(original, eco), 0.0, 1.0))


def ssim(original: np.ndarray, eco: np.ndarray) -> float:
    """Compute SSIM for grayscale glyph arrays without scikit-image."""
    x = 1.0 - to_float01(original)
    y = 1.0 - to_float01(eco)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    value = np.mean(numerator / np.maximum(denominator, 1e-8))
    return float(np.clip(value, 0.0, 1.0))


def connected_component_count(foreground: np.ndarray) -> int:
    """Count connected foreground components."""
    mask = binarize(foreground)
    if mask.max(initial=0) == 0:
        return 0
    count, _ = cv2.connectedComponents(mask, connectivity=8)
    return max(0, int(count) - 1)


def topology_penalty(original: np.ndarray, eco: np.ndarray) -> float:
    """Penalty for disconnected or erased glyph topology."""
    original_mask = binarize(original)
    eco_mask = binarize(eco)

    original_area = int(original_mask.sum())
    eco_area = int(eco_mask.sum())
    if original_area == 0:
        return 1.0
    if eco_area == 0:
        return 10.0

    original_components = connected_component_count(original_mask)
    eco_components = connected_component_count(eco_mask)
    component_delta = abs(eco_components - original_components) / max(1, original_components)

    lost_all = max(0.0, 0.08 - (eco_area / max(1, original_area))) * 10.0
    return float(component_delta + lost_all)


def evaluate_tradeoff(
    original: np.ndarray,
    eco: np.ndarray,
    target_saving: float | None = None,
) -> dict[str, float]:
    """Evaluate the core guide metrics for one glyph."""
    ratio = ink_ratio(original, eco)
    saving = 1.0 - ratio
    metrics = {
        "ink_ratio": float(ratio),
        "ink_saving": float(saving),
        "ssim": ssim(original, eco),
        "topology_penalty": topology_penalty(original, eco),
    }
    if target_saving is not None:
        metrics["target_saving"] = float(target_saving)
        metrics["target_error"] = abs(float(target_saving) - float(saving))
    return metrics


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average numeric metric dictionaries."""
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row if isinstance(row[key], (int, float))})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}
