from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidateMetrics:
    ink_original: float
    ink_eco: float
    ink_saving: float
    ssim: float
    connected_component_delta: int
    hole_delta: int
    topology_penalty: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ink_original": self.ink_original,
            "ink_eco": self.ink_eco,
            "ink_saving": self.ink_saving,
            "ssim": self.ssim,
            "connected_component_delta": self.connected_component_delta,
            "hole_delta": self.hole_delta,
            "topology_penalty": self.topology_penalty,
        }


def ink_area(image: np.ndarray) -> float:
    return float(np.clip(image, 0.0, 1.0).sum())


def apply_cut_mask(glyph: np.ndarray, cut_mask: np.ndarray) -> np.ndarray:
    mask = np.clip(cut_mask, 0.0, 1.0)
    return np.clip(glyph * (1.0 - mask), 0.0, 1.0).astype(np.float32)


def ink_saving(original: np.ndarray, eco: np.ndarray) -> float:
    original_ink = ink_area(original)
    if original_ink <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - ink_area(eco) / original_ink, 0.0, 1.0))


def ssim_score(original: np.ndarray, eco: np.ndarray) -> float:
    if original.shape != eco.shape:
        raise ValueError("SSIM inputs must have the same shape")
    if original.max() == 0 and eco.max() == 0:
        return 1.0
    img1 = original.astype(np.float32)
    img2 = eco.astype(np.float32)
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = (11, 11)
    sigma = 1.5
    mu1 = cv2.GaussianBlur(img1, kernel, sigma)
    mu2 = cv2.GaussianBlur(img2, kernel, sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, kernel, sigma) - mu1_mu2
    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-8)
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))


def _component_count(binary: np.ndarray) -> int:
    num_labels, _labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return max(0, int(num_labels) - 1)


def _hole_count(binary: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    hierarchy = hierarchy[0]
    holes = 0
    for item in hierarchy:
        parent = int(item[3])
        if parent >= 0:
            holes += 1
    return holes


def topology_stats(image: np.ndarray) -> tuple[int, int]:
    binary = (image > 0.18).astype(np.uint8)
    return _component_count(binary), _hole_count(binary)


def evaluate_candidate(original: np.ndarray, cut_mask: np.ndarray) -> CandidateMetrics:
    eco = apply_cut_mask(original, cut_mask)
    original_components, original_holes = topology_stats(original)
    eco_components, eco_holes = topology_stats(eco)
    component_delta = abs(eco_components - original_components)
    hole_delta = abs(eco_holes - original_holes)
    topo_penalty = float(component_delta + 0.35 * hole_delta)
    original_ink = ink_area(original)
    eco_ink = ink_area(eco)
    saving = 0.0 if original_ink <= 1e-6 else float(np.clip(1.0 - eco_ink / original_ink, 0.0, 1.0))
    return CandidateMetrics(
        ink_original=original_ink,
        ink_eco=eco_ink,
        ink_saving=saving,
        ssim=ssim_score(original, eco),
        connected_component_delta=component_delta,
        hole_delta=hole_delta,
        topology_penalty=topo_penalty,
    )


def candidate_score(
    metrics: CandidateMetrics,
    target_saving: float,
    ssim_weight: float = 2.5,
    saving_weight: float = 3.0,
    topology_weight: float = 1.25,
) -> float:
    saving_gap = abs(metrics.ink_saving - target_saving)
    return float(
        ssim_weight * (1.0 - metrics.ssim)
        + saving_weight * saving_gap
        + topology_weight * metrics.topology_penalty
    )
