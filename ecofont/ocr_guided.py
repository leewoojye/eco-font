"""OCR-guided rule optimization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import numpy as np

from .image_ops import binarize, distance_transform
from .metrics import evaluate_tradeoff
from .ocr_surrogate import OCREvaluator
from .rules import RuleResult, apply_rule, candidate_params


@dataclass(frozen=True)
class OCRGuidedWeights:
    ocr_weight: float = 1.0
    target_weight: float = 3.0
    ink_weight: float = 0.2
    outline_reward_weight: float = 0.35


def outline_change_ratio(original: np.ndarray, remove_mask: np.ndarray) -> float:
    """How much of the removed ink touches the glyph outline."""
    fg = binarize(original)
    dist = distance_transform(original)
    edge = (fg > 0) & (dist <= 2.0)
    removed = remove_mask > 0.2
    total_removed = int(removed.sum())
    if total_removed == 0:
        return 0.0
    return float((removed & edge).sum() / total_removed)


def ocr_guided_loss(metrics: dict[str, float], weights: OCRGuidedWeights) -> float:
    """Loss that uses OCR confidence instead of SSIM/topology preservation."""
    return float(
        weights.ocr_weight * metrics["ocr_loss"]
        + weights.target_weight * metrics.get("target_error", 0.0)
        + weights.ink_weight * metrics["ink_ratio"]
        - weights.outline_reward_weight * metrics.get("outline_change_ratio", 0.0)
    )


def optimize_rule_ocr_guided(
    foreground: np.ndarray,
    char: str,
    target_saving: float,
    evaluator: OCREvaluator,
    weights: OCRGuidedWeights | None = None,
    candidate_limit: int | None = None,
) -> RuleResult:
    """Search outline-changing candidates with OCR confidence as readability loss."""
    weights = weights or OCRGuidedWeights()
    candidates = candidate_params(target_saving, include_outline=True)
    iterator = candidates if candidate_limit is None else list(islice(candidates, candidate_limit))

    prepared = []
    eco_images = []
    for params in iterator:
        remove, eco = apply_rule(foreground, params)
        prepared.append((params, remove, eco))
        eco_images.append(eco)

    if not prepared:
        raise RuntimeError("No OCR-guided rule candidates were generated")

    ocr_rows = evaluator.score_batch(eco_images, [char] * len(eco_images))
    best: RuleResult | None = None
    for (params, remove, eco), ocr_row in zip(prepared, ocr_rows, strict=True):
        metrics = evaluate_tradeoff(foreground, eco, target_saving=target_saving)
        metrics.update(
            {
                "ocr_confidence": float(ocr_row["ocr_confidence"]),
                "ocr_pred_confidence": float(ocr_row["ocr_pred_confidence"]),
                "ocr_loss": float(ocr_row["ocr_loss"]),
                "outline_change_ratio": outline_change_ratio(foreground, remove),
            }
        )
        metrics["ocr_correct"] = 1.0 if bool(ocr_row["ocr_correct"]) else 0.0
        loss = ocr_guided_loss(metrics, weights)
        result = RuleResult(params=params, remove_mask=remove, eco=eco, metrics=metrics, loss=loss)
        if best is None or result.loss < best.loss:
            best = result

    if best is None:
        raise RuntimeError("OCR-guided optimization failed")
    return best
