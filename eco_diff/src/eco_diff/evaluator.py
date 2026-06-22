from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .font_render import skeleton_map
from .metrics import ink_saving, ssim_score, topology_stats
from .ocr import recognize_with_tesseract, tesseract_available


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    ink_saving: float
    saving_gap: float
    ssim: float
    skeleton_recall: float
    component_delta: int
    hole_delta: int
    ocr_text: str | None
    ocr_match: bool | None
    template_ocr_text: str | None
    template_ocr_score: float | None
    template_ocr_match: bool | None

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            "score": self.score,
            "ink_saving": self.ink_saving,
            "saving_gap": self.saving_gap,
            "ssim": self.ssim,
            "skeleton_recall": self.skeleton_recall,
            "component_delta": self.component_delta,
            "hole_delta": self.hole_delta,
            "ocr_text": self.ocr_text,
            "ocr_match": self.ocr_match,
            "template_ocr_text": self.template_ocr_text,
            "template_ocr_score": self.template_ocr_score,
            "template_ocr_match": self.template_ocr_match,
        }


def _skeleton_recall(original: np.ndarray, eco: np.ndarray) -> float:
    original_skel = skeleton_map(original) > 0
    if not original_skel.any():
        return 1.0
    eco_binary = (eco > 0.12).astype(np.uint8)
    # Allow a small local shift because the diffusion branch may distort outlines.
    eco_dilated = cv2.dilate(eco_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return float((eco_dilated[original_skel] > 0).mean())


def _template_view(image: np.ndarray) -> np.ndarray:
    binary = (image > 0.10).astype(np.uint8)
    binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
    max_value = float(blurred.max())
    if max_value > 0:
        blurred /= max_value
    return blurred.astype(np.float32)


def recognize_by_templates(image: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str | None, float | None]:
    if not templates:
        return None, None
    image_vec = _template_view(image).reshape(-1)
    image_vec = image_vec - float(image_vec.mean())
    image_norm = float(np.linalg.norm(image_vec))
    if image_norm <= 1e-8:
        return None, None
    best_char: str | None = None
    best_score = -1.0
    for ch, template in templates.items():
        template_vec = _template_view(template).reshape(-1)
        template_vec = template_vec - float(template_vec.mean())
        denom = image_norm * float(np.linalg.norm(template_vec))
        if denom <= 1e-8:
            continue
        score = float(np.dot(image_vec, template_vec) / denom)
        if score > best_score:
            best_char = ch
            best_score = score
    return best_char, best_score if best_char is not None else None


def evaluate_eco_candidate(
    original: np.ndarray,
    eco: np.ndarray,
    target_saving: float,
    expected_char: str | None = None,
    ocr_lang: str | None = None,
    template_ocr: dict[str, np.ndarray] | None = None,
) -> EvaluationResult:
    eco = np.clip(eco, 0.0, 1.0).astype(np.float32)
    saving = ink_saving(original, eco)
    saving_gap = abs(saving - target_saving)
    ssim = ssim_score(original, eco)
    recall = _skeleton_recall(original, eco)
    original_components, original_holes = topology_stats(original)
    eco_components, eco_holes = topology_stats(eco)
    component_delta = abs(eco_components - original_components)
    hole_delta = abs(eco_holes - original_holes)

    ocr_text: str | None = None
    ocr_match: bool | None = None
    ocr_penalty = 0.0
    if ocr_lang and expected_char and tesseract_available():
        try:
            ocr_text = recognize_with_tesseract(eco, lang=ocr_lang, psm=10)
            ocr_match = expected_char in ocr_text
            ocr_penalty = 0.0 if ocr_match else 1.5
        except Exception as exc:  # OCR is an optional evaluator.
            ocr_text = f"OCR_ERROR:{exc}"
            ocr_match = None
            ocr_penalty = 0.35

    template_text: str | None = None
    template_score: float | None = None
    template_match: bool | None = None
    template_penalty = 0.0
    if template_ocr and expected_char:
        template_text, template_score = recognize_by_templates(eco, template_ocr)
        template_match = template_text == expected_char
        if template_match:
            template_penalty = 0.2 * max(0.0, 0.65 - float(template_score or 0.0))
        else:
            template_penalty = 1.0

    # Lower is better. SSIM is reported but excluded from selection because
    # outline distortion is acceptable for the aggressive eco glyph objective.
    score = (
        3.0 * saving_gap
        + 1.2 * (1.0 - recall)
        + 0.35 * component_delta
        + 0.08 * hole_delta
        + ocr_penalty
        + template_penalty
    )
    return EvaluationResult(
        score=float(score),
        ink_saving=float(saving),
        saving_gap=float(saving_gap),
        ssim=float(ssim),
        skeleton_recall=float(recall),
        component_delta=component_delta,
        hole_delta=hole_delta,
        ocr_text=ocr_text,
        ocr_match=ocr_match,
        template_ocr_text=template_text,
        template_ocr_score=template_score,
        template_ocr_match=template_match,
    )


def enforce_ink_budget(candidate: np.ndarray, original: np.ndarray, target_saving: float, allow_outline_shift: int = 2) -> np.ndarray:
    """Project a generated glyph to the requested foreground ink budget.

    Diffusion gives a ranked foreground probability field. This keeps the best
    candidate pixels up to the allowed ink area, while permitting a small outline
    shift around the original glyph.
    """
    candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)
    original_binary = (original > 0.05).astype(np.uint8)
    if allow_outline_shift > 0:
        kernel_size = allow_outline_shift * 2 + 1
        allowed = cv2.dilate(
            original_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        ).astype(bool)
    else:
        allowed = original_binary.astype(bool)
    if not allowed.any():
        return np.zeros_like(candidate, dtype=np.float32)

    original_ink = float(np.clip(original, 0.0, 1.0).sum())
    budget = max(1.0, original_ink * (1.0 - float(np.clip(target_saving, 0.0, 0.95))))
    candidate_smooth = cv2.GaussianBlur(candidate, (5, 5), 0)
    original_smooth = cv2.GaussianBlur(original, (5, 5), 0)
    skel = (skeleton_map(original) > 0).astype(np.uint8)
    if skel.max() > 0:
        distance_to_skel = cv2.distanceTransform((1 - skel).astype(np.uint8), cv2.DIST_L2, 5)
        skeleton_proximity = np.exp(-distance_to_skel / 2.25).astype(np.float32)
    else:
        skeleton_proximity = original_smooth

    # Rank by diffusion preference, but bias toward continuous, readable stroke
    # centers. This prevents high-saving candidates from collapsing into speckles.
    score_map = 0.45 * candidate_smooth + 0.35 * skeleton_proximity + 0.20 * original_smooth
    score_map = cv2.GaussianBlur(score_map.astype(np.float32), (3, 3), 0)
    ys, xs = np.where(allowed)
    scores = score_map[ys, xs]
    order = np.argsort(-scores)
    cutoff = int(round(budget))
    cutoff = max(1, min(cutoff, len(order)))
    selected = order[:cutoff]
    projected = np.zeros_like(candidate, dtype=np.float32)
    projected[ys[selected], xs[selected]] = 1.0
    return projected
