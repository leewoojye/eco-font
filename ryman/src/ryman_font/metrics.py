from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import shutil
import subprocess
import tempfile
import unicodedata

from .render import skeleton_map


def ink_area(image: np.ndarray) -> float:
    return float(np.clip(image, 0.0, 1.0).sum())


def ink_saving(original: np.ndarray, eco: np.ndarray) -> float:
    original_ink = ink_area(original)
    if original_ink <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - ink_area(eco) / original_ink, 0.0, 1.0))


def _component_count(binary: np.ndarray) -> int:
    num, _labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return max(0, int(num) - 1)


def topology(image: np.ndarray) -> tuple[int, int]:
    binary = (image > 0.15).astype(np.uint8)
    components = _component_count(binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for item in hierarchy[0]:
            if int(item[3]) >= 0:
                holes += 1
    return components, holes


def skeleton_recall(original: np.ndarray, eco: np.ndarray) -> float:
    skel = skeleton_map(original) > 0
    if not skel.any():
        return 1.0
    eco_binary = (eco > 0.10).astype(np.uint8)
    eco_dilated = cv2.dilate(eco_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return float((eco_dilated[skel] > 0).mean())


def template_view(image: np.ndarray) -> np.ndarray:
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
    image_vec = template_view(image).reshape(-1)
    image_vec -= float(image_vec.mean())
    norm = float(np.linalg.norm(image_vec))
    if norm <= 1e-8:
        return None, None
    best_char: str | None = None
    best_score = -1.0
    for ch, template in templates.items():
        t = template_view(template).reshape(-1)
        t -= float(t.mean())
        denom = norm * float(np.linalg.norm(t))
        if denom <= 1e-8:
            continue
        score = float(np.dot(image_vec, t) / denom)
        if score > best_score:
            best_char = ch
            best_score = score
    return best_char, best_score if best_char is not None else None


def _normalize_ocr_text(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = unicodedata.normalize("NFC", text)
    normalized = "".join(ch for ch in normalized if not ch.isspace())
    return normalized or None


def _tesseract_languages() -> set[str]:
    if shutil.which("tesseract") is None:
        return set()
    try:
        proc = subprocess.run(["tesseract", "--list-langs"], check=False, capture_output=True, text=True, timeout=8)
    except Exception:
        return set()
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {line for line in lines if not line.lower().startswith("list of available")}


def _ocr_ready(lang: str) -> tuple[bool, str | None]:
    if shutil.which("tesseract") is None:
        return False, "tesseract executable not found"
    langs = _tesseract_languages()
    requested = [item for item in lang.split("+") if item]
    missing = [item for item in requested if item not in langs]
    if missing:
        return False, "missing tesseract language data: " + ",".join(missing)
    return True, None


def _prepare_tesseract_image(image: np.ndarray, scale: int = 8, pad: int = 24) -> Image.Image:
    binary = (image > 0.08).astype(np.uint8) * 255
    # Tesseract expects dark text on a light background. Our previews are white
    # glyphs on black, so invert and add generous whitespace around one glyph.
    inverted = 255 - binary
    padded = cv2.copyMakeBorder(inverted, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    if scale > 1:
        padded = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(padded, mode="L")


def recognize_by_tesseract(image: np.ndarray, expected_char: str | None = None, lang: str = "kor", psm: int = 10) -> dict:
    ready, error = _ocr_ready(lang)
    result = {
        "available": ready,
        "lang": lang,
        "psm": int(psm),
        "text": None,
        "confidence": None,
        "match": None,
        "error": error,
    }
    if not ready:
        return result

    with tempfile.TemporaryDirectory(prefix="ryman_ocr_") as tmp:
        path = f"{tmp}/glyph.png"
        _prepare_tesseract_image(image).save(path)
        cmd = ["tesseract", path, "stdout", "-l", lang, "--psm", str(int(psm)), "--oem", "1", "tsv"]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=12)
        except Exception as exc:
            result["available"] = False
            result["error"] = str(exc)
            return result
    if proc.returncode != 0:
        result["error"] = proc.stderr.strip() or f"tesseract exited with code {proc.returncode}"
        return result

    words: list[str] = []
    confidences: list[float] = []
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if lines:
        header = lines[0].split("\t")
        try:
            conf_idx = header.index("conf")
            text_idx = header.index("text")
        except ValueError:
            conf_idx = text_idx = -1
        if conf_idx >= 0 and text_idx >= 0:
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) <= max(conf_idx, text_idx):
                    continue
                text = _normalize_ocr_text(parts[text_idx])
                if not text:
                    continue
                words.append(text)
                try:
                    conf = float(parts[conf_idx])
                except ValueError:
                    conf = -1.0
                if conf >= 0:
                    confidences.append(conf)
    text = _normalize_ocr_text("".join(words))
    result["text"] = text
    result["confidence"] = float(np.mean(confidences)) if confidences else None
    if expected_char is not None:
        expected = _normalize_ocr_text(expected_char)
        result["match"] = bool(text == expected or (text is not None and expected in text))
    return result


def line_aesthetic_score(eco: np.ndarray) -> float:
    """Heuristic: prefer intentional parallel/contour lines over blobs/speckles."""
    binary = (eco > 0.15).astype(np.uint8)
    if binary.max() == 0:
        return 0.0
    edges = cv2.Canny((binary * 255).astype(np.uint8), 50, 150)
    line_density = float(edges.sum() / 255.0) / max(1.0, float(binary.sum()))
    components, _holes = topology(eco)
    component_penalty = max(0.0, (components - 4) / 20.0)
    score = np.clip(0.75 * min(line_density / 0.8, 1.0) + 0.25 * (1.0 - component_penalty), 0.0, 1.0)
    return float(score)


def evaluate(
    original: np.ndarray,
    eco: np.ndarray,
    target_saving: float,
    expected_char: str | None = None,
    templates: dict[str, np.ndarray] | None = None,
    ocr_engine: str = "tesseract",
    ocr_lang: str = "kor",
    ocr_psm: int = 10,
) -> dict:
    saving = ink_saving(original, eco)
    orig_components, orig_holes = topology(original)
    eco_components, eco_holes = topology(eco)
    use_template = expected_char is not None and ocr_engine in {"template", "both"}
    use_tesseract = expected_char is not None and ocr_engine in {"tesseract", "both"}
    pred_char, pred_score = recognize_by_templates(eco, templates or {}) if use_template else (None, None)
    tess = recognize_by_tesseract(eco, expected_char, lang=ocr_lang, psm=ocr_psm) if use_tesseract else {}
    return {
        "ink_saving": saving,
        "saving_gap": abs(saving - target_saving),
        "skeleton_recall": skeleton_recall(original, eco),
        "component_delta": abs(eco_components - orig_components),
        "hole_delta": abs(eco_holes - orig_holes),
        "aesthetic_line_score": line_aesthetic_score(eco),
        "ocr_engine": ocr_engine,
        "tesseract_ocr_available": tess.get("available"),
        "tesseract_ocr_lang": tess.get("lang"),
        "tesseract_ocr_psm": tess.get("psm"),
        "tesseract_ocr_text": tess.get("text"),
        "tesseract_ocr_confidence": tess.get("confidence"),
        "tesseract_ocr_match": tess.get("match"),
        "tesseract_ocr_error": tess.get("error"),
        "template_ocr_text": pred_char,
        "template_ocr_score": pred_score,
        "template_ocr_match": (pred_char == expected_char) if use_template else None,
    }
