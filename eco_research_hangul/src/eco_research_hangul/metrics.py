from __future__ import annotations

import shutil
import subprocess
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


def ink_area(image: np.ndarray) -> float:
    return float(np.clip(image, 0.0, 1.0).sum())


def ink_saving(source: np.ndarray, eco: np.ndarray) -> float:
    denom = ink_area(source)
    if denom <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - ink_area(eco) / denom, -1.0, 1.0))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.clip(a, 0.0, 1.0) - np.clip(b, 0.0, 1.0)) ** 2))


def binary_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.2) -> float:
    aa = np.asarray(a > threshold, dtype=bool)
    bb = np.asarray(b > threshold, dtype=bool)
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum() / union)


def normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not ch.isspace())
    return text or None


@lru_cache(maxsize=1)
def tesseract_languages() -> set[str]:
    if shutil.which("tesseract") is None:
        return set()
    proc = subprocess.run(["tesseract", "--list-langs"], check=False, capture_output=True, text=True, timeout=8)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {line for line in lines if not line.lower().startswith("list of available")}


def prepare_ocr_image(image: np.ndarray, scale: int = 8, pad: int = 24) -> Image.Image:
    binary = (np.clip(image, 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    inverted = 255 - binary
    padded = cv2.copyMakeBorder(inverted, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    if scale > 1:
        padded = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(padded, mode="L")


def recognize_tesseract(image: np.ndarray, expected_char: str | None = None, lang: str = "kor", psm: int = 10) -> dict:
    result = {
        "available": False,
        "lang": lang,
        "psm": int(psm),
        "text": None,
        "confidence": None,
        "match": None,
        "exact_match": None,
        "error": None,
    }
    if shutil.which("tesseract") is None:
        result["error"] = "tesseract executable not found"
        return result
    missing = [item for item in lang.split("+") if item and item not in tesseract_languages()]
    if missing:
        result["error"] = "missing tesseract language data: " + ",".join(missing)
        return result
    result["available"] = True
    with tempfile.TemporaryDirectory(prefix="eco_research_ocr_") as tmp:
        path = Path(tmp) / "glyph.png"
        prepare_ocr_image(image).save(path)
        cmd = ["tesseract", str(path), "stdout", "-l", lang, "--psm", str(int(psm)), "--oem", "1", "tsv"]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=12)
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
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) <= max(conf_idx, text_idx):
                continue
            text = normalize_text(parts[text_idx])
            if not text:
                continue
            words.append(text)
            try:
                conf = float(parts[conf_idx])
            except ValueError:
                conf = -1.0
            if conf >= 0:
                confidences.append(conf)
    text = normalize_text("".join(words))
    result["text"] = text
    result["confidence"] = float(np.mean(confidences)) if confidences else None
    if expected_char is not None:
        expected = normalize_text(expected_char)
        result["exact_match"] = bool(text == expected)
        result["match"] = bool(text == expected or (text is not None and expected in text))
    return result


def recognize_tesseract_multi(
    image: np.ndarray,
    expected_char: str | None = None,
    lang: str = "kor",
    psms: Sequence[int] = (10,),
) -> dict:
    results = [recognize_tesseract(image, expected_char=expected_char, lang=lang, psm=int(psm)) for psm in psms]
    if not results:
        return recognize_tesseract(image, expected_char=expected_char, lang=lang, psm=10)

    def rank(result: dict) -> tuple[float, float, float]:
        exact = 1.0 if result.get("exact_match") is True else 0.0
        match = 1.0 if result.get("match") is True else 0.0
        confidence = float(result.get("confidence") or 0.0)
        return (exact, match, confidence)

    best = max(results, key=rank)
    best = dict(best)
    best["psm_candidates"] = [
        {
            "psm": int(result.get("psm")),
            "text": result.get("text"),
            "confidence": result.get("confidence"),
            "match": result.get("match"),
            "exact_match": result.get("exact_match"),
            "error": result.get("error"),
        }
        for result in results
    ]
    return best


def evaluate_sample(
    source: np.ndarray,
    generated: np.ndarray,
    target: np.ndarray | None,
    expected_char: str,
    ocr_lang: str = "kor",
    ocr_psm: int | Sequence[int] = 10,
) -> dict:
    if isinstance(ocr_psm, Sequence) and not isinstance(ocr_psm, (str, bytes)):
        ocr = recognize_tesseract_multi(generated, expected_char=expected_char, lang=ocr_lang, psms=ocr_psm)
    else:
        ocr = recognize_tesseract(generated, expected_char=expected_char, lang=ocr_lang, psm=int(ocr_psm))
    metrics = {
        "ink_saving": ink_saving(source, generated),
        "source_ink": ink_area(source),
        "generated_ink": ink_area(generated),
        "tesseract_available": ocr["available"],
        "tesseract_text": ocr["text"],
        "tesseract_confidence": ocr["confidence"],
        "tesseract_match": ocr["match"],
        "tesseract_exact_match": ocr["exact_match"],
        "tesseract_error": ocr["error"],
    }
    if target is not None:
        metrics["target_mse"] = mse(generated, target)
        metrics["target_iou"] = binary_iou(generated, target)
        metrics["target_ink_saving"] = ink_saving(source, target)
    return metrics
