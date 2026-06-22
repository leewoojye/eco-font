from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def recognize_with_tesseract(image: np.ndarray, lang: str = "eng", psm: int = 10) -> str:
    """Run local Tesseract if installed.

    This wrapper is optional because OCR engines are often environment-specific.
    It is intended for evaluation or pseudo-label filtering, not backpropagation.
    """
    if not tesseract_available():
        raise RuntimeError("tesseract binary is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "glyph.png"
        arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(image_path)
        cmd = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            lang,
            "--psm",
            str(psm),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip()
