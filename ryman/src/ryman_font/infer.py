from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import ensure_dir
from .metrics import evaluate
from .model import build_model
from .pseudo import input_channels, project_canvas_to_ink_budget, project_to_ink_budget
from .render import has_visible_glyph, render_glyph, save_gray
from .vectorize import export_ttf


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt.get("model_config") or ckpt.get("config", {}).get("model", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, glyph: np.ndarray, target_saving: float, device: torch.device, force_budget: bool = True, target_style: str = "contour", char: str | None = None) -> np.ndarray:
    x = input_channels(glyph, target_saving, style=target_style, char=char)
    tensor = torch.from_numpy(x[None]).to(device=device, dtype=torch.float32)
    prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy().astype(np.float32)
    if force_budget:
        if target_style == "canonical":
            return project_canvas_to_ink_budget(prob, glyph, target_saving)
        return project_to_ink_budget(prob, glyph, target_saving)
    return np.clip(prob, 0.0, 1.0)


def run_inference(
    checkpoint: str | Path,
    font: str | Path,
    chars: list[str],
    out_dir: str | Path,
    target_saving: float,
    image_size: int = 96,
    font_size: int = 76,
    device_name: str = "auto",
    export_font: str | Path | None = None,
    force_budget: bool = True,
    target_style: str | None = None,
    ocr_engine: str = "tesseract",
    ocr_lang: str = "kor",
    ocr_psm: int = 10,
) -> Path:
    device = _device(device_name)
    out = ensure_dir(out_dir)
    original_dir = ensure_dir(out / "original")
    eco_dir = ensure_dir(out / "eco")
    manifest_path = out / "inference_manifest.jsonl"
    model, ckpt = load_checkpoint(checkpoint, device)
    if target_style is None:
        target_style = str(ckpt.get("config", {}).get("data", {}).get("target_style", "contour"))
    templates: dict[str, np.ndarray] = {}
    for ch in chars:
        rendered = render_glyph(font, ch, image_size=image_size, font_size=font_size)
        if has_visible_glyph(rendered.image):
            templates[ch] = rendered.image
    bitmaps: dict[str, np.ndarray] = {}
    with manifest_path.open("w", encoding="utf-8") as f:
        for ch in tqdm(chars, desc="infer"):
            rendered = render_glyph(font, ch, image_size=image_size, font_size=font_size)
            if not has_visible_glyph(rendered.image):
                continue
            eco = predict(model, rendered.image, target_saving, device, force_budget=force_budget, target_style=target_style, char=ch)
            char_id = f"u{ord(ch):04x}"
            save_gray(original_dir / f"{char_id}.png", rendered.image)
            save_gray(eco_dir / f"{char_id}.png", eco)
            metrics = evaluate(
                rendered.image,
                eco,
                target_saving,
                expected_char=ch,
                templates=templates,
                ocr_engine=ocr_engine,
                ocr_lang=ocr_lang,
                ocr_psm=ocr_psm,
            )
            bitmaps[ch] = eco
            record = {
                "font": str(font),
                "char": ch,
                "char_id": char_id,
                "target_saving": float(target_saving),
                "target_style": target_style,
                "original": str(Path("original") / f"{char_id}.png"),
                "eco": str(Path("eco") / f"{char_id}.png"),
                "metrics": metrics,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if export_font:
        export_ttf(font, bitmaps, export_font)
    return manifest_path
