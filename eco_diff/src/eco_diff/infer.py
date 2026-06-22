from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import ensure_dir
from .font_render import has_visible_glyph, model_input_channels, render_glyph, save_gray
from .metrics import apply_cut_mask, evaluate_candidate
from .models import build_model
from .vectorize import export_ttf_from_bitmaps


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("model", {})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def force_target_saving_mask(probs: np.ndarray, glyph: np.ndarray, target_saving: float) -> np.ndarray:
    """Convert probabilities to a binary mask with approximately target ink saving.

    The model is intentionally conservative after short smoke training. This
    helper is useful for preview and ablation samples because it preserves the
    model's pixel ranking while making the cut area match the requested saving.
    """
    inside = glyph > 0.05
    if not inside.any():
        return np.zeros_like(probs, dtype=np.float32)
    target_ink = float(np.clip(target_saving, 0.0, 0.9)) * float(glyph[inside].sum())
    if target_ink <= 0:
        return np.zeros_like(probs, dtype=np.float32)

    ys, xs = np.where(inside)
    scores = probs[ys, xs]
    weights = glyph[ys, xs]
    order = np.argsort(-scores)
    cumulative = np.cumsum(weights[order])
    cutoff = int(np.searchsorted(cumulative, target_ink, side="left")) + 1
    cutoff = max(1, min(cutoff, len(order)))
    selected = order[:cutoff]
    mask = np.zeros_like(probs, dtype=np.float32)
    mask[ys[selected], xs[selected]] = 1.0
    return mask


def predict_mask(
    model: torch.nn.Module,
    glyph: np.ndarray,
    target_saving: float,
    device: torch.device,
    threshold: float | None = None,
    force_saving: bool = False,
) -> np.ndarray:
    x = model_input_channels(glyph, target_saving)
    with torch.no_grad():
        tensor = torch.from_numpy(x[None]).to(device=device, dtype=torch.float32)
        probs = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
    probs *= (glyph > 0.05).astype(np.float32)
    if force_saving:
        return force_target_saving_mask(probs, glyph, target_saving)
    if threshold is not None:
        probs = (probs >= threshold).astype(np.float32)
    return probs.astype(np.float32)


def run_inference(
    checkpoint: str | Path,
    font: str | Path,
    chars: list[str],
    out_dir: str | Path,
    target_saving: float,
    image_size: int = 96,
    font_size: int = 76,
    threshold: float | None = None,
    force_saving: bool = False,
    device_name: str = "auto",
    export_ttf: str | Path | None = None,
) -> Path:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    out = ensure_dir(out_dir)
    original_dir = ensure_dir(out / "original")
    mask_dir = ensure_dir(out / "mask")
    eco_dir = ensure_dir(out / "eco")
    manifest_path = out / "inference_manifest.jsonl"
    model = load_checkpoint(checkpoint, device)
    bitmaps_for_ttf: dict[str, np.ndarray] = {}

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for ch in tqdm(chars, desc="infer"):
            rendered = render_glyph(font, ch, image_size=image_size, font_size=font_size)
            if not has_visible_glyph(rendered.image):
                continue
            mask = predict_mask(
                model,
                rendered.image,
                target_saving,
                device,
                threshold=threshold,
                force_saving=force_saving,
            )
            eco = apply_cut_mask(rendered.image, mask)
            char_id = f"u{ord(ch):04x}"
            save_gray(original_dir / f"{char_id}.png", rendered.image)
            save_gray(mask_dir / f"{char_id}.png", mask)
            save_gray(eco_dir / f"{char_id}.png", eco)
            metrics = evaluate_candidate(rendered.image, mask).to_dict()
            bitmaps_for_ttf[ch] = eco
            record = {
                "font_path": str(font),
                "char": ch,
                "char_id": char_id,
                "target_saving": float(target_saving),
                "original": str(Path("original") / f"{char_id}.png"),
                "mask": str(Path("mask") / f"{char_id}.png"),
                "eco": str(Path("eco") / f"{char_id}.png"),
                "metrics": metrics,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    if export_ttf:
        export_ttf_from_bitmaps(font, bitmaps_for_ttf, export_ttf)
    return manifest_path
