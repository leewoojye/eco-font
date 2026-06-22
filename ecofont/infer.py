"""Inference helpers for trained and rule-based eco masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .font_io import filter_supported_chars
from .image_ops import distance_transform, features_for_glyph, make_contact_sheet, save_foreground_png, save_mask_png
from .metrics import average_metrics, evaluate_tradeoff
from .model import EcoMaskUNet
from .ocr_guided import OCRGuidedWeights, optimize_rule_ocr_guided
from .ocr_surrogate import OCREvaluator
from .render import render_glyph, safe_char_name
from .rules import optimize_rule
from .text_presets import characters_for_language
from .train import resolve_device


def load_model(checkpoint_path: str | Path, device: str = "auto") -> tuple[EcoMaskUNet, torch.device]:
    device_obj = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    config = checkpoint.get("model_config", {"input_channels": 4, "base_channels": 32})
    model = EcoMaskUNet(**config).to(device_obj)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device_obj


def predict_remove_mask(
    model: EcoMaskUNet,
    device: torch.device,
    foreground: np.ndarray,
    target_saving: float,
    threshold: float = 0.5,
) -> np.ndarray:
    x = features_for_glyph(foreground, target_saving)[None, :, :, :]
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

    dist = distance_transform(foreground)
    fg = foreground > 0.2
    candidate = fg & (dist >= 1.2)
    remove = np.zeros_like(foreground, dtype=np.float32)

    target_area = float(np.clip(target_saving, 0.0, 0.8)) * float(foreground.sum())
    candidate_indices = np.flatnonzero(candidate)
    if target_area <= 0.0 or candidate_indices.size == 0:
        return remove

    flat_probs = probs.reshape(-1)
    flat_ink = foreground.reshape(-1)
    eligible = candidate_indices[flat_probs[candidate_indices] >= threshold]
    if flat_ink[eligible].sum(initial=0.0) < target_area:
        eligible = candidate_indices
    if eligible.size == 0:
        return remove

    order = eligible[np.argsort(flat_probs[eligible])[::-1]]
    cumulative_ink = np.cumsum(flat_ink[order])
    keep = int(np.searchsorted(cumulative_ink, target_area, side="left")) + 1
    remove.reshape(-1)[order[:keep]] = 1.0
    return remove.astype(np.float32)


def infer_font(
    font_path: str | Path,
    output: str | Path,
    checkpoint: str | Path | None = None,
    ocr_checkpoint: str | Path | None = None,
    method: str = "model",
    language: str = "ko",
    text: str | None = None,
    target_saving: float = 0.25,
    image_size: int = 128,
    threshold: float = 0.5,
    device: str = "auto",
    max_chars: int | None = None,
    candidate_limit: int | None = None,
    ocr_weight: float = 1.0,
    ocr_target_weight: float = 3.0,
    ocr_ink_weight: float = 0.2,
    outline_reward_weight: float = 0.35,
) -> dict:
    """Run model or rule-based inference for glyph previews."""
    output_path = Path(output)
    glyph_dir = output_path / "glyphs"
    glyph_dir.mkdir(parents=True, exist_ok=True)

    chars = characters_for_language(language, text)
    if max_chars is not None:
        chars = chars[:max_chars]
    supported, missing = filter_supported_chars(font_path, chars)
    if not supported:
        raise ValueError(f"No requested characters are supported by font: {font_path}")

    model = None
    device_obj = None
    ocr_evaluator = None
    if method == "model":
        if checkpoint is None:
            raise ValueError("--checkpoint is required when --method model")
        model, device_obj = load_model(checkpoint, device=device)
    elif method == "ocr-rules":
        if ocr_checkpoint is None:
            raise ValueError("--ocr-checkpoint is required when --method ocr-rules")
        ocr_evaluator = OCREvaluator(ocr_checkpoint, device=device)
    elif method != "rules":
        raise ValueError("--method must be 'model', 'rules', or 'ocr-rules'")

    rows: list[dict] = []
    sheet_rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for ch in tqdm(supported, desc="infer", unit="glyph"):
        rendered = render_glyph(font_path, ch, image_size=image_size)
        if method == "rules":
            result = optimize_rule(rendered.foreground, target_saving, candidate_limit=candidate_limit)
            remove = result.remove_mask
            eco = result.eco
            rule_params = result.params_dict()
            loss = result.loss
        elif method == "ocr-rules":
            assert ocr_evaluator is not None
            weights = OCRGuidedWeights(
                ocr_weight=ocr_weight,
                target_weight=ocr_target_weight,
                ink_weight=ocr_ink_weight,
                outline_reward_weight=outline_reward_weight,
            )
            result = optimize_rule_ocr_guided(
                rendered.foreground,
                ch,
                target_saving,
                evaluator=ocr_evaluator,
                weights=weights,
                candidate_limit=candidate_limit,
            )
            remove = result.remove_mask
            eco = result.eco
            rule_params = result.params_dict()
            loss = result.loss
        else:
            assert model is not None and device_obj is not None
            remove = predict_remove_mask(model, device_obj, rendered.foreground, target_saving, threshold=threshold)
            eco = rendered.foreground * (1.0 - remove)
            rule_params = None
            loss = None

        metrics = evaluate_tradeoff(rendered.foreground, eco, target_saving=target_saving)
        if method == "ocr-rules":
            metrics.update(result.metrics)
        name = safe_char_name(ch)
        save_foreground_png(glyph_dir / f"{name}_original.png", rendered.foreground)
        save_foreground_png(glyph_dir / f"{name}_eco.png", eco)
        save_mask_png(glyph_dir / f"{name}_mask.png", remove)

        rows.append(
            {
                "char": ch,
                "codepoint": rendered.codepoint,
                "metrics": metrics,
                "rule_params": rule_params,
                "loss": loss,
                "files": {
                    "original": f"glyphs/{name}_original.png",
                    "eco": f"glyphs/{name}_eco.png",
                    "mask": f"glyphs/{name}_mask.png",
                },
            }
        )
        if len(sheet_rows) < 24:
            sheet_rows.append((f"{ch} {rendered.codepoint}", rendered.foreground, eco, remove))

    summary = {
        "font": str(font_path),
        "method": method,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "ocr_checkpoint": str(ocr_checkpoint) if ocr_checkpoint is not None else None,
        "language": language,
        "target_saving": float(target_saving),
        "image_size": image_size,
        "supported_count": len(supported),
        "missing_count": len(missing),
        "missing": [{"char": ch, "codepoint": f"U+{ord(ch):04X}"} for ch in missing],
        "average_metrics": average_metrics([row["metrics"] for row in rows]),
        "glyphs": rows,
    }
    (output_path / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_contact_sheet(sheet_rows, output_path / "preview.png")
    return summary
