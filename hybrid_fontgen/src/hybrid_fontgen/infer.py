from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .metrics import evaluate, mean_metrics
from .model import HybridEcoNet
from .ocr import OCREvaluator
from .priors import EXTRA_INFERENCE_STYLES, STYLES, input_channels, make_target, project_to_budget
from .render import has_visible_glyph, render_glyph, save_gray, supported_chars, unique_chars
from .report import candidate_sheet, contact_sheet
from .train import device_from_name
from .vectorize import export_ttf as export_ttf_file


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[HybridEcoNet, list[str]]:
    data = torch.load(checkpoint, map_location=device)
    styles = list(data.get("styles", STYLES))
    config = data.get("model_config", {"input_channels": 8, "base_channels": 24, "num_styles": len(styles)})
    model = HybridEcoNet(**config).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, styles


@torch.no_grad()
def predict(model: HybridEcoNet, glyph: np.ndarray, style: str, style_id: int, target_saving: float, device: torch.device) -> np.ndarray:
    x = input_channels(glyph, style, target_saving)
    tensor = torch.from_numpy(x[None]).to(device)
    style_tensor = torch.tensor([style_id], dtype=torch.long, device=device)
    prob = torch.sigmoid(model(tensor, style_tensor))[0, 0].cpu().numpy().astype(np.float32)
    prob *= (glyph > 0.05).astype(np.float32)
    return project_to_budget(prob, glyph, target_saving)


def _candidate_score(metrics: dict, ocr: dict | None, ocr_threshold: float, style: str, void_style_weight: float) -> float:
    if ocr is not None:
        ocr_conf = float(ocr.get("ocr_confidence", 0.0))
        if (not ocr.get("ocr_match")) or ocr_conf < ocr_threshold:
            return -1_000.0 + ocr_conf
    void_bonus = void_style_weight if "void" in style else 0.0
    return (
        1.18 * float(metrics["style_novelty"])
        + 0.82 * float(metrics["aesthetic_score"])
        + 0.55 * float(metrics["ink_saving"])
        + 0.64 * float(metrics.get("void_score", 0.0))
        + void_bonus
        - 2.0 * float(metrics["saving_gap"])
        - 0.055 * max(0.0, float(metrics["component_delta"]) - 6.0)
        - 0.15 * max(0.0, 0.50 - float(metrics["skeleton_recall"]))
    )


def _void_score(original: np.ndarray, eco: np.ndarray) -> float:
    original_binary = original > 0.12
    eco_binary = eco > 0.12
    removed = original_binary & (~eco_binary)
    if not removed.any():
        return 0.0
    # Internal removals are removals that do not touch the immediate outside.
    dist = cv2.distanceTransform(original_binary.astype(np.uint8), cv2.DIST_L2, 5)
    interior_removed = removed & (dist >= 2.0)
    return float(interior_removed.sum() / max(1, removed.sum()))


def run_inference(
    checkpoint: str | Path,
    font: str | Path,
    chars: str,
    out_dir: str | Path,
    style: str = "auto",
    target_saving: float = 0.60,
    ocr_checkpoint: str | Path | None = None,
    image_size: int = 96,
    device_name: str = "auto",
    export_ttf: str | Path | None = None,
    ocr_threshold: float = 0.72,
    void_style_weight: float = 0.90,
    save_candidates: bool = False,
) -> dict:
    device = device_from_name(device_name)
    model, styles = load_model(checkpoint, device)
    valid_styles = styles + EXTRA_INFERENCE_STYLES
    if style != "auto" and style not in valid_styles:
        raise ValueError(f"Unknown style: {style}. Valid: auto,{','.join(valid_styles)}")
    evaluator = OCREvaluator(ocr_checkpoint, device=device_name) if ocr_checkpoint else None

    requested, missing = supported_chars(font, unique_chars(chars))
    out = Path(out_dir)
    original_dir = out / "original"
    eco_dir = out / "eco"
    candidate_dir = out / "candidates"
    original_dir.mkdir(parents=True, exist_ok=True)
    eco_dir.mkdir(parents=True, exist_ok=True)
    if save_candidates:
        candidate_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    candidate_rows = []
    candidate_sheet_rows = []
    sheet_rows = []
    bitmaps: dict[str, np.ndarray] = {}
    for ch in tqdm(requested, desc="infer", unit="glyph"):
        rendered = render_glyph(font, ch, image_size=image_size)
        if not has_visible_glyph(rendered.image):
            continue
        candidate_styles = valid_styles if style == "auto" else [style]
        candidates = []
        images = []
        image_chars = []
        for st in candidate_styles:
            if st in styles:
                eco = predict(model, rendered.image, st, styles.index(st), target_saving, device)
                source = "model"
            else:
                eco, _score = make_target(rendered.image, st, target_saving)
                source = "prior"
            metrics = evaluate(rendered.image, eco, target_saving)
            metrics["void_score"] = _void_score(rendered.image, eco)
            candidates.append({"style": st, "eco": eco, "metrics": metrics, "source": source})
            images.append(eco)
            image_chars.append(ch)
        ocr_rows = evaluator.score(images, image_chars) if evaluator else [None] * len(candidates)
        best_i = max(
            range(len(candidates)),
            key=lambda i: _candidate_score(
                candidates[i]["metrics"],
                ocr_rows[i],
                ocr_threshold,
                candidates[i]["style"],
                void_style_weight,
            ),
        )
        if _candidate_score(candidates[best_i]["metrics"], ocr_rows[best_i], ocr_threshold, candidates[best_i]["style"], void_style_weight) < -999.0:
            best_i = max(
                range(len(candidates)),
                key=lambda i: (
                    float(ocr_rows[i].get("ocr_confidence", 0.0)) if ocr_rows[i] else 0.0,
                    float(candidates[i]["metrics"]["style_novelty"]),
                ),
            )
        chosen = candidates[best_i]
        ocr = ocr_rows[best_i]
        char_id = f"u{ord(ch):04x}"
        if save_candidates:
            glyph_candidate_dir = candidate_dir / char_id
            glyph_candidate_dir.mkdir(parents=True, exist_ok=True)
            sheet_candidates = []
            for i, candidate in enumerate(candidates):
                candidate_ocr = ocr_rows[i]
                candidate_metrics = dict(candidate["metrics"])
                if candidate_ocr:
                    candidate_metrics.update(candidate_ocr)
                selected = i == best_i
                name = f"{i:02d}_{candidate['style']}_{candidate['source']}{'_selected' if selected else ''}.png"
                save_gray(glyph_candidate_dir / name, candidate["eco"])
                candidate_rows.append(
                    {
                        "char": ch,
                        "char_id": char_id,
                        "style": candidate["style"],
                        "candidate_source": candidate["source"],
                        "selected": selected,
                        "image": str(Path("candidates") / char_id / name),
                        "metrics": candidate_metrics,
                    }
                )
                label = (
                    f"{'*' if selected else ' '} {candidate['style']}\n"
                    f"s{candidate_metrics['ink_saving']:.2f} o{candidate_metrics.get('ocr_confidence', 0.0):.2f}"
                )
                sheet_candidates.append({"label": label, "image": candidate["eco"], "selected": selected})
            candidate_sheet_rows.append({"label": char_id, "original": rendered.image, "candidates": sheet_candidates})
        save_gray(original_dir / f"{char_id}.png", rendered.image)
        save_gray(eco_dir / f"{char_id}.png", chosen["eco"])
        bitmaps[ch] = chosen["eco"]
        metrics = dict(chosen["metrics"])
        if ocr:
            metrics.update(ocr)
        rows.append(
            {
                "char": ch,
                "char_id": char_id,
                "chosen_style": chosen["style"],
                "candidate_source": chosen["source"],
                "target_saving": float(target_saving),
                "original": str(Path("original") / f"{char_id}.png"),
                "eco": str(Path("eco") / f"{char_id}.png"),
                "metrics": metrics,
            }
        )
        if len(sheet_rows) < 24:
            sheet_rows.append((f"{char_id} {chosen['style']}", rendered.image, chosen["eco"]))

    if export_ttf:
        export_ttf_file(font, bitmaps, export_ttf)
    summary = {
        "checkpoint": str(checkpoint),
        "font": str(font),
        "style": style,
        "target_saving": float(target_saving),
        "ocr_checkpoint": str(ocr_checkpoint) if ocr_checkpoint else None,
        "ocr_threshold": float(ocr_threshold),
        "void_style_weight": float(void_style_weight),
        "missing": [{"char": ch, "codepoint": f"U+{ord(ch):04X}"} for ch in missing],
        "average_metrics": mean_metrics([row["metrics"] for row in rows]),
        "glyphs": rows,
        "candidates": candidate_rows,
        "export_ttf": str(export_ttf) if export_ttf else None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if save_candidates:
        (out / "candidates.json").write_text(json.dumps(candidate_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        if candidate_sheet_rows:
            candidate_sheet(candidate_sheet_rows, out / "candidate_sheet.png")
    if sheet_rows:
        contact_sheet(sheet_rows, out / "preview.png")
    return summary
