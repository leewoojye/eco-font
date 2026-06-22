from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .losses import generator_loss
from .metrics import candidate_score, evaluate, mean_metrics
from .model import CFFontEcoNet
from .ocr import OCREvaluator
from .priors import ECO_STYLES, input_hint_channels, make_target, project_to_budget
from .render import has_visible_glyph, read_chars_file, render_glyph, save_gray, supported_chars, unique_chars
from .report import candidate_sheet, contact_sheet
from .train import device_from_name


def _torch_load(path: str | Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _softmax_np(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(values)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-12, None)


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[CFFontEcoNet, dict[str, Any]]:
    data = _torch_load(checkpoint, map_location=device)
    model = CFFontEcoNet(**data["model_config"]).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, data


def render_ref_stack(font: str | Path, ref_chars: list[str], image_size: int) -> np.ndarray:
    refs = []
    for ch in ref_chars:
        rendered = render_glyph(font, ch, image_size=image_size)
        refs.append(rendered.image)
    return np.stack(refs)[:, None, :, :].astype(np.float32)


@torch.no_grad()
def font_embedding(model: CFFontEcoNet, refs: np.ndarray, device: torch.device) -> np.ndarray:
    images = torch.from_numpy(refs).to(device)
    embeddings = model.content_embeddings(images).detach().cpu().numpy()
    return embeddings.reshape(-1).astype(np.float32)


def infer_cfm_weights(target_embedding: np.ndarray, basis_embeddings: np.ndarray, temperature: float) -> np.ndarray:
    distances = np.abs(target_embedding[None, :] - basis_embeddings).mean(axis=1)
    return _softmax_np(-distances[None, :] / max(float(temperature), 1e-6), axis=1)[0].astype(np.float32)


def _render_basis_stack(basis_fonts: list[dict[str, Any]], char: str, image_size: int) -> np.ndarray:
    images = []
    for font in basis_fonts:
        rendered = render_glyph(font["path"], char, image_size=image_size)
        images.append(rendered.image)
    return np.stack(images)[:, None, :, :].astype(np.float32)


def refine_style_vector(
    model: CFFontEcoNet,
    font: str | Path,
    ref_chars: list[str],
    basis_fonts: list[dict[str, Any]],
    weights: np.ndarray,
    style_id: int,
    target_saving: float,
    image_size: int,
    device: torch.device,
    steps: int,
    lr: float = 0.035,
) -> torch.Tensor:
    refs = render_ref_stack(font, ref_chars, image_size)
    with torch.no_grad():
        base_style = model.encode_style_refs(torch.from_numpy(refs[None]).to(device)).detach()
    if steps <= 0:
        return base_style

    basis = []
    hints = []
    targets = []
    glyphs = []
    for ch in ref_chars:
        glyph = render_glyph(font, ch, image_size=image_size).image
        target, _score = make_target(glyph, ECO_STYLES[style_id], target_saving)
        basis.append(_render_basis_stack(basis_fonts, ch, image_size))
        hints.append(input_hint_channels(glyph, ECO_STYLES[style_id], target_saving))
        targets.append(target[None])
        glyphs.append(glyph[None])
    basis_t = torch.from_numpy(np.stack(basis)).to(device)
    hints_t = torch.from_numpy(np.stack(hints)).to(device)
    targets_t = torch.from_numpy(np.stack(targets)).to(device)
    glyphs_t = torch.from_numpy(np.stack(glyphs)).to(device)
    weights_t = torch.from_numpy(np.tile(weights[None, :], (len(ref_chars), 1))).to(device)
    style_id_t = torch.full((len(ref_chars),), style_id, dtype=torch.long, device=device)
    saving_t = torch.full((len(ref_chars),), float(target_saving), dtype=torch.float32, device=device)
    with torch.no_grad():
        content_feature = model.content_feature_from_basis(basis_t, weights_t)
    style_vec = base_style.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([style_vec], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        expanded = style_vec.expand(len(ref_chars), -1)
        logits = model.decode_from_features(content_feature, hints_t, expanded, style_id_t, saving_t)
        loss, _detail = generator_loss(logits, targets_t, glyphs_t, saving_t, hints_t[:, 1:2], pcl_weight=1.0)
        loss.backward()
        opt.step()
    return style_vec.detach()


@torch.no_grad()
def predict(
    model: CFFontEcoNet,
    basis_stack: np.ndarray,
    cfm_weights: np.ndarray,
    style_vec: torch.Tensor,
    glyph: np.ndarray,
    style_id: int,
    target_saving: float,
    device: torch.device,
) -> np.ndarray:
    basis_t = torch.from_numpy(basis_stack[None]).to(device)
    weights_t = torch.from_numpy(cfm_weights[None]).to(device)
    hints = input_hint_channels(glyph, ECO_STYLES[style_id], target_saving)
    hints_t = torch.from_numpy(hints[None]).to(device)
    style_id_t = torch.tensor([style_id], dtype=torch.long, device=device)
    saving_t = torch.tensor([float(target_saving)], dtype=torch.float32, device=device)
    content_feature = model.content_feature_from_basis(basis_t, weights_t)
    logits = model.decode_from_features(content_feature, hints_t, style_vec, style_id_t, saving_t)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    prob *= (glyph > 0.05).astype(np.float32)
    return project_to_budget(prob, glyph, target_saving)


def run_inference(
    checkpoint: str | Path,
    font: str | Path,
    chars: str,
    out_dir: str | Path,
    style: str = "auto",
    target_saving: float = 0.60,
    ocr_checkpoint: str | Path | None = None,
    image_size: int | None = None,
    device_name: str = "auto",
    ocr_threshold: float = 0.70,
    isr_steps: int = 0,
    save_candidates: bool = False,
) -> dict[str, Any]:
    device = device_from_name(device_name)
    model, data = load_model(checkpoint, device)
    dataset_summary = data["dataset_summary"]
    image_size = int(image_size or dataset_summary["config"]["image_size"])
    ref_chars = list(data.get("ref_chars") or dataset_summary.get("ref_chars") or dataset_summary["chars"][:8])
    basis_fonts = list(data["basis_fonts"])
    basis_ids = list(data["basis_font_indices"])
    font_embeddings = np.asarray(data["font_embeddings"], dtype=np.float32)
    basis_embeddings = font_embeddings[basis_ids]
    target_refs = render_ref_stack(font, ref_chars, image_size)
    target_embedding = font_embedding(model, target_refs, device)
    weights = infer_cfm_weights(target_embedding, basis_embeddings, model.model_config["cfm_temperature"])

    requested, missing = supported_chars(font, unique_chars(chars))
    out = Path(out_dir)
    original_dir = out / "original"
    eco_dir = out / "eco"
    candidate_dir = out / "candidates"
    original_dir.mkdir(parents=True, exist_ok=True)
    eco_dir.mkdir(parents=True, exist_ok=True)
    if save_candidates:
        candidate_dir.mkdir(parents=True, exist_ok=True)
    evaluator = OCREvaluator(ocr_checkpoint, device=device_name) if ocr_checkpoint else None
    valid_styles = ECO_STYLES if style == "auto" else [style]
    for item in valid_styles:
        if item not in ECO_STYLES:
            raise ValueError(f"Unknown style {item}. Valid: auto,{','.join(ECO_STYLES)}")

    style_vectors: dict[str, torch.Tensor] = {}
    for item in valid_styles:
        style_id = ECO_STYLES.index(item)
        style_vectors[item] = refine_style_vector(
            model,
            font,
            ref_chars,
            basis_fonts,
            weights,
            style_id,
            target_saving,
            image_size,
            device,
            steps=isr_steps,
        )

    rows = []
    candidate_rows = []
    candidate_sheet_rows = []
    sheet_rows = []
    for ch in tqdm(requested, desc="infer", unit="glyph"):
        rendered = render_glyph(font, ch, image_size=image_size)
        if not has_visible_glyph(rendered.image):
            continue
        basis_stack = _render_basis_stack(basis_fonts, ch, image_size)
        candidates = []
        images = []
        image_chars = []
        for item in valid_styles:
            style_id = ECO_STYLES.index(item)
            eco = predict(model, basis_stack, weights, style_vectors[item], rendered.image, style_id, target_saving, device)
            metrics = evaluate(rendered.image, eco, target_saving)
            candidates.append({"style": item, "eco": eco, "metrics": metrics})
            images.append(eco)
            image_chars.append(ch)
        ocr_rows = evaluator.score(images, image_chars) if evaluator else [None] * len(candidates)
        best_i = max(range(len(candidates)), key=lambda i: candidate_score(candidates[i]["metrics"], ocr_rows[i], ocr_threshold))
        if candidate_score(candidates[best_i]["metrics"], ocr_rows[best_i], ocr_threshold) < -999.0:
            best_i = max(
                range(len(candidates)),
                key=lambda i: (
                    float(ocr_rows[i].get("ocr_confidence", 0.0)) if ocr_rows[i] else 0.0,
                    float(candidates[i]["metrics"]["skeleton_recall"]),
                ),
            )
        chosen = candidates[best_i]
        ocr = ocr_rows[best_i]
        char_id = f"u{ord(ch):04x}"
        save_gray(original_dir / f"{char_id}.png", rendered.image)
        save_gray(eco_dir / f"{char_id}.png", chosen["eco"])

        if save_candidates:
            glyph_candidate_dir = candidate_dir / char_id
            glyph_candidate_dir.mkdir(parents=True, exist_ok=True)
            sheet_candidates = []
            for i, candidate in enumerate(candidates):
                candidate_metrics = dict(candidate["metrics"])
                if ocr_rows[i]:
                    candidate_metrics.update(ocr_rows[i])
                selected = i == best_i
                name = f"{i:02d}_{candidate['style']}{'_selected' if selected else ''}.png"
                save_gray(glyph_candidate_dir / name, candidate["eco"])
                candidate_rows.append(
                    {
                        "char": ch,
                        "char_id": char_id,
                        "style": candidate["style"],
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

        metrics = dict(chosen["metrics"])
        if ocr:
            metrics.update(ocr)
        rows.append(
            {
                "char": ch,
                "char_id": char_id,
                "chosen_style": chosen["style"],
                "target_saving": float(target_saving),
                "original": str(Path("original") / f"{char_id}.png"),
                "eco": str(Path("eco") / f"{char_id}.png"),
                "metrics": metrics,
            }
        )
        if len(sheet_rows) < 24:
            sheet_rows.append((f"{char_id} {chosen['style']}", rendered.image, chosen["eco"]))

    summary = {
        "checkpoint": str(checkpoint),
        "font": str(font),
        "basis_fonts": basis_fonts,
        "basis_weights_for_target": weights.tolist(),
        "style": style,
        "target_saving": float(target_saving),
        "ocr_checkpoint": str(ocr_checkpoint) if ocr_checkpoint else None,
        "ocr_threshold": float(ocr_threshold),
        "isr_steps": int(isr_steps),
        "missing": [{"char": ch, "codepoint": f"U+{ord(ch):04X}"} for ch in missing],
        "average_metrics": mean_metrics([row["metrics"] for row in rows]),
        "glyphs": rows,
        "candidates": candidate_rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if sheet_rows:
        contact_sheet(sheet_rows, out / "preview.png")
    if save_candidates and candidate_sheet_rows:
        (out / "candidates.json").write_text(json.dumps(candidate_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        candidate_sheet(candidate_sheet_rows, out / "candidate_sheet.png")
    return summary


def chars_from_optional_file(chars: str | None, chars_file: Path | None) -> str:
    if chars_file:
        return read_chars_file(chars_file)
    if chars:
        return chars
    raise ValueError("Either --chars or --chars-file is required")
