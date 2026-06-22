from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import ensure_dir
from .diffusion import DiffusionSchedule
from .diffusion_model import build_diffusion_model
from .evaluator import enforce_ink_budget, evaluate_eco_candidate
from .font_render import has_visible_glyph, model_input_channels, render_glyph, save_gray
from .metrics import ink_saving
from .vectorize import export_ttf_from_bitmaps


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_diffusion_checkpoint(checkpoint_path: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("model", {})
    diffusion_cfg = ckpt.get("diffusion_config") or ckpt.get("config", {}).get("diffusion", {})
    model = build_diffusion_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    schedule = DiffusionSchedule(
        timesteps=int(diffusion_cfg.get("timesteps", 64)),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
        device=device,
    )
    return model, schedule


@torch.no_grad()
def generate_diffusion_candidates(
    model,
    schedule: DiffusionSchedule,
    glyph: np.ndarray,
    target_saving: float,
    num_candidates: int,
    sample_steps: int | None,
    device: torch.device,
    force_ink_budget: bool,
    allow_outline_shift: int,
) -> list[np.ndarray]:
    condition_np = model_input_channels(glyph, target_saving)
    condition = torch.from_numpy(condition_np[None]).to(device=device, dtype=torch.float32)
    condition = condition.repeat(num_candidates, 1, 1, 1)
    samples = schedule.sample_loop(
        model,
        condition,
        image_size=glyph.shape[0],
        channels=1,
        steps=sample_steps,
    )
    images = ((samples[:, 0].detach().cpu().numpy() + 1.0) * 0.5).clip(0.0, 1.0)
    out: list[np.ndarray] = []
    for img in images:
        if force_ink_budget:
            img = enforce_ink_budget(img, glyph, target_saving, allow_outline_shift=allow_outline_shift)
        out.append(img.astype(np.float32))
    return out


def sample_diffusion(
    checkpoint: str | Path,
    font: str | Path,
    chars: list[str],
    out_dir: str | Path,
    target_saving: float,
    image_size: int = 96,
    font_size: int = 76,
    num_candidates: int = 4,
    sample_steps: int | None = None,
    force_ink_budget: bool = True,
    allow_outline_shift: int = 2,
    ocr_lang: str | None = None,
    template_ocr: bool = True,
    device_name: str = "auto",
    export_ttf: str | Path | None = None,
) -> Path:
    device = _device(device_name)
    out = ensure_dir(out_dir)
    original_dir = ensure_dir(out / "original")
    best_dir = ensure_dir(out / "best_eco")
    candidate_root = ensure_dir(out / "candidates")
    manifest_path = out / "diffusion_manifest.jsonl"
    model, schedule = load_diffusion_checkpoint(checkpoint, device)
    bitmaps_for_ttf: dict[str, np.ndarray] = {}
    template_images = {}
    if template_ocr:
        for template_char in chars:
            template_rendered = render_glyph(font, template_char, image_size=image_size, font_size=font_size)
            if has_visible_glyph(template_rendered.image):
                template_images[template_char] = template_rendered.image

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for ch in tqdm(chars, desc="diff-sample"):
            rendered = render_glyph(font, ch, image_size=image_size, font_size=font_size)
            if not has_visible_glyph(rendered.image):
                continue
            char_id = f"u{ord(ch):04x}"
            save_gray(original_dir / f"{char_id}.png", rendered.image)
            candidates = generate_diffusion_candidates(
                model,
                schedule,
                rendered.image,
                target_saving,
                num_candidates=num_candidates,
                sample_steps=sample_steps,
                device=device,
                force_ink_budget=force_ink_budget,
                allow_outline_shift=allow_outline_shift,
            )
            evaluated = []
            for idx, candidate in enumerate(candidates):
                candidate_dir = ensure_dir(candidate_root / f"{idx:02d}")
                candidate_path = candidate_dir / f"{char_id}.png"
                save_gray(candidate_path, candidate)
                metrics = evaluate_eco_candidate(
                    rendered.image,
                    candidate,
                    target_saving=target_saving,
                    expected_char=ch,
                    ocr_lang=ocr_lang,
                    template_ocr=template_images if template_ocr else None,
                )
                evaluated.append((metrics.score, idx, candidate, metrics, candidate_path))
            evaluated.sort(key=lambda item: item[0])
            _score, best_idx, best_candidate, best_metrics, best_candidate_path = evaluated[0]
            save_gray(best_dir / f"{char_id}.png", best_candidate)
            bitmaps_for_ttf[ch] = best_candidate
            record = {
                "font_path": str(font),
                "char": ch,
                "char_id": char_id,
                "target_saving": float(target_saving),
                "actual_ink_saving": float(ink_saving(rendered.image, best_candidate)),
                "original": str(Path("original") / f"{char_id}.png"),
                "best_eco": str(Path("best_eco") / f"{char_id}.png"),
                "best_candidate": best_idx,
                "best_candidate_path": str(best_candidate_path.relative_to(out)),
                "best_metrics": best_metrics.to_dict(),
                "candidates": [
                    {
                        "index": idx,
                        "path": str(path.relative_to(out)),
                        "metrics": metrics.to_dict(),
                    }
                    for _score, idx, _candidate, metrics, path in evaluated
                ],
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    if export_ttf:
        export_ttf_from_bitmaps(font, bitmaps_for_ttf, export_ttf)
    return manifest_path
