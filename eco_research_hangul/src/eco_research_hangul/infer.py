from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import ensure_dir, load_yaml, read_chars
from .dataset import condition_from_source
from .diffusion import DiffusionSchedule
from .metrics import evaluate_sample, ink_saving
from .model import build_model
from .render import has_visible_glyph, render_glyph, save_gray


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    schedule_cfg = ckpt.get("diffusion_config", {})
    schedule = DiffusionSchedule(
        timesteps=int(schedule_cfg.get("timesteps", 64)),
        beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
        beta_end=float(schedule_cfg.get("beta_end", 0.02)),
        device=device,
    )
    return model, schedule, ckpt


@torch.no_grad()
def sample_one(
    model,
    schedule: DiffusionSchedule,
    source: np.ndarray,
    target_saving: float,
    device: torch.device,
    sample_steps: int | None,
    prediction_type: str,
) -> np.ndarray:
    condition = condition_from_source(source, target_saving)
    tensor = torch.from_numpy(condition[None]).to(device=device, dtype=torch.float32)
    sample = schedule.sample_loop(model, tensor, image_size=source.shape[0], steps=sample_steps, prediction_type=prediction_type)
    return ((sample[0, 0].detach().cpu().numpy() + 1.0) * 0.5).clip(0.0, 1.0).astype(np.float32)


def run_inference_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    base = config_path.parent.parent
    cfg = load_yaml(config_path)
    inf_cfg = cfg["inference"]
    data_cfg = cfg["data"]
    out_dir = ensure_dir(base / inf_cfg["output_dir"])
    source_dir = ensure_dir(out_dir / "source")
    generated_dir = ensure_dir(out_dir / "generated")
    target_dir = ensure_dir(out_dir / "target")
    checkpoint = base / inf_cfg["checkpoint"]
    font = Path(inf_cfg["font"])
    target_font = Path(inf_cfg["target_font"]) if inf_cfg.get("target_font") else None
    chars = read_chars(chars=inf_cfg.get("chars"), charset_file=base / inf_cfg["charset_file"] if inf_cfg.get("charset_file") else None)
    image_size = int(data_cfg.get("image_size", 96))
    font_size = int(data_cfg.get("font_size", 76))
    device = _device(str(inf_cfg.get("device", "auto")))
    model, schedule, ckpt = load_checkpoint(checkpoint, device)
    prediction_type = str(ckpt.get("diffusion_config", {}).get("prediction_type", "epsilon"))
    sample_steps = int(inf_cfg["sample_steps"]) if inf_cfg.get("sample_steps") is not None else None
    ocr_lang = str(inf_cfg.get("ocr_lang", "kor"))
    manifest = out_dir / "inference_manifest.jsonl"
    records: list[dict] = []
    with manifest.open("w", encoding="utf-8") as f:
        for ch in tqdm(chars, desc="infer"):
            source = render_glyph(font, ch, image_size=image_size, font_size=font_size)
            if not has_visible_glyph(source.image):
                continue
            target = None
            target_saving = inf_cfg.get("target_saving", "auto")
            if target_font:
                target_render = render_glyph(target_font, ch, image_size=image_size, font_size=font_size)
                if has_visible_glyph(target_render.image):
                    target = target_render.image
                    if target_saving == "auto":
                        target_saving = ink_saving(source.image, target)
            if target_saving == "auto":
                target_saving = 0.35
            generated = sample_one(model, schedule, source.image, float(target_saving), device, sample_steps, prediction_type)
            char_id = f"u{ord(ch):04x}"
            save_gray(source_dir / f"{char_id}.png", source.image)
            save_gray(generated_dir / f"{char_id}.png", generated)
            if target is not None:
                save_gray(target_dir / f"{char_id}.png", target)
            metrics = evaluate_sample(source.image, generated, target, expected_char=ch, ocr_lang=ocr_lang)
            row = {
                "font": str(font),
                "target_font": str(target_font) if target_font else None,
                "checkpoint": str(checkpoint),
                "char": ch,
                "char_id": char_id,
                "target_saving": float(target_saving),
                "source": str(Path("source") / f"{char_id}.png"),
                "generated": str(Path("generated") / f"{char_id}.png"),
                "target": str(Path("target") / f"{char_id}.png") if target is not None else None,
                "metrics": metrics,
            }
            records.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"manifest={manifest}")
    return manifest
