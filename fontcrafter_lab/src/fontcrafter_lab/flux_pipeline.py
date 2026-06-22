from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image

from .render import boundary_mask, render_glyph_mask, unique_chars
from .proxy import make_context_canvas
from .report import contact_sheet, make_mask_preview


@dataclass(frozen=True)
class FluxConfig:
    font: Path
    chars: str
    element_image: Path
    out_dir: Path
    prompt: str = "a stylized glyph made of the reference element, pure black background"
    model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    size: int = 512
    steps: int = 28
    guidance_scale: float = 30.0
    seed: int = 0
    device: str = "cuda"
    hf_home: Path | None = None
    edge_repaint: bool = True


def _load_flux_pipeline(model_id: str, device: str):
    try:
        from diffusers import FluxFillPipeline
    except Exception as exc:
        raise RuntimeError(
            "FluxFillPipeline is unavailable. Install a current diffusers build, "
            "for example: pip install -U diffusers transformers accelerate"
        ) from exc
    pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    return pipe.to(device)


def _right_half(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((w // 2, 0, w, h))


def _edge_repaint_flux(pipe, prompt: str, result: Image.Image, glyph_mask: Image.Image, element: Image.Image, cfg: FluxConfig, seed: int) -> Image.Image:
    edge = boundary_mask(glyph_mask, width=max(9, cfg.size // 42))
    image = Image.new("RGB", (cfg.size * 2, cfg.size), "black")
    image.paste(element.resize((cfg.size, cfg.size)), (0, 0))
    image.paste(result.resize((cfg.size, cfg.size)), (cfg.size, 0))
    mask = Image.new("L", (cfg.size * 2, cfg.size), 0)
    mask.paste(edge, (cfg.size, 0))
    generated = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=cfg.size,
        width=cfg.size * 2,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=max(8, cfg.steps // 2),
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    return _right_half(generated)


def run_flux(config: FluxConfig) -> dict:
    if config.hf_home:
        os.environ["HF_HOME"] = str(config.hf_home)
        os.environ["HF_HUB_CACHE"] = str(config.hf_home / "hub")
    out = config.out_dir
    glyph_dir = out / "glyphs"
    context_dir = out / "context_inputs"
    glyph_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    pipe = _load_flux_pipeline(config.model_id, config.device)
    element = Image.open(config.element_image).convert("RGB").resize((config.size, config.size), Image.Resampling.BICUBIC)
    rows = []
    sheet_rows = []
    for i, ch in enumerate(unique_chars(config.chars)):
        glyph = render_glyph_mask(config.font, ch, size=config.size)
        context, fill_mask = make_context_canvas(element, glyph.mask)
        seed = config.seed + i * 101
        generated = pipe(
            prompt=config.prompt,
            image=context,
            mask_image=fill_mask,
            height=config.size,
            width=config.size * 2,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        result = _right_half(generated)
        if config.edge_repaint:
            result = _edge_repaint_flux(pipe, config.prompt, result, glyph.mask, element, config, seed + 17)
        stem = glyph.codepoint.replace("+", "")
        mask_path = glyph_dir / f"{stem}_mask.png"
        result_path = glyph_dir / f"{stem}_fontcrafter_flux.png"
        context_path = context_dir / f"{stem}_context.png"
        fill_mask_path = context_dir / f"{stem}_inpaint_mask.png"
        glyph.mask.save(mask_path)
        result.save(result_path)
        context.save(context_path)
        fill_mask.save(fill_mask_path)
        sheet_rows.append((stem, element, make_mask_preview(glyph.mask), result))
        rows.append(
            {
                "char": ch,
                "codepoint": glyph.codepoint,
                "mask": str(mask_path.relative_to(out)),
                "result": str(result_path.relative_to(out)),
                "context_image": str(context_path.relative_to(out)),
                "inpaint_mask": str(fill_mask_path.relative_to(out)),
            }
        )
    contact_sheet(sheet_rows, out / "contact_sheet.png")
    summary = {
        "mode": "flux_in_context",
        "paper_match_note": "Uses the disclosed visual in-context inpainting formulation. Official CMA and attention-redirection weights/hooks are not included unless supplied separately.",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "glyphs": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
