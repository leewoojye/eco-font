from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .elements import make_element
from .render import boundary_mask, render_glyph_mask, unique_chars
from .report import contact_sheet, make_mask_preview


@dataclass(frozen=True)
class ProxyConfig:
    font: Path
    chars: str
    out_dir: Path
    element_kind: str = "blue_stone"
    size: int = 512
    seed: int = 7
    edge_repaint: bool = True


def _tile_texture(element: Image.Image, size: int) -> np.ndarray:
    src = element.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(src, dtype=np.float32)


def _distance_fields(mask: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary = (np.asarray(mask.convert("L"), dtype=np.uint8) > 16).astype(np.uint8)
    inside_dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside_dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    edge = np.exp(-((inside_dist - 2.0) ** 2) / 26.0) * binary
    return binary.astype(bool), inside_dist.astype(np.float32), edge.astype(np.float32)


def _shape_aware_texture(element: Image.Image, mask: Image.Image, seed: int) -> Image.Image:
    size = mask.size[0]
    rng = np.random.default_rng(seed)
    texture = _tile_texture(element, size)
    binary, dist, edge = _distance_fields(mask)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    flow_x = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), size / 36)
    flow_y = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), size / 36)
    amp = 10.0 + 7.0 * np.clip(dist / max(float(dist.max()), 1.0), 0.0, 1.0)
    map_x = np.clip(xx + flow_x * amp, 0, size - 1).astype(np.float32)
    map_y = np.clip(yy + flow_y * amp, 0, size - 1).astype(np.float32)
    warped = cv2.remap(texture, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    light = 0.62 + 0.42 * np.clip(dist / max(float(dist.max()), 1.0), 0.0, 1.0)
    line = 0.88 + 0.16 * np.sin((xx * 0.05 + yy * 0.03 + dist * 0.45))
    styled = warped * light[..., None] * line[..., None]
    styled = np.clip(styled, 0, 255)
    out = np.zeros_like(styled)
    out[binary] = styled[binary]
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def _object_texture(element: Image.Image, mask: Image.Image, seed: int) -> Image.Image:
    size = mask.size[0]
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    elem = np.asarray(element.resize((size, size)), dtype=np.uint8)
    binary, dist, _edge = _distance_fields(mask)
    ys, xs = np.where(binary & (dist > 3))
    if len(xs) == 0:
        return Image.fromarray(base, mode="RGB")
    count = max(32, len(xs) // 520)
    for _ in range(count):
        idx = int(rng.integers(0, len(xs)))
        cx, cy = int(xs[idx]), int(ys[idx])
        r = int(rng.integers(max(4, size // 70), max(7, size // 34)))
        color = tuple(int(v) for v in elem[cy, cx])
        overlay = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(overlay, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
        overlay = cv2.bitwise_and(overlay, (binary.astype(np.uint8) * 255))
        shade = cv2.GaussianBlur(overlay, (0, 0), max(1.0, r / 3.0)).astype(np.float32) / 255.0
        for channel in range(3):
            base[..., channel] = np.maximum(base[..., channel], (shade * color[channel]).astype(np.uint8))
    texture = _shape_aware_texture(element, mask, seed + 911)
    tex_arr = np.asarray(texture, dtype=np.uint8)
    base[binary & (base.sum(axis=-1) == 0)] = (tex_arr[binary & (base.sum(axis=-1) == 0)] * 0.55).astype(np.uint8)
    return Image.fromarray(base, mode="RGB")


def edge_repaint(result: Image.Image, element: Image.Image, mask: Image.Image, seed: int) -> Image.Image:
    size = mask.size[0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(result.convert("RGB"), dtype=np.float32)
    texture = _tile_texture(element, size)
    edge = np.asarray(boundary_mask(mask, width=max(7, size // 48)).convert("L"), dtype=np.float32) / 255.0
    jitter = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), size / 80)
    strength = np.clip(edge * (0.35 + 0.45 * jitter), 0.0, 0.85)
    arr = arr * (1.0 - strength[..., None]) + texture * strength[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def make_context_canvas(element: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    size = mask.size[0]
    image = Image.new("RGB", (size * 2, size), "black")
    image.paste(element.convert("RGB").resize((size, size)), (0, 0))
    fill_mask = Image.new("L", (size * 2, size), 0)
    fill_mask.paste(mask.convert("L"), (size, 0))
    return image, fill_mask


def stylize_proxy(element: Image.Image, mask: Image.Image, seed: int, edge: bool = True) -> Image.Image:
    kind_score = np.asarray(element.convert("RGB"), dtype=np.float32).std(axis=(0, 1)).mean()
    if kind_score > 58:
        out = _object_texture(element, mask, seed)
    else:
        out = _shape_aware_texture(element, mask, seed)
    if edge:
        out = edge_repaint(out, element, mask, seed + 37)
    return out


def run_proxy(config: ProxyConfig) -> dict:
    out = config.out_dir
    glyph_dir = out / "glyphs"
    context_dir = out / "context_inputs"
    glyph_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    element = make_element(config.element_kind, config.size, config.seed)
    element_path = out / f"element_{config.element_kind}.png"
    element.save(element_path)

    rows = []
    sheet_rows = []
    for i, ch in enumerate(unique_chars(config.chars)):
        glyph = render_glyph_mask(config.font, ch, size=config.size)
        result = stylize_proxy(element, glyph.mask, config.seed + i * 101, edge=config.edge_repaint)
        context, fill_mask = make_context_canvas(element, glyph.mask)
        stem = glyph.codepoint.replace("+", "")
        mask_path = glyph_dir / f"{stem}_mask.png"
        result_path = glyph_dir / f"{stem}_fontcrafter_proxy.png"
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
        "mode": "local_proxy",
        "paper_match_note": "Proxy only. It follows FontCrafter input formulation and post-processing ideas, but not the unpublished CMA weights or attention hooks.",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "element": str(element_path.relative_to(out)),
        "glyphs": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
