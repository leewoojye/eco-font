from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .config import ensure_dir, load_yaml, read_chars
from .guided import make_candidates, perforation_features
from .infer import _device, load_checkpoint, sample_one
from .metrics import ink_saving
from .render import has_visible_glyph, render_glyph, save_gray


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:80] or "candidate"


def _load_font(size: int, preferred: str | Path | None = None) -> ImageFont.ImageFont:
    paths = []
    if preferred is not None:
        paths.append(str(preferred))
    paths.extend(
        [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_rgb(image: np.ndarray, cell_size: int) -> Image.Image:
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L").convert("RGB")
    if pil.size != (cell_size, cell_size):
        pil = pil.resize((cell_size, cell_size), Image.Resampling.NEAREST)
    return pil


def _candidate_sheet(
    char: str,
    rows: list[dict],
    output: Path,
    cell_size: int = 96,
    label_font: str | Path | None = None,
) -> None:
    ui_font = _load_font(14)
    char_font = _load_font(14, preferred=label_font)
    small = _load_font(11)
    cols = 5
    label_h = 34
    pad = 8
    item_w = cell_size + pad
    item_h = cell_size + label_h + pad
    sheet_w = pad + cols * item_w
    sheet_h = pad + math.ceil(len(rows) / cols) * item_h + 28
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 6), char, fill=(20, 20, 20), font=char_font)
    draw.text((pad + 24, 6), "candidate generation preview", fill=(20, 20, 20), font=ui_font)
    for idx, row in enumerate(rows):
        x = pad + (idx % cols) * item_w
        y = 32 + (idx // cols) * item_h
        title = f"{idx:02d} {row['name']}"
        if len(title) > 20:
            title = title[:19] + "."
        draw.text((x, y), title, fill=(20, 20, 20), font=small)
        draw.text((x, y + 14), f"save={row['ink_saving']:.2f}", fill=(70, 70, 70), font=small)
        img = _to_rgb(row["image"], cell_size)
        py = y + label_h
        sheet.paste(img, (x, py))
        draw.rectangle((x, py, x + cell_size - 1, py + cell_size - 1), outline=(210, 210, 210))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _overview_sheet(
    char_rows: list[dict],
    output: Path,
    cell_size: int = 48,
    label_font: str | Path | None = None,
) -> None:
    ui_font = _load_font(14)
    char_font = _load_font(14, preferred=label_font)
    small = _load_font(10)
    pad = 6
    label_w = 56
    label_h = 22
    max_candidates = max((len(item["rows"]) for item in char_rows), default=0)
    sheet_w = label_w + pad + max_candidates * (cell_size + 2) + pad
    sheet_h = label_h + pad + len(char_rows) * (cell_size + 18) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 4), "candidate generation overview", fill=(20, 20, 20), font=ui_font)
    for r_idx, item in enumerate(char_rows):
        y = label_h + pad + r_idx * (cell_size + 18)
        draw.text((pad, y + 14), item["char"], fill=(20, 20, 20), font=char_font)
        for c_idx, row in enumerate(item["rows"]):
            x = label_w + c_idx * (cell_size + 2)
            sheet.paste(_to_rgb(row["image"], cell_size), (x, y))
            draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), outline=(215, 215, 215))
            draw.text((x + 2, y + cell_size + 1), str(c_idx), fill=(60, 60, 60), font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def generate_candidate_preview(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    chars: str | None = None,
    max_chars: int | None = None,
) -> Path:
    config_path = Path(config_path)
    base = config_path.parent.parent
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    inf_cfg = cfg["guided_inference"]
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = ensure_dir(base / output_dir) if output_dir else ensure_dir(base / "outputs" / "candidate_preview")
    source_dir = ensure_dir(out_dir / "source")
    candidate_dir = ensure_dir(out_dir / "candidates")
    sheet_dir = ensure_dir(out_dir / "sheets")
    manifest_path = out_dir / "candidate_preview_manifest.jsonl"

    checkpoint = base / inf_cfg["checkpoint"]
    font = Path(inf_cfg["font"])
    if not font.is_absolute():
        font = base / font
    label_font = Path(inf_cfg.get("preview_label_font", font))
    if not label_font.is_absolute():
        label_font = base / label_font
    selected_chars = read_chars(
        chars=chars if chars is not None else inf_cfg.get("chars"),
        charset_file=base / inf_cfg["charset_file"] if chars is None and inf_cfg.get("charset_file") else None,
    )
    if max_chars is not None:
        selected_chars = selected_chars[: int(max_chars)]

    image_size = int(data_cfg.get("image_size", 96))
    font_size = int(data_cfg.get("font_size", 76))
    device = _device(str(inf_cfg.get("device", "auto")))
    model, schedule, ckpt = load_checkpoint(checkpoint, device)
    prediction_type = str(ckpt.get("diffusion_config", {}).get("prediction_type", "epsilon"))
    sample_steps = int(inf_cfg["sample_steps"]) if inf_cfg.get("sample_steps") is not None else None
    target_saving = float(inf_cfg.get("target_saving", 0.42))
    diffusion_candidates = int(inf_cfg.get("diffusion_candidates", 2))

    overview_rows: list[dict] = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for ch in tqdm(selected_chars, desc="candidate-preview"):
            source = render_glyph(font, ch, image_size=image_size, font_size=font_size)
            if not has_visible_glyph(source.image):
                continue
            diffusion_images = [
                sample_one(model, schedule, source.image, target_saving, device, sample_steps, prediction_type)
                for _ in range(diffusion_candidates)
            ]
            candidates = make_candidates(source.image, diffusion_images)
            char_id = f"u{ord(ch):04x}"
            save_gray(source_dir / f"{char_id}.png", source.image)

            rows: list[dict] = []
            for idx, candidate in enumerate(candidates):
                features = perforation_features(candidate.image)
                saving = ink_saving(source.image, candidate.image)
                filename = f"{idx:02d}_{_safe_name(candidate.name)}.png"
                rel_path = Path("candidates") / char_id / filename
                save_gray(out_dir / rel_path, candidate.image)
                row = {
                    "char": ch,
                    "char_id": char_id,
                    "index": idx,
                    "name": candidate.name,
                    "source": str(Path("source") / f"{char_id}.png"),
                    "candidate": str(rel_path),
                    "ink_saving": float(saving),
                    "small_hole_count": int(features["small_hole_count"]),
                    "small_fragment_count": int(features["small_fragment_count"]),
                    "channel_area_ratio": float(features["channel_area_ratio"]),
                }
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append({**row, "image": candidate.image})
            _candidate_sheet(ch, rows, sheet_dir / f"{char_id}_candidates.png", label_font=label_font)
            overview_rows.append({"char": ch, "rows": rows})
    _overview_sheet(overview_rows, out_dir / "contact_sheet.png", label_font=label_font)
    print(f"manifest={manifest_path}")
    print(f"overview={out_dir / 'contact_sheet.png'}")
    print(f"sheets={sheet_dir}")
    return manifest_path
