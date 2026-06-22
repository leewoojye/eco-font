from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .font_bank import FontInfo, info_to_dict, inventory, select_base_font, select_style_fonts
from .fusion import Candidate, generate_param_grid, fuse_images
from .metrics import score_candidate
from .render import has_visible_glyph, load_ui_font, render_char, save_gray, to_rgb_tile


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_dir"] = str(path.parent)
    config["_root_dir"] = str(path.parent.parent)
    return config


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def read_charset(root: Path, charset_file: str) -> list[str]:
    text = resolve_path(root, charset_file).read_text(encoding="utf-8")
    chars = [ch for ch in text.strip() if not ch.isspace()]
    return list(dict.fromkeys(chars))


def resolve_font_dirs(root: Path, config: dict) -> list[Path]:
    return [resolve_path(root, item) for item in config["font_dirs"]]


def build_template_bank(chars: list[str], fonts: list[FontInfo], image_size: int, font_size: int) -> dict[str, list[np.ndarray]]:
    bank: dict[str, list[np.ndarray]] = {ch: [] for ch in chars}
    for ch in chars:
        for font in fonts:
            try:
                rendered = render_char(font.path, ch, image_size=image_size, font_size=font_size)
            except Exception:
                continue
            if has_visible_glyph(rendered.image):
                bank[ch].append(rendered.image)
    return bank


def _candidate_name(style: FontInfo, index: int, params: dict) -> str:
    mode = params["eco_mode"]
    return f"{style.family[:18].replace(' ', '_')}_{index:03d}_{mode}"


def generate_for_char(
    char: str,
    base_font: FontInfo,
    style_fonts: list[FontInfo],
    template_bank: dict[str, list[np.ndarray]],
    config: dict,
) -> tuple[np.ndarray, list[tuple[Candidate, dict]]]:
    image_size = int(config["image_size"])
    font_size = int(config["font_size"])
    base = render_char(base_font.path, char, image_size=image_size, font_size=font_size)
    if not has_visible_glyph(base.image):
        return base.image, []

    scored: list[tuple[Candidate, dict]] = []
    target = float(config["target_ink_saving"])
    weights = config["weights"]
    rng = random.Random(int(config["random_seed"]) + ord(char))

    for style_font in style_fonts:
        try:
            style = render_char(style_font.path, char, image_size=image_size, font_size=font_size)
        except Exception:
            continue
        if not has_visible_glyph(style.image):
            continue
        params_grid = generate_param_grid(config, str(style_font.path))
        rng.shuffle(params_grid)
        # Keep the grid bounded while still sampling each style family.
        for i, params in enumerate(params_grid[: max(8, int(config["candidate_limit_per_char"]))]):
            image = fuse_images(base.image, style.image, params)
            diversity = min(1.0, 0.20 + 0.20 * abs(params.width_scale - 1.0) / 0.08 + 0.20 * abs(params.slant) / 0.08 + 0.25 * params.alpha)
            score = score_candidate(base.image, image, char, template_bank, target, weights, diversity_bonus=diversity)
            candidate = Candidate(
                name=_candidate_name(
                    style_font,
                    i,
                    {
                        "eco_mode": params.eco_mode,
                    },
                ),
                image=image,
                params=params,
                style_family=style_font.family,
            )
            scored.append((candidate, score.to_dict()))

    scored.sort(key=lambda item: item[1]["total"], reverse=True)
    return base.image, scored


def _script_font(script_font_path: str | Path, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(script_font_path), size)
    except Exception:
        return load_ui_font(size)


def _write_char_sheet(
    char: str,
    char_id: str,
    source: np.ndarray,
    selected: list[dict],
    out_path: Path,
    base_name: str,
    script_font_path: str | Path,
) -> None:
    cols = 5
    tile = 104
    cell_w = 148
    cell_h = 166
    header_h = 44
    rows = int(np.ceil((len(selected) + 1) / cols))
    sheet = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_ui_font(14, bold=True)
    small_font = load_ui_font(10)
    cherokee_font = _script_font(script_font_path, 18)
    draw.text((8, 5), char, fill="black", font=cherokee_font)
    draw.text((36, 8), f"/ {char_id} style-fusion candidates", fill="black", font=title_font)
    draw.text((8, 25), f"base={base_name}", fill="black", font=small_font)

    cells = [
        {
            "name": "source",
            "image": source,
            "score": {"ink_saving": 0.0, "aesthetic": 0.0, "readability_margin": 0.0, "total": 0.0},
        }
    ] + selected
    for idx, item in enumerate(cells):
        col = idx % cols
        row = idx // cols
        x = col * cell_w + 8
        y = header_h + row * cell_h + 4
        score = item["score"]
        draw.text((x, y), f"{idx:02d} {item['name'][:21]}", fill="black", font=small_font)
        draw.text((x, y + 12), f"save={score['ink_saving']:.2f} aes={score['aesthetic']:.2f}", fill="black", font=small_font)
        draw.text((x, y + 24), f"read={score['readability_margin']:.2f} total={score['total']:.2f}", fill="black", font=small_font)
        sheet.paste(to_rgb_tile(item["image"], tile), (x, y + 46))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _write_contact_sheet(rows: list[dict], out_path: Path, script_font_path: str | Path) -> None:
    cols = 9
    tile = 72
    label_w = 82
    cell_w = 96
    header_h = 64
    row_h = 98
    sheet = Image.new("RGB", (label_w + cols * cell_w, header_h + len(rows) * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_ui_font(14, bold=True)
    small_font = load_ui_font(10)
    cherokee_font = _script_font(script_font_path, 16)
    draw.text((8, 8), "Cherokee style fusion + aesthetic prior", fill="black", font=title_font)
    draw.text((label_w + 8, 38), "source", fill="black", font=small_font)
    for col in range(1, cols):
        draw.text((label_w + col * cell_w + 8, 38), f"top{col}", fill="black", font=small_font)

    for r, row in enumerate(rows):
        y = header_h + r * row_h
        draw.text((8, y + 12), row["char"], fill="black", font=cherokee_font)
        draw.text((8, y + 38), row["char_id"].upper(), fill="black", font=small_font)
        sheet.paste(to_rgb_tile(row["source"], tile), (label_w + 8, y + 6))
        for c, item in enumerate(row["selected"][: cols - 1], start=1):
            x = label_w + c * cell_w + 8
            sheet.paste(to_rgb_tile(item["image"], tile), (x, y + 6))
            draw.text((x, y + 80), f"{item['score']['ink_saving']:.2f}/{item['score']['aesthetic']:.2f}", fill="black", font=small_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def run(config_path: str | Path, out_dir: str | Path | None = None) -> Path:
    config = load_config(config_path)
    root = Path(config["_root_dir"]).resolve()
    random.seed(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]))

    chars = read_charset(root, config["charset_file"])
    font_dirs = resolve_font_dirs(root, config)
    fonts = inventory(font_dirs)
    usable = [font for font in fonts if font.usable]
    base = select_base_font(usable, config["base_font_preference"])
    style_fonts = select_style_fonts(usable, base, int(config["style_font_limit"]))
    eval_fonts = [base] + style_fonts
    template_bank = build_template_bank(chars, eval_fonts, int(config["image_size"]), int(config["font_size"]))

    out = Path(out_dir) if out_dir else root / "outputs" / config["experiment_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(exist_ok=True)
    (out / "source").mkdir(exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)

    manifest_path = out / "manifest.jsonl"
    rows: list[dict] = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for char in chars:
            char_id = f"u{ord(char):04x}"
            source, scored = generate_for_char(char, base, style_fonts, template_bank, config)
            save_gray(out / "source" / f"{char_id}.png", source)
            selected_records: list[dict] = []
            for rank, (candidate, score) in enumerate(scored[: int(config["top_k_per_char"])], start=1):
                rel = Path("candidates") / char_id / f"{rank:02d}_{candidate.name}.png"
                save_gray(out / rel, candidate.image)
                record = {
                    "char": char,
                    "char_id": char_id,
                    "rank": rank,
                    "name": candidate.name,
                    "path": str(rel),
                    "style_family": candidate.style_family,
                    "params": {
                        "alpha": candidate.params.alpha,
                        "weight_delta": candidate.params.weight_delta,
                        "width_scale": candidate.params.width_scale,
                        "slant": candidate.params.slant,
                        "eco_mode": candidate.params.eco_mode,
                        "style_font": candidate.params.style_font,
                    },
                    "score": score,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                selected_records.append({"name": candidate.name, "image": candidate.image, "score": score})
            _write_char_sheet(
                char,
                char_id,
                source,
                selected_records,
                out / "sheets" / f"{char_id}_candidates.png",
                base.full_name,
                base.path,
            )
            rows.append({"char": char, "char_id": char_id, "source": source, "selected": selected_records})

    _write_contact_sheet(rows, out / "contact_sheet.png", base.path)
    summary = {
        "method": "Cherokee style fusion + aesthetic prior",
        "config": str(Path(config_path).resolve()),
        "output_dir": str(out.resolve()),
        "chars": len(chars),
        "base_font": info_to_dict(base),
        "style_fonts": [info_to_dict(font) for font in style_fonts],
        "usable_fonts_seen": len(usable),
        "manifest": "manifest.jsonl",
        "contact_sheet": "contact_sheet.png",
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out
