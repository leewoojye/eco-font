from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def summarize_outputs(root: str | Path, folders: list[str]) -> dict:
    root = Path(root)
    summary: dict[str, dict] = {}
    for folder in folders:
        path = root / folder / "inference_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary[folder] = {"n": len(rows)}
        metric_keys = ["ink_saving", "saving_gap", "skeleton_recall", "component_delta", "hole_delta", "aesthetic_line_score"]
        for key in metric_keys:
            summary[folder][f"mean_{key}"] = sum(float(r["metrics"][key]) for r in rows) / len(rows)
        tess_available = [r["metrics"].get("tesseract_ocr_available") for r in rows]
        tess_matches = [r["metrics"].get("tesseract_ocr_match") for r in rows if r["metrics"].get("tesseract_ocr_match") is not None]
        tess_scores = [r["metrics"].get("tesseract_ocr_confidence") for r in rows if r["metrics"].get("tesseract_ocr_confidence") is not None]
        summary[folder]["tesseract_ocr_available_count"] = sum(1 for item in tess_available if item is True)
        summary[folder]["tesseract_ocr_accuracy"] = sum(1 for item in tess_matches if item is True) / len(tess_matches) if tess_matches else None
        summary[folder]["mean_tesseract_ocr_confidence"] = sum(float(s) for s in tess_scores) / len(tess_scores) if tess_scores else None

        matches = [r["metrics"].get("template_ocr_match") for r in rows if r["metrics"].get("template_ocr_match") is not None]
        scores = [r["metrics"].get("template_ocr_score") for r in rows if r["metrics"].get("template_ocr_score") is not None]
        summary[folder]["template_ocr_accuracy"] = sum(1 for m in matches if m is True) / len(matches) if matches else None
        summary[folder]["mean_template_ocr_score"] = sum(float(s) for s in scores) / len(scores) if scores else None
    (root / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def contact_sheet(root: str | Path, folders: list[str], labels: list[str], chars: list[str], output: str | Path, font_path: str | Path | None = None) -> Path:
    root = Path(root)
    output = Path(output)
    try:
        label_font = ImageFont.truetype(str(font_path), 16) if font_path else ImageFont.load_default()
        small_font = ImageFont.truetype(str(font_path), 12) if font_path else ImageFont.load_default()
    except Exception:
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    cell = 96
    pad = 10
    label_h = 34
    row_label_w = 130
    pair_w = cell * 2 + pad
    col_w = pair_w + pad
    cols = min(10, len(chars))
    groups = [chars[i : i + cols] for i in range(0, len(chars), cols)]
    header_h = 44
    row_h = label_h + cell
    sheet_w = pad + row_label_w + cols * col_w
    sheet_h = header_h + len(folders) * len(groups) * row_h + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 10), "Ryman-inspired eco font experiment: original | generated eco", fill=(20, 20, 20), font=label_font)
    row_index = 0
    for folder, label in zip(folders, labels):
        for group_idx, group in enumerate(groups):
            y0 = header_h + row_index * row_h
            row_label = label if group_idx == 0 else label + " cont."
            draw.text((pad, y0 + label_h + 4), row_label, fill=(30, 30, 30), font=small_font)
            for c_idx, ch in enumerate(group):
                char_id = f"u{ord(ch):04x}"
                x0 = pad + row_label_w + c_idx * col_w
                draw.text((x0 + 3, y0 + 7), ch, fill=(0, 0, 0), font=label_font)
                paths = [root / folder / "original" / f"{char_id}.png", root / folder / "eco" / f"{char_id}.png"]
                for i, p in enumerate(paths):
                    img = Image.open(p).convert("RGB") if p.exists() else Image.new("RGB", (cell, cell), (245, 245, 245))
                    if img.size != (cell, cell):
                        img = img.resize((cell, cell), Image.Resampling.NEAREST)
                    px = x0 + i * (cell + pad)
                    py = y0 + label_h
                    sheet.paste(img, (px, py))
                    draw.rectangle((px, py, px + cell - 1, py + cell - 1), outline=(210, 210, 210))
            row_index += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output
