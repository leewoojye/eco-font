from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def summarize(root: str | Path) -> dict:
    root = Path(root)
    manifest = root / "inference_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary: dict[str, float | int | None] = {"n": len(rows)}
    metric_keys = ["ink_saving", "target_ink_saving", "target_mse", "target_iou"]
    for key in metric_keys:
        values = [r["metrics"].get(key) for r in rows if r["metrics"].get(key) is not None]
        summary[f"mean_{key}"] = sum(float(v) for v in values) / len(values) if values else None
    matches = [r["metrics"].get("tesseract_match") for r in rows if r["metrics"].get("tesseract_match") is not None]
    exact = [r["metrics"].get("tesseract_exact_match") for r in rows if r["metrics"].get("tesseract_exact_match") is not None]
    conf = [r["metrics"].get("tesseract_confidence") for r in rows if r["metrics"].get("tesseract_confidence") is not None]
    summary["tesseract_match_accuracy"] = sum(1 for item in matches if item is True) / len(matches) if matches else None
    summary["tesseract_exact_accuracy"] = sum(1 for item in exact if item is True) / len(exact) if exact else None
    summary["mean_tesseract_confidence"] = sum(float(v) for v in conf) / len(conf) if conf else None
    objectives = [r["metrics"].get("objective") for r in rows if isinstance(r["metrics"].get("objective"), dict)]
    if objectives:
        for key in [
            "score",
            "style_score",
            "channel_score",
            "hole_penalty",
            "fragment_penalty",
            "small_hole_count",
            "small_fragment_count",
            "channel_area_ratio",
        ]:
            values = [obj.get(key) for obj in objectives if obj.get(key) is not None]
            summary[f"mean_objective_{key}"] = sum(float(v) for v in values) / len(values) if values else None
        for key in ["passes_ocr_gate", "passes_design_gate"]:
            values = [obj.get(key) for obj in objectives if obj.get(key) is not None]
            summary[f"{key}_rate"] = sum(1 for item in values if item is True) / len(values) if values else None
    (root / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def contact_sheet(root: str | Path, output: str | Path, label_font: str | Path | None = None) -> Path:
    root = Path(root)
    output = Path(output)
    manifest = root / "inference_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        font = ImageFont.truetype(str(label_font), 15) if label_font else ImageFont.load_default()
        small = ImageFont.truetype(str(label_font), 11) if label_font else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    rows = rows[:24]
    image_keys = ["source", "generated", "target"] if any(row.get("target") for row in rows) else ["source", "generated"]
    cell = 96
    pad = 8
    label_h = 24
    col_w = cell * len(image_keys) + pad * (len(image_keys) - 1)
    cols = 4
    item_h = label_h + cell
    sheet_w = pad + cols * (col_w + pad)
    sheet_h = pad + ((len(rows) + cols - 1) // cols) * (item_h + pad) + 28
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    header = "source | generated | real eco target" if "target" in image_keys else "source | guided generated"
    draw.text((pad, 8), header, fill=(20, 20, 20), font=font)
    y_start = 32
    for idx, row in enumerate(rows):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (col_w + pad)
        y0 = y_start + r * (item_h + pad)
        text = f"{row['char']} save={row['metrics'].get('ink_saving', 0):.2f}"
        draw.text((x0, y0), text, fill=(20, 20, 20), font=small)
        for j, key in enumerate(image_keys):
            rel = row.get(key)
            if not rel:
                img = Image.new("RGB", (cell, cell), (245, 245, 245))
            else:
                img = Image.open(root / rel).convert("RGB")
            if img.size != (cell, cell):
                img = img.resize((cell, cell), Image.Resampling.NEAREST)
            px = x0 + j * (cell + pad)
            py = y0 + label_h
            sheet.paste(img, (px, py))
            draw.rectangle((px, py, px + cell - 1, py + cell - 1), outline=(210, 210, 210))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def make_report(root: str | Path, output: str | Path, label_font: str | Path | None = None) -> dict:
    summary = summarize(root)
    path = contact_sheet(root, output, label_font=label_font)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"contact_sheet={path}")
    return summary
