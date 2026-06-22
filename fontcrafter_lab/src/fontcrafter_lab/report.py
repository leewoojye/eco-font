from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def contact_sheet(rows: list[tuple[str, Image.Image, Image.Image, Image.Image]], output: str | Path) -> Path:
    if not rows:
        raise ValueError("No rows for contact sheet")
    cell = rows[0][1].size[0]
    pad = 10
    label_h = 28
    cols = 3
    sheet = Image.new("RGB", (pad + cols * (cell + pad), pad + len(rows) * (cell + label_h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    headers = ["element", "mask", "result"]
    for c, header in enumerate(headers):
        draw.text((pad + c * (cell + pad), 2), header, fill=(0, 0, 0))
    for idx, (label, element, mask_rgb, result) in enumerate(rows):
        top = pad + idx * (cell + label_h + pad)
        y = top + label_h
        safe = label.encode("ascii", errors="ignore").decode("ascii") or f"row-{idx}"
        draw.text((pad, top), safe[:42], fill=(0, 0, 0))
        sheet.paste(element.resize((cell, cell)), (pad, y))
        sheet.paste(mask_rgb.resize((cell, cell)), (pad * 2 + cell, y))
        sheet.paste(result.resize((cell, cell)), (pad * 3 + cell * 2, y))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def make_mask_preview(mask: Image.Image) -> Image.Image:
    out = Image.new("RGB", mask.size, "black")
    out.paste(Image.new("RGB", mask.size, "white"), mask=mask.convert("L"))
    return out
