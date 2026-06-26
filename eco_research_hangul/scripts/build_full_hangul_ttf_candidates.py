from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ECO_FONT_ROOT = Path(__file__).resolve().parents[2]
if str(ECO_FONT_ROOT) not in sys.path:
    sys.path.insert(0, str(ECO_FONT_ROOT))

from ecofont.server.ttf_export import export_ttf_from_bitmaps  # noqa: E402

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover - optional speed/quality dependency
    skeletonize = None

HANGUL_START = 0xAC00
HANGUL_END = 0xD7A3
FULL_HANGUL_COUNT = HANGUL_END - HANGUL_START + 1


@dataclass(frozen=True)
class StyleRecipe:
    style_id: str
    description: str
    apply: Callable[[np.ndarray, int], np.ndarray]


def log(message: str) -> None:
    print(message, flush=True)


def hangul_syllables() -> list[str]:
    return [chr(codepoint) for codepoint in range(HANGUL_START, HANGUL_END + 1)]


def font_cmap(font_path: Path) -> dict[int, str]:
    font = TTFont(str(font_path), lazy=True)
    cmap: dict[int, str] = {}
    for table in font["cmap"].tables:
        cmap.update(table.cmap)
    return cmap


def validate_full_hangul_font(font_path: Path) -> list[str]:
    if not font_path.exists():
        raise FileNotFoundError(font_path)
    font = TTFont(str(font_path), lazy=True)
    if "glyf" not in font:
        raise ValueError("source font must be a TrueType glyf TTF; CFF/OTF/TTC export is not supported")
    cmap: dict[int, str] = {}
    for table in font["cmap"].tables:
        cmap.update(table.cmap)
    chars = hangul_syllables()
    missing = [ch for ch in chars if ord(ch) not in cmap]
    if missing:
        raise ValueError(
            f"source font is not full Hangul: present={len(chars) - len(missing)}/{FULL_HANGUL_COUNT}, "
            f"first_missing=U+{ord(missing[0]):04X}"
        )
    return chars


def _binary(image: np.ndarray, threshold: float = 0.12) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) > threshold).astype(np.uint8)


def _soften(mask: np.ndarray, sigma: float = 0.45) -> np.ndarray:
    out = np.asarray(mask, dtype=np.float32)
    if sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _erode_ink(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = _binary(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=max(1, int(iterations)))
    return _soften(eroded)


def _close_perforations(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = _binary(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(iterations)))
    return _soften(closed)


def _inline_engrave(
    image: np.ndarray,
    line_width: int = 1,
    strength: float = 0.92,
    min_distance: float = 1.2,
    pre_erode: int = 0,
) -> np.ndarray:
    mask = _binary(image)
    if pre_erode > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_for_skeleton = cv2.erode(mask, kernel, iterations=pre_erode)
    else:
        mask_for_skeleton = mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if skeletonize is None:
        skel = np.logical_and(mask_for_skeleton > 0, dist >= float(min_distance))
    else:
        skel = skeletonize(mask_for_skeleton.astype(bool))
        skel = np.logical_and(skel, dist >= float(min_distance))
    skel = skel.astype(np.uint8)
    if int(line_width) > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_width), int(line_width)))
        skel = cv2.dilate(skel, kernel, iterations=1)
    out = np.clip(image, 0.0, 1.0).astype(np.float32).copy()
    out[skel > 0] *= max(0.0, 1.0 - float(strength))
    return _soften(out)


def _diagonal_perforation(
    image: np.ndarray,
    codepoint: int,
    period: int = 9,
    width: int = 2,
    keep_distance: float = 2.1,
    pre_erode: int = 0,
) -> np.ndarray:
    mask = _binary(image)
    if pre_erode > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask, kernel, iterations=pre_erode)
        if int(eroded.sum()) > 20:
            mask = eroded
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    yy, xx = np.indices(mask.shape)
    stripes = ((xx + yy + (codepoint % period)) % period) < int(width)
    dashes = ((xx - yy + ((codepoint * 3) % 17)) % 17) <= 10
    holes = stripes & dashes & (dist >= float(keep_distance))
    holes = cv2.dilate(
        holes.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    out = mask.copy()
    out[np.logical_and(holes > 0, dist >= float(keep_distance))] = 0
    return _soften(out)


def style_recipes() -> list[StyleRecipe]:
    return [
        StyleRecipe("source_original", "Original source glyph baseline.", lambda image, _cp: np.clip(image, 0, 1)),
        StyleRecipe("source_erode1", "One-step ink erosion.", lambda image, _cp: _erode_ink(image, 1)),
        StyleRecipe("source_erode2", "Two-step ink erosion.", lambda image, _cp: _erode_ink(image, 2)),
        StyleRecipe("source_inline_soft", "Light centerline engraving.", lambda image, _cp: _inline_engrave(image, 1, 0.65, 1.15)),
        StyleRecipe("source_inline_w1", "Strong one-pixel centerline engraving.", lambda image, _cp: _inline_engrave(image, 1, 0.92, 1.15)),
        StyleRecipe("source_inline_w2", "Wider centerline engraving.", lambda image, _cp: _inline_engrave(image, 2, 0.95, 1.5)),
        StyleRecipe("source_erode1_inline_soft", "One-step erosion plus light centerline engraving.", lambda image, _cp: _inline_engrave(_erode_ink(image, 1), 1, 0.65, 1.1)),
        StyleRecipe("source_inline_erode_w1", "One-step erosion plus strong one-pixel centerline engraving.", lambda image, _cp: _inline_engrave(_erode_ink(image, 1), 1, 0.92, 1.1)),
        StyleRecipe("source_inline_erode_w2", "One-step erosion plus wider centerline engraving.", lambda image, _cp: _inline_engrave(_erode_ink(image, 1), 2, 0.95, 1.4)),
        StyleRecipe("source_closed_inline_w1", "Closed counters plus strong centerline engraving.", lambda image, _cp: _inline_engrave(_close_perforations(image), 1, 0.92, 1.15)),
        StyleRecipe("source_closed_inline_w2", "Closed counters plus wider centerline engraving.", lambda image, _cp: _inline_engrave(_close_perforations(image), 2, 0.95, 1.5)),
        StyleRecipe("source_erode2_inline_soft", "Two-step erosion plus light engraving.", lambda image, _cp: _inline_engrave(_erode_ink(image, 2), 1, 0.65, 1.1)),
        StyleRecipe("source_erode2_inline_w1", "Two-step erosion plus strong engraving.", lambda image, _cp: _inline_engrave(_erode_ink(image, 2), 1, 0.92, 1.1)),
        StyleRecipe("source_erode2_inline_w2", "Two-step erosion plus wider engraving.", lambda image, _cp: _inline_engrave(_erode_ink(image, 2), 2, 0.95, 1.35)),
        StyleRecipe("source_diag_t45", "Diagonal perforation texture, conservative.", lambda image, cp: _diagonal_perforation(image, cp, 10, 1, 2.5, 0)),
        StyleRecipe("source_diag_t60", "Diagonal perforation texture, stronger.", lambda image, cp: _diagonal_perforation(image, cp, 9, 2, 2.25, 0)),
        StyleRecipe("source_erode_diag_t45", "Erosion plus conservative diagonal perforation.", lambda image, cp: _diagonal_perforation(image, cp, 10, 1, 2.15, 1)),
        StyleRecipe("source_erode_diag_t60", "Erosion plus stronger diagonal perforation.", lambda image, cp: _diagonal_perforation(image, cp, 9, 2, 1.95, 1)),
        StyleRecipe("source_closed_diag_t45", "Closed counters plus conservative diagonal perforation.", lambda image, cp: _diagonal_perforation(_close_perforations(image), cp, 10, 1, 2.5, 0)),
        StyleRecipe("source_closed_diag_t60", "Closed counters plus stronger diagonal perforation.", lambda image, cp: _diagonal_perforation(_close_perforations(image), cp, 9, 2, 2.25, 0)),
    ]


def safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in token.split("_") if part).lower() or "candidate"


def render_glyph(font: ImageFont.FreeTypeFont, ch: str, image_size: int, oversample: int) -> np.ndarray:
    scale = max(1, int(oversample))
    size = image_size * scale
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), ch, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        return np.zeros((image_size, image_size), dtype=np.float32)
    x = (size - width) // 2 - bbox[0]
    y = (size - height) // 2 - bbox[1]
    draw.text((x, y), ch, font=font, fill=255)
    if scale > 1:
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr[arr < 0.01] = 0.0
    return arr.astype(np.float32)


def has_visible_glyph(image: np.ndarray) -> bool:
    return int((image > 0.05).sum()) >= 12


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def ink_area(image: np.ndarray) -> float:
    return float(np.clip(image, 0.0, 1.0).sum())


def ink_saving(source: np.ndarray, eco: np.ndarray) -> float:
    denom = ink_area(source)
    if denom <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - ink_area(eco) / denom, -1.0, 1.0))


def load_or_render_sources(
    source_font: Path,
    chars: list[str],
    image_size: int,
    font_size: int,
    oversample: int,
) -> dict[str, np.ndarray]:
    log(f"[source] rendering {len(chars)} full Hangul glyphs at {image_size}px")
    font = ImageFont.truetype(str(source_font), font_size * max(1, int(oversample)))
    source_images: dict[str, np.ndarray] = {}
    start = time.time()
    for idx, ch in enumerate(chars, 1):
        image = render_glyph(font, ch, image_size, oversample)
        if has_visible_glyph(image):
            source_images[ch] = image
        if idx % 500 == 0 or idx == len(chars):
            elapsed = time.time() - start
            log(f"[source] {idx}/{len(chars)} rendered, visible={len(source_images)}, elapsed={elapsed:.1f}s")
    if len(source_images) != len(chars):
        raise ValueError(f"rendered visible glyph count mismatch: visible={len(source_images)}/{len(chars)}")
    return source_images


def build_style_ttf(
    source_font: Path,
    output_dir: Path,
    job_label: str,
    source_images: dict[str, np.ndarray],
    chars: list[str],
    recipe: StyleRecipe,
    style_index: int,
    total_styles: int,
    overwrite: bool,
) -> dict:
    token = safe_token(recipe.style_id)
    output_path = output_dir / f"hangul_full_{style_index:02d}_{token}.ttf"
    metrics_path = output_dir / f"hangul_full_{style_index:02d}_{token}.json"
    if output_path.exists() and metrics_path.exists() and not overwrite:
        log(f"[style {style_index + 1}/{total_styles}] skip existing {output_path.name}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    start = time.time()
    bitmaps: dict[str, np.ndarray] = {}
    ink_values: list[float] = []
    log(f"[style {style_index + 1}/{total_styles}] start {recipe.style_id}")
    for idx, ch in enumerate(chars, 1):
        source = source_images[ch]
        generated = recipe.apply(source, ord(ch))
        generated = np.clip(generated, 0.0, 1.0).astype(np.float32)
        if not has_visible_glyph(generated):
            generated = source
        bitmaps[ch] = generated
        ink_values.append(ink_saving(source, generated))
        if idx % 1000 == 0 or idx == len(chars):
            elapsed = time.time() - start
            log(f"[style {style_index + 1}/{total_styles}] {idx}/{len(chars)} bitmaps, elapsed={elapsed:.1f}s")

    family = f"Eco Hangul Full {recipe.style_id.replace('_', ' ').title()} {job_label}"
    postscript = f"EcoHangulFull{style_index:02d}{job_label}-Regular"
    _, replaced, skipped = export_ttf_from_bitmaps(
        source_font,
        bitmaps,
        output_path,
        family_name=family,
        postscript_name=postscript,
    )
    row = {
        "style_index": style_index,
        "style_id": recipe.style_id,
        "description": recipe.description,
        "path": str(output_path),
        "file_size_bytes": output_path.stat().st_size,
        "requested_glyphs": len(chars),
        "replaced_glyphs": int(replaced),
        "skipped_glyphs": len(skipped),
        "skipped_chars": skipped[:128],
        "mean_ink_saving": mean(ink_values),
        "ocr_evaluated": False,
        "elapsed_seconds": time.time() - start,
    }
    metrics_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"[style {style_index + 1}/{total_styles}] done {output_path.name}, "
        f"replaced={replaced}, skipped={len(skipped)}, elapsed={row['elapsed_seconds']:.1f}s"
    )
    del bitmaps
    gc.collect()
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 20 full Hangul eco-style TTF candidates without OCR evaluation."
    )
    parser.add_argument("--source-font", type=Path, default=Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/woojye2020/decs_jupyter_lab/eco-font/eco_research_hangul/outputs/hangul_full_ttf_20_no_ocr"),
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--font-size", type=int, default=76)
    parser.add_argument("--oversample", type=int, default=3)
    parser.add_argument("--styles", type=int, default=20)
    parser.add_argument("--job-label", default=time.strftime("%Y%m%d%H%M"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log("[run] full Hangul TTF candidate generation without OCR")
    log(f"[run] source_font={args.source_font}")
    log(f"[run] output_dir={args.output_dir}")
    chars = validate_full_hangul_font(args.source_font)
    log(f"[run] validated full Hangul coverage: {len(chars)}/{FULL_HANGUL_COUNT}")
    if args.validate_only:
        log("[run] validate-only complete")
        return
    recipes = style_recipes()[: max(1, min(args.styles, len(style_recipes())))]
    source_images = load_or_render_sources(
        args.source_font,
        chars,
        image_size=args.image_size,
        font_size=args.font_size,
        oversample=args.oversample,
    )
    rows: list[dict] = []
    manifest_path = args.output_dir / "manifest.json"
    for style_index, recipe in enumerate(recipes):
        row = build_style_ttf(
            args.source_font,
            args.output_dir,
            args.job_label,
            source_images,
            chars,
            recipe,
            style_index,
            len(recipes),
            overwrite=args.overwrite,
        )
        rows.append(row)
        manifest = {
            "status": "running",
            "source_font": str(args.source_font),
            "output_dir": str(args.output_dir),
            "hangul_glyphs": len(chars),
            "styles_requested": len(recipes),
            "styles_completed": len(rows),
            "ocr_evaluated": False,
            "elapsed_seconds": time.time() - start,
            "candidates": rows,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "status": "completed",
        "source_font": str(args.source_font),
        "output_dir": str(args.output_dir),
        "hangul_glyphs": len(chars),
        "styles_requested": len(recipes),
        "styles_completed": len(rows),
        "ocr_evaluated": False,
        "elapsed_seconds": time.time() - start,
        "candidates": rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[run] completed {len(rows)} full Hangul TTF files in {manifest['elapsed_seconds']:.1f}s")
    log(f"[run] manifest={manifest_path}")


if __name__ == "__main__":
    main()
