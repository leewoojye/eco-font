from __future__ import annotations

import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .research_runner import API_OUTPUT_ROOT, ECO_FONT_ROOT, RESEARCH_SRC
from .schemas import FontGenerationSpec
from .script_sets import chars_for_codepoint_set
from .ttf_export import export_ttf_from_bitmaps

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover - fallback for stripped deployments
    skeletonize = None


def _ensure_research_import_path() -> None:
    path = str(RESEARCH_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)


def _asset_url(job_id: str, rel_path: str | Path) -> str:
    rel = Path(rel_path).as_posix().lstrip("/")
    return f"/v1/assets/{quote(job_id)}/{quote(rel, safe='/')}"


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in token.split("_") if part).lower() or "candidate"


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
    holes = cv2.dilate(holes.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    out = mask.copy()
    out[np.logical_and(holes > 0, dist >= float(keep_distance))] = 0
    return _soften(out)


@dataclass(frozen=True)
class StyleRecipe:
    style_id: str
    description: str
    apply: Callable[[np.ndarray, int], np.ndarray]


def _style_recipes() -> list[StyleRecipe]:
    return [
        StyleRecipe("source_original", "Original source glyph baseline.", lambda image, _cp: np.clip(image, 0, 1)),
        StyleRecipe("source_erode1", "One-step ink erosion.", lambda image, _cp: _erode_ink(image, 1)),
        StyleRecipe("source_erode2", "Two-step ink erosion.", lambda image, _cp: _erode_ink(image, 2)),
        StyleRecipe("source_inline_soft", "Light centerline engraving.", lambda image, _cp: _inline_engrave(image, 1, 0.65, 1.15)),
        StyleRecipe("source_inline_w1", "Strong one-pixel centerline engraving.", lambda image, _cp: _inline_engrave(image, 1, 0.92, 1.15)),
        StyleRecipe("source_inline_w2", "Wider centerline engraving.", lambda image, _cp: _inline_engrave(image, 2, 0.95, 1.5)),
        StyleRecipe(
            "source_erode1_inline_soft",
            "One-step erosion plus light centerline engraving.",
            lambda image, _cp: _inline_engrave(_erode_ink(image, 1), 1, 0.65, 1.1),
        ),
        StyleRecipe(
            "source_inline_erode_w1",
            "One-step erosion plus strong one-pixel centerline engraving.",
            lambda image, _cp: _inline_engrave(_erode_ink(image, 1), 1, 0.92, 1.1),
        ),
        StyleRecipe(
            "source_inline_erode_w2",
            "One-step erosion plus wider centerline engraving.",
            lambda image, _cp: _inline_engrave(_erode_ink(image, 1), 2, 0.95, 1.4),
        ),
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


def _render_glyph(font_path: Path, ch: str, image_size: int, font_size: int, oversample: int = 3) -> np.ndarray:
    scale = max(1, int(oversample))
    size = image_size * scale
    font = ImageFont.truetype(str(font_path), font_size * scale)
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


def _visible(image: np.ndarray) -> bool:
    return int((image > 0.05).sum()) >= 12


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _sample_chars(chars: list[str], mode: str, sample_size: int) -> list[str]:
    if mode == "none":
        return []
    if mode == "full" or len(chars) <= sample_size:
        return list(chars)
    indices = np.linspace(0, len(chars) - 1, num=sample_size, dtype=int)
    return [chars[int(idx)] for idx in indices]


class UploadedFontGenerationRunner:
    def __init__(self, output_root: Path = API_OUTPUT_ROOT) -> None:
        self.output_root = output_root

    def job_root(self, job_id: str) -> Path:
        return self.output_root / job_id / "font_generation"

    def prepare_spec(self, spec: FontGenerationSpec) -> None:
        if spec.method != "eco_research_guided":
            raise ValueError(f"method '{spec.method}' is not implemented for uploaded TTF batch generation")
        if spec.script != "cherokee":
            raise ValueError("uploaded TTF batch generation currently supports script='cherokee'")
        if spec.use_diffusion:
            raise ValueError("use_diffusion=true is not implemented for full-TTF batch generation yet")
        if spec.codepoint_set not in {"cherokee_full", "uploaded_cherokee"}:
            raise ValueError("Cherokee TTF generation requires codepoint_set='cherokee_full' or 'uploaded_cherokee'")

    def save_upload(self, job_id: str, filename: str, data: bytes) -> Path:
        suffix = Path(filename or "input.ttf").suffix.lower()
        if suffix not in {".ttf", ".otf"}:
            raise ValueError("font upload must be a .ttf or .otf file")
        input_dir = self.job_root(job_id) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        target = input_dir / f"input{suffix}"
        target.write_bytes(data)
        try:
            font = TTFont(str(target), lazy=True)
            if "glyf" not in font:
                raise ValueError("uploaded font must be a TrueType glyf font; CFF/OTF outlines are not supported yet")
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target

    def run(self, job_id: str, spec: FontGenerationSpec, input_font: str | Path, original_filename: str) -> dict[str, Any]:
        self.prepare_spec(spec)
        _ensure_research_import_path()
        from eco_research_hangul.metrics import ink_saving, recognize_tesseract_multi

        input_font = Path(input_font)
        root = self.job_root(job_id)
        candidates_dir = root / "candidates"
        previews_dir = root / "previews"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        previews_dir.mkdir(parents=True, exist_ok=True)

        chars, missing_chars = chars_for_codepoint_set(spec.script, spec.codepoint_set, input_font)
        recipes = _style_recipes()[: spec.candidate_count]
        source_images: dict[str, np.ndarray] = {}
        for ch in chars:
            image = _render_glyph(input_font, ch, spec.image_size, spec.font_size)
            if _visible(image):
                source_images[ch] = image
        visible_chars = [ch for ch in chars if ch in source_images]
        if not visible_chars:
            raise ValueError("uploaded font has no visible glyphs for the requested codepoint set")

        eval_chars = _sample_chars(
            visible_chars,
            "none" if not spec.evaluation.ocr and not spec.evaluation.ink else spec.evaluation.ocr_eval_mode,
            spec.evaluation.eval_sample_size,
        )
        ocr_lang = spec.evaluation.ocr_lang or ("chr" if spec.script == "cherokee" else "kor")
        raw_psm = spec.evaluation.ocr_psm
        psms = raw_psm if isinstance(raw_psm, list) else [int(raw_psm)]
        preview_text = spec.preview_text or "".join(visible_chars[: min(24, len(visible_chars))])

        candidate_rows: list[dict[str, Any]] = []
        for index, recipe in enumerate(recipes):
            token = _safe_token(recipe.style_id)
            ttf_path = candidates_dir / f"candidate_{index:02d}_{token}.ttf"
            preview_path = previews_dir / f"candidate_{index:02d}_{token}.png"
            bitmaps: dict[str, np.ndarray] = {}
            ink_values: list[float] = []
            ocr_exact: list[float] = []
            ocr_match: list[float] = []
            ocr_conf: list[float | None] = []
            ocr_available: bool | None = None

            for ch in visible_chars:
                generated = recipe.apply(source_images[ch], ord(ch))
                generated = np.clip(generated, 0.0, 1.0).astype(np.float32)
                if not _visible(generated):
                    generated = source_images[ch]
                bitmaps[ch] = generated

                if ch in eval_chars:
                    if spec.evaluation.ink:
                        ink_values.append(float(ink_saving(source_images[ch], generated)))
                    if spec.evaluation.ocr and spec.evaluation.ocr_eval_mode != "none":
                        ocr = recognize_tesseract_multi(generated, expected_char=ch, lang=ocr_lang, psms=psms)
                        ocr_available = bool(ocr.get("available"))
                        ocr_exact.append(1.0 if ocr.get("exact_match") is True else 0.0)
                        ocr_match.append(1.0 if ocr.get("match") is True else 0.0)
                        ocr_conf.append(ocr.get("confidence"))

            family = f"Eco {spec.script.title()} {recipe.style_id.replace('_', ' ').title()} {job_id[:8]}"
            postscript = f"Eco{spec.script.title()}{index:02d}{job_id[:8]}-Regular"
            output_ttf, replaced, skipped_chars = export_ttf_from_bitmaps(
                input_font,
                bitmaps,
                ttf_path,
                family_name=family,
                postscript_name=postscript,
            )
            self._render_preview(output_ttf, preview_path, preview_text, recipe.style_id)

            mean_exact = _mean(ocr_exact)
            mean_match = _mean(ocr_match)
            mean_conf = _mean(ocr_conf)
            if mean_exact is None and mean_match is None:
                mean_ocr_score = None
            else:
                mean_ocr_score = float(max(mean_exact or 0.0, 0.5 * (mean_match or 0.0)))

            candidate_rows.append(
                {
                    "candidate_id": f"candidate_{index:02d}",
                    "style_id": recipe.style_id,
                    "description": recipe.description,
                    "ttf_url": _asset_url(job_id, Path("font_generation") / output_ttf.relative_to(root)),
                    "preview_url": _asset_url(job_id, Path("font_generation") / preview_path.relative_to(root)),
                    "file_size_bytes": output_ttf.stat().st_size,
                    "coverage": {
                        "requested_glyphs": len(chars),
                        "visible_source_glyphs": len(visible_chars),
                        "replaced_glyphs": int(replaced),
                        "skipped_glyphs": len(skipped_chars),
                        "skipped_chars": skipped_chars[:64],
                    },
                    "metrics": {
                        "eval_glyphs": len(eval_chars),
                        "mean_ink_saving": _mean(ink_values),
                        "mean_ocr_score": mean_ocr_score,
                        "mean_ocr_exact_match": mean_exact,
                        "mean_ocr_match": mean_match,
                        "mean_ocr_confidence": mean_conf,
                        "ocr_available": ocr_available,
                        "ocr_lang": ocr_lang if spec.evaluation.ocr else None,
                        "style_score": None,
                    },
                }
            )

        manifest = {
            "job_id": job_id,
            "status": "completed",
            "script": spec.script,
            "method": spec.method,
            "generation_mode": "uploaded_ttf_style_recipe_batch",
            "diffusion_used": False,
            "input_filename": original_filename,
            "codepoint_set": spec.codepoint_set,
            "coverage": {
                "requested_glyphs": len(chars) + len(missing_chars),
                "covered_glyphs": len(chars),
                "visible_source_glyphs": len(visible_chars),
                "missing_glyphs": len(missing_chars),
                "missing_codepoints": [f"U+{ord(ch):04X}" for ch in missing_chars],
            },
            "evaluation": self._dump_model(spec.evaluation),
            "outputs": {},
            "candidates": candidate_rows,
        }
        zip_path = root / "cherokee_candidates.zip"

        manifest["outputs"] = {
            "zip_url": _asset_url(job_id, Path("font_generation") / zip_path.relative_to(root)),
            "manifest_url": _asset_url(job_id, "font_generation/manifest.json"),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_zip(zip_path, root, manifest_path, candidate_rows)
        result_path = root / "api_result.json"
        result_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def _render_preview(self, font_path: Path, output_path: Path, text: str, style_id: str) -> None:
        font = ImageFont.truetype(str(font_path), 56)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        width = max(900, min(1800, 48 * max(8, len(text)) + 40))
        image = Image.new("RGB", (width, 180), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 18), style_id, fill="black", font=label_font)
        draw.text((20, 72), text, fill="black", font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    def _write_zip(self, zip_path: Path, root: Path, manifest_path: Path, candidate_rows: list[dict[str, Any]]) -> None:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, manifest_path.relative_to(root))
            for row in candidate_rows:
                ttf_rel = Path(row["ttf_url"].split("/font_generation/", 1)[1])
                preview_rel = Path(row["preview_url"].split("/font_generation/", 1)[1])
                archive.write(root / ttf_rel, ttf_rel)
                archive.write(root / preview_rel, preview_rel)

    @staticmethod
    def _dump_model(model: Any) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()


SUPPORTED_FONT_GENERATION_METHODS = [
    {
        "name": "eco_research_guided",
        "supported_scripts": ["cherokee"],
        "supported_codepoint_sets": ["cherokee_full", "uploaded_cherokee"],
        "max_candidate_count": 20,
        "output_formats": ["zip"],
        "description": "Uploaded TTF to candidate complete Cherokee TTFs using eco_research_hangul-style ink-saving recipes.",
    }
]
