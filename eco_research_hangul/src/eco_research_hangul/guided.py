from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from tqdm import tqdm

from .config import ensure_dir, load_yaml, read_chars
from .infer import _device, load_checkpoint, sample_one
from .metrics import evaluate_sample, ink_saving
from .render import has_visible_glyph, render_glyph, save_gray


@dataclass(frozen=True)
class Candidate:
    name: str
    image: np.ndarray


def _binary(image: np.ndarray, threshold: float = 0.12) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) > threshold).astype(np.uint8)


def _soften(mask: np.ndarray, sigma: float = 0.55) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def erode_ink(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = _binary(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=max(1, int(iterations)))
    return _soften(eroded)


def close_perforations(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = _binary(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(iterations)))
    return _soften(closed)


def inline_engrave(
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
    skel = skeletonize(mask_for_skeleton.astype(bool))
    skel = np.logical_and(skel, dist >= float(min_distance)).astype(np.uint8)
    if int(line_width) > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_width), int(line_width)))
        skel = cv2.dilate(skel, kernel, iterations=1)
    out = np.clip(image, 0.0, 1.0).astype(np.float32).copy()
    out[skel > 0] *= max(0.0, 1.0 - float(strength))
    return _soften(out)


def make_candidates(source: np.ndarray, diffusion_images: list[np.ndarray]) -> list[Candidate]:
    candidates: list[Candidate] = []
    bases = [("source", source)]
    for idx, image in enumerate(diffusion_images):
        bases.append((f"diffusion_{idx}", image))
        bases.append((f"diffusion_{idx}_closed", close_perforations(image, iterations=1)))

    seen: set[bytes] = set()
    for name, base in bases:
        variants = [
            Candidate(f"{name}_original", np.clip(base, 0.0, 1.0).astype(np.float32)),
            Candidate(f"{name}_erode1", erode_ink(base, iterations=1)),
            Candidate(f"{name}_erode2", erode_ink(base, iterations=2)),
            Candidate(f"{name}_inline_soft", inline_engrave(base, line_width=1, strength=0.65, min_distance=1.15)),
            Candidate(f"{name}_inline_w1", inline_engrave(base, line_width=1, strength=0.92, min_distance=1.15)),
            Candidate(f"{name}_inline_w2", inline_engrave(base, line_width=2, strength=0.95, min_distance=1.5)),
            Candidate(f"{name}_erode1_inline_soft", inline_engrave(erode_ink(base, iterations=1), line_width=1, strength=0.65, min_distance=1.1)),
            Candidate(f"{name}_inline_erode_w1", inline_engrave(erode_ink(base, iterations=1), line_width=1, strength=0.92, min_distance=1.1)),
            Candidate(f"{name}_inline_erode_w2", inline_engrave(erode_ink(base, iterations=1), line_width=2, strength=0.95, min_distance=1.4)),
        ]
        for candidate in variants:
            key = (candidate.image > 0.12).astype(np.uint8).tobytes()
            if key not in seen and has_visible_glyph(candidate.image):
                seen.add(key)
                candidates.append(candidate)
    return candidates


def perforation_features(image: np.ndarray) -> dict[str, float]:
    mask = _binary(image)
    glyph_area = float(mask.sum())
    if glyph_area <= 0:
        return {
            "small_hole_count": 0.0,
            "small_hole_area_ratio": 0.0,
            "channel_area_ratio": 0.0,
            "small_fragment_count": 0.0,
            "small_fragment_area_ratio": 0.0,
        }
    background = (1 - mask).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    small_holes = 0
    small_area = 0.0
    channel_area = 0.0
    h, w = mask.shape
    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches_border:
            continue
        short = max(1, min(int(bw), int(bh)))
        aspect = max(int(bw), int(bh)) / short
        if area <= 80 and aspect < 3.0:
            small_holes += 1
            small_area += float(area)
        else:
            channel_area += float(area)
    fg_labels, _, fg_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small_fragments = 0
    small_fragment_area = 0.0
    for label in range(1, fg_labels):
        area = int(fg_stats[label, cv2.CC_STAT_AREA])
        if area <= 30:
            small_fragments += 1
            small_fragment_area += float(area)
    return {
        "small_hole_count": float(small_holes),
        "small_hole_area_ratio": float(small_area / glyph_area),
        "channel_area_ratio": float(channel_area / glyph_area),
        "small_fragment_count": float(small_fragments),
        "small_fragment_area_ratio": float(small_fragment_area / glyph_area),
    }


def render_style_reference(
    font_path: str | Path,
    output_path: str | Path,
    text: str,
    image_size: int = 96,
    font_size: int = 76,
) -> np.ndarray:
    glyphs = []
    for ch in text:
        if ch.isspace():
            continue
        rendered = render_glyph(font_path, ch, image_size=image_size, font_size=font_size)
        if has_visible_glyph(rendered.image):
            glyphs.append(rendered.image)
    if not glyphs:
        raise ValueError(f"No visible style glyphs rendered from {font_path}")
    cols = min(8, max(1, int(np.ceil(np.sqrt(len(glyphs))))))
    rows = int(np.ceil(len(glyphs) / cols))
    sheet = Image.new("L", (cols * image_size, rows * image_size), 0)
    for idx, glyph in enumerate(glyphs):
        arr = np.clip(glyph * 255.0, 0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(arr, mode="L"), ((idx % cols) * image_size, (idx // cols) * image_size))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return np.asarray(sheet, dtype=np.float32) / 255.0


class VGGStyleScorer:
    def __init__(self, reference: np.ndarray, device: torch.device, enabled: bool = True) -> None:
        self.enabled = enabled
        self.device = device
        self.layers = (3, 8, 17)
        self.model = None
        self.reference_grams: list[torch.Tensor] = []
        if not enabled:
            return
        try:
            from torchvision.models import VGG19_Weights, vgg19

            weights = VGG19_Weights.IMAGENET1K_V1
            model = vgg19(weights=weights).features[: max(self.layers) + 1].to(device).eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.model = model
            with torch.no_grad():
                self.reference_grams = self._grams(self._to_tensor(reference))
        except Exception as exc:
            print(f"warning: VGG style scorer disabled: {exc}")
            self.enabled = False

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        ink = np.clip(image, 0.0, 1.0).astype(np.float32)
        pil = Image.fromarray(((1.0 - ink) * 255.0).astype(np.uint8), mode="L")
        pil = pil.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def _gram(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        features = x.reshape(b, c, h * w)
        return torch.bmm(features, features.transpose(1, 2)) / float(c * h * w)

    def _grams(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if self.model is None:
            return []
        grams = []
        x = tensor
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.layers:
                grams.append(self._gram(x))
        return grams

    def loss(self, image: np.ndarray) -> float:
        if not self.enabled or self.model is None:
            return 0.0
        with torch.no_grad():
            grams = self._grams(self._to_tensor(image))
            losses = [torch.mean((cur - ref) ** 2) for cur, ref in zip(grams, self.reference_grams)]
            return float(torch.stack(losses).mean().detach().cpu())


def _style_loss_to_score(losses: list[float]) -> list[float]:
    if not losses:
        return []
    lo = min(losses)
    hi = max(losses)
    if hi - lo < 1e-12:
        return [1.0 for _ in losses]
    return [float(1.0 - (loss - lo) / (hi - lo)) for loss in losses]


def score_candidates(
    source: np.ndarray,
    candidates: list[Candidate],
    expected_char: str,
    ocr_lang: str,
    ocr_psm: int | Sequence[int],
    style_scorer: VGGStyleScorer,
    target_saving: float,
    min_ocr_confidence: float,
    weights: dict,
    max_small_holes: int,
    max_small_fragments: int,
    oversave_margin: float,
) -> tuple[Candidate, list[dict]]:
    rows = []
    style_losses = [style_scorer.loss(candidate.image) for candidate in candidates]
    style_scores = _style_loss_to_score(style_losses)
    for candidate, style_loss, style_score in zip(candidates, style_losses, style_scores):
        metrics = evaluate_sample(
            source,
            candidate.image,
            None,
            expected_char=expected_char,
            ocr_lang=ocr_lang,
            ocr_psm=ocr_psm,
        )
        saving = float(metrics["ink_saving"])
        ocr_conf = float(metrics["tesseract_confidence"] or 0.0)
        exact = metrics["tesseract_exact_match"] is True
        match = metrics["tesseract_match"] is True
        ocr_score = (1.0 if exact else 0.45 if match else 0.0) + 0.25 * min(1.0, ocr_conf / 100.0)
        ink_score = min(max(saving, 0.0), target_saving) / max(target_saving, 1e-6)
        over_save_penalty = max(0.0, saving - (target_saving + oversave_margin))
        perf = perforation_features(candidate.image)
        small_holes = int(perf["small_hole_count"])
        small_fragments = int(perf["small_fragment_count"])
        hole_penalty = small_holes / 20.0 + perf["small_hole_area_ratio"]
        fragment_penalty = small_fragments / 12.0 + perf["small_fragment_area_ratio"]
        channel_score = min(1.0, perf["channel_area_ratio"] * 3.0)
        objective = (
            float(weights.get("ocr", 2.5)) * ocr_score
            + float(weights.get("ink", 2.0)) * ink_score
            + float(weights.get("style", 1.0)) * style_score
            + float(weights.get("channel", 0.8)) * channel_score
            - float(weights.get("perforation", 1.8)) * hole_penalty
            - float(weights.get("fragmentation", 4.0)) * fragment_penalty
            - float(weights.get("oversave", 1.0)) * over_save_penalty
        )
        metrics["objective"] = {
            "score": float(objective),
            "candidate": candidate.name,
            "target_saving": float(target_saving),
            "ocr_score": float(ocr_score),
            "ink_score": float(ink_score),
            "style_loss": float(style_loss),
            "style_score": float(style_score),
            "channel_score": float(channel_score),
            "hole_penalty": float(hole_penalty),
            "fragment_penalty": float(fragment_penalty),
            "small_hole_count": int(perf["small_hole_count"]),
            "small_hole_area_ratio": float(perf["small_hole_area_ratio"]),
            "channel_area_ratio": float(perf["channel_area_ratio"]),
            "small_fragment_count": int(perf["small_fragment_count"]),
            "small_fragment_area_ratio": float(perf["small_fragment_area_ratio"]),
            "passes_ocr_gate": bool(exact and ocr_conf >= min_ocr_confidence),
            "passes_design_gate": bool(
                small_holes <= int(max_small_holes) and small_fragments <= int(max_small_fragments)
            ),
        }
        rows.append({"candidate": candidate, "metrics": metrics})
    gated = [
        row
        for row in rows
        if row["metrics"]["objective"]["passes_ocr_gate"] and row["metrics"]["objective"]["passes_design_gate"]
    ]
    pool = gated or [row for row in rows if row["metrics"]["objective"]["passes_ocr_gate"]] or rows
    best_row = max(pool, key=lambda item: item["metrics"]["objective"]["score"])
    return best_row["candidate"], [{"name": row["candidate"].name, "metrics": row["metrics"]} for row in rows]


def run_guided_inference_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    base = config_path.parent.parent
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    inf_cfg = cfg["guided_inference"]
    out_dir = ensure_dir(base / inf_cfg["output_dir"])
    source_dir = ensure_dir(out_dir / "source")
    generated_dir = ensure_dir(out_dir / "generated")
    candidate_dir = ensure_dir(out_dir / "candidates")
    checkpoint = base / inf_cfg["checkpoint"]
    font = Path(inf_cfg["font"])
    chars = read_chars(chars=inf_cfg.get("chars"), charset_file=base / inf_cfg["charset_file"] if inf_cfg.get("charset_file") else None)
    image_size = int(data_cfg.get("image_size", 96))
    font_size = int(data_cfg.get("font_size", 76))
    device = _device(str(inf_cfg.get("device", "auto")))
    model, schedule, ckpt = load_checkpoint(checkpoint, device)
    prediction_type = str(ckpt.get("diffusion_config", {}).get("prediction_type", "epsilon"))
    sample_steps = int(inf_cfg["sample_steps"]) if inf_cfg.get("sample_steps") is not None else None
    ocr_lang = str(inf_cfg.get("ocr_lang", "kor"))
    raw_ocr_psm = inf_cfg.get("ocr_psm", 8)
    if isinstance(raw_ocr_psm, list):
        ocr_psm = [int(item) for item in raw_ocr_psm]
    else:
        ocr_psm = int(raw_ocr_psm)
    target_saving = float(inf_cfg.get("target_saving", 0.45))
    diffusion_candidates = int(inf_cfg.get("diffusion_candidates", 2))
    min_ocr_confidence = float(inf_cfg.get("min_ocr_confidence", 45.0))
    max_small_holes = int(inf_cfg.get("max_small_holes", 0))
    max_small_fragments = int(inf_cfg.get("max_small_fragments", 2))
    oversave_margin = float(inf_cfg.get("oversave_margin", 0.05))
    weights = dict(inf_cfg.get("weights", {}))
    style_font = base / inf_cfg["style_reference_font"]
    style_text = str(inf_cfg.get("style_reference_text", "RYMANECOBeautifulSustainable1234567890"))
    style_reference_path = out_dir / "style_reference.png"
    style_reference = render_style_reference(style_font, style_reference_path, style_text, image_size=image_size, font_size=font_size)
    style_scorer = VGGStyleScorer(style_reference, device=device, enabled=bool(inf_cfg.get("use_vgg_style", True)))
    manifest = out_dir / "inference_manifest.jsonl"
    candidate_manifest = out_dir / "candidate_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f, candidate_manifest.open("w", encoding="utf-8") as cf:
        for ch in tqdm(chars, desc="guided-infer"):
            source = render_glyph(font, ch, image_size=image_size, font_size=font_size)
            if not has_visible_glyph(source.image):
                continue
            diffusion_images = [
                sample_one(model, schedule, source.image, target_saving, device, sample_steps, prediction_type)
                for _ in range(diffusion_candidates)
            ]
            candidates = make_candidates(source.image, diffusion_images)
            best, candidate_rows = score_candidates(
                source.image,
                candidates,
                expected_char=ch,
                ocr_lang=ocr_lang,
                ocr_psm=ocr_psm,
                style_scorer=style_scorer,
                target_saving=target_saving,
                min_ocr_confidence=min_ocr_confidence,
                weights=weights,
                max_small_holes=max_small_holes,
                max_small_fragments=max_small_fragments,
                oversave_margin=oversave_margin,
            )
            char_id = f"u{ord(ch):04x}"
            save_gray(source_dir / f"{char_id}.png", source.image)
            save_gray(generated_dir / f"{char_id}.png", best.image)
            best_metrics = next(row["metrics"] for row in candidate_rows if row["name"] == best.name)
            row = {
                "mode": "ocr_ink_style_guided",
                "font": str(font),
                "checkpoint": str(checkpoint),
                "style_reference_font": str(style_font),
                "style_reference": str(style_reference_path.relative_to(out_dir)),
                "char": ch,
                "char_id": char_id,
                "target_saving": float(target_saving),
                "source": str(Path("source") / f"{char_id}.png"),
                "generated": str(Path("generated") / f"{char_id}.png"),
                "target": None,
                "metrics": best_metrics,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for idx, candidate_row in enumerate(candidate_rows):
                candidate_name = candidate_row["name"]
                if idx < int(inf_cfg.get("save_top_candidates", 0)):
                    path = candidate_dir / f"{char_id}_{idx:02d}_{candidate_name}.png"
                    image = next(c.image for c in candidates if c.name == candidate_name)
                    save_gray(path, image)
                    candidate_row["path"] = str(path.relative_to(out_dir))
                cf.write(json.dumps({"char": ch, "char_id": char_id, **candidate_row}, ensure_ascii=False) + "\n")
    print(f"manifest={manifest}")
    print(f"candidate_manifest={candidate_manifest}")
    return manifest
