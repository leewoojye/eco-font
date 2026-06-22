from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import BuildConfig, build_dataset, chars_from_args, parse_fonts
from .infer import chars_from_optional_file, run_inference
from .ocr import OCRConfig, train_ocr
from .priors import ECO_STYLES
from .train import train_cf_font


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _styles(value: str) -> tuple[str, ...]:
    styles = tuple(part.strip() for part in value.split(",") if part.strip())
    for style in styles:
        if style not in ECO_STYLES:
            raise argparse.ArgumentTypeError(f"Unknown style {style}. Valid: {','.join(ECO_STYLES)}")
    return styles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cf-font-eco")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset")
    build.add_argument("--fonts", default=None, help="Comma-separated font file list")
    build.add_argument("--fonts-glob", default=None, help="Glob for font files")
    build.add_argument("--chars", default=None)
    build.add_argument("--chars-file", default=None, type=Path)
    build.add_argument("--out-dir", required=True, type=Path)
    build.add_argument("--styles", default=",".join(ECO_STYLES), type=_styles)
    build.add_argument("--target-savings", default="0.45,0.60", type=_floats)
    build.add_argument("--image-size", default=96, type=int)
    build.add_argument("--font-size", default=None, type=int)
    build.add_argument("--limit-chars", default=None, type=int)
    build.add_argument("--ref-count", default=8, type=int)

    train = sub.add_parser("train")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--base-epochs", default=2, type=int)
    train.add_argument("--cf-epochs", default=2, type=int)
    train.add_argument("--batch-size", default=32, type=int)
    train.add_argument("--learning-rate", default=1e-3, type=float)
    train.add_argument("--base-channels", default=24, type=int)
    train.add_argument("--style-dim", default=96, type=int)
    train.add_argument("--basis-count", default=4, type=int)
    train.add_argument("--cfm-temperature", default=0.18, type=float)
    train.add_argument("--pcl-weight", default=1.0, type=float)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", default=23, type=int)
    train.add_argument("--ref-count", default=None, type=int)

    ocr = sub.add_parser("train-ocr")
    ocr.add_argument("--dataset", required=True, type=Path)
    ocr.add_argument("--out", required=True, type=Path)
    ocr.add_argument("--samples-per-char", default=32, type=int)
    ocr.add_argument("--epochs", default=4, type=int)
    ocr.add_argument("--batch-size", default=64, type=int)
    ocr.add_argument("--learning-rate", default=1e-3, type=float)
    ocr.add_argument("--device", default="auto")
    ocr.add_argument("--seed", default=29, type=int)

    infer = sub.add_parser("infer")
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--font", required=True, type=Path)
    infer.add_argument("--chars", default=None)
    infer.add_argument("--chars-file", default=None, type=Path)
    infer.add_argument("--out-dir", required=True, type=Path)
    infer.add_argument("--style", default="auto")
    infer.add_argument("--target-saving", default=0.60, type=float)
    infer.add_argument("--ocr-checkpoint", default=None, type=Path)
    infer.add_argument("--image-size", default=None, type=int)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--ocr-threshold", default=0.70, type=float)
    infer.add_argument("--isr-steps", default=0, type=int)
    infer.add_argument("--save-candidates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-dataset":
        summary = build_dataset(
            BuildConfig(
                fonts=parse_fonts(args.fonts, args.fonts_glob),
                chars=chars_from_args(args.chars, args.chars_file),
                out_dir=args.out_dir,
                styles=args.styles,
                target_savings=args.target_savings,
                image_size=args.image_size,
                font_size=args.font_size,
                limit_chars=args.limit_chars,
                ref_count=args.ref_count,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "train":
        summary = train_cf_font(
            dataset=args.dataset,
            out=args.out,
            base_epochs=args.base_epochs,
            cf_epochs=args.cf_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            base_channels=args.base_channels,
            style_dim=args.style_dim,
            basis_count=args.basis_count,
            cfm_temperature=args.cfm_temperature,
            pcl_weight=args.pcl_weight,
            device_name=args.device,
            seed=args.seed,
            ref_count=args.ref_count,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "train-ocr":
        summary = train_ocr(
            OCRConfig(
                dataset=args.dataset,
                out=args.out,
                samples_per_char=args.samples_per_char,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                seed=args.seed,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "infer":
        summary = run_inference(
            checkpoint=args.checkpoint,
            font=args.font,
            chars=chars_from_optional_file(args.chars, args.chars_file),
            out_dir=args.out_dir,
            style=args.style,
            target_saving=args.target_saving,
            ocr_checkpoint=args.ocr_checkpoint,
            image_size=args.image_size,
            device_name=args.device,
            ocr_threshold=args.ocr_threshold,
            isr_steps=args.isr_steps,
            save_candidates=args.save_candidates,
        )
        print(json.dumps(summary["average_metrics"], ensure_ascii=False, indent=2))
        return
    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
