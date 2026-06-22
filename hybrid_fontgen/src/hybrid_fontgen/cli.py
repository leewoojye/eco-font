from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import BuildConfig, build_dataset
from .infer import run_inference
from .ocr import OCRConfig, train_ocr
from .priors import STYLES
from .train import train_generator


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _styles(value: str) -> tuple[str, ...]:
    styles = tuple(part.strip() for part in value.split(",") if part.strip())
    for style in styles:
        if style not in STYLES:
            raise argparse.ArgumentTypeError(f"Unknown style {style}. Valid: {','.join(STYLES)}")
    return styles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hybrid-fontgen")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset")
    build.add_argument("--font", required=True, type=Path)
    build.add_argument("--chars", required=True)
    build.add_argument("--out-dir", required=True, type=Path)
    build.add_argument("--styles", default=",".join(STYLES), type=_styles)
    build.add_argument("--target-savings", default="0.45,0.60", type=_floats)
    build.add_argument("--image-size", default=96, type=int)
    build.add_argument("--font-size", default=None, type=int)

    train = sub.add_parser("train")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--epochs", default=8, type=int)
    train.add_argument("--batch-size", default=16, type=int)
    train.add_argument("--learning-rate", default=1e-3, type=float)
    train.add_argument("--base-channels", default=24, type=int)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", default=11, type=int)

    ocr = sub.add_parser("train-ocr")
    ocr.add_argument("--font", required=True, type=Path)
    ocr.add_argument("--chars", required=True)
    ocr.add_argument("--out", required=True, type=Path)
    ocr.add_argument("--image-size", default=96, type=int)
    ocr.add_argument("--samples-per-char", default=64, type=int)
    ocr.add_argument("--epochs", default=8, type=int)
    ocr.add_argument("--batch-size", default=48, type=int)
    ocr.add_argument("--learning-rate", default=1e-3, type=float)
    ocr.add_argument("--device", default="auto")
    ocr.add_argument("--seed", default=19, type=int)

    infer = sub.add_parser("infer")
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--font", required=True, type=Path)
    infer.add_argument("--chars", required=True)
    infer.add_argument("--out-dir", required=True, type=Path)
    infer.add_argument("--style", default="auto")
    infer.add_argument("--target-saving", default=0.60, type=float)
    infer.add_argument("--ocr-checkpoint", default=None, type=Path)
    infer.add_argument("--image-size", default=96, type=int)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--export-ttf", default=None, type=Path)
    infer.add_argument("--ocr-threshold", default=0.72, type=float)
    infer.add_argument("--void-style-weight", default=0.90, type=float)
    infer.add_argument("--save-candidates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-dataset":
        summary = build_dataset(
            BuildConfig(
                font=args.font,
                chars=args.chars,
                out_dir=args.out_dir,
                styles=args.styles,
                target_savings=args.target_savings,
                image_size=args.image_size,
                font_size=args.font_size,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return
    if args.command == "train":
        summary = train_generator(
            dataset=args.dataset,
            out=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
            base_channels=args.base_channels,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "train-ocr":
        summary = train_ocr(
            OCRConfig(
                font=args.font,
                chars=args.chars,
                out=args.out,
                image_size=args.image_size,
                samples_per_char=args.samples_per_char,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                seed=args.seed,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "infer":
        summary = run_inference(
            checkpoint=args.checkpoint,
            font=args.font,
            chars=args.chars,
            out_dir=args.out_dir,
            style=args.style,
            target_saving=args.target_saving,
            ocr_checkpoint=args.ocr_checkpoint,
            image_size=args.image_size,
            device_name=args.device,
            export_ttf=args.export_ttf,
            ocr_threshold=args.ocr_threshold,
            void_style_weight=args.void_style_weight,
            save_candidates=args.save_candidates,
        )
        print(json.dumps(summary["average_metrics"], ensure_ascii=False, indent=2))
        return
    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
