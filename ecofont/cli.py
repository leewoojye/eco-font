"""Command line interface for EcoFont AI Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DatasetBuildConfig, build_dataset
from .font_io import inspect_font
from .infer import infer_font
from .ocr_surrogate import OCRTrainConfig, train_ocr_surrogate
from .rules import RuleWeights
from .text_presets import characters_for_language
from .train import train_model


def _parse_targets(value: str) -> tuple[float, ...]:
    targets = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not targets:
        raise argparse.ArgumentTypeError("targets must contain at least one float")
    for target in targets:
        if not 0.0 < target < 0.8:
            raise argparse.ArgumentTypeError("each target must be between 0 and 0.8")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecofont", description="Train and run Glyph-to-EcoMask models.")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect-font", help="Check font coverage for a language preset.")
    inspect_cmd.add_argument("--font", required=True, type=Path)
    inspect_cmd.add_argument("--language", default="ko", choices=["ko", "chr", "mixed"])
    inspect_cmd.add_argument("--text", default=None)

    data_cmd = sub.add_parser("build-dataset", help="Create rule-optimized pseudo-label samples.")
    data_cmd.add_argument("--font", action="append", required=True, type=Path, help="TTF/OTF font path. Repeatable.")
    data_cmd.add_argument("--language", default="ko", choices=["ko", "chr", "mixed"])
    data_cmd.add_argument("--text", default=None)
    data_cmd.add_argument("--targets", default=(0.15, 0.25, 0.35), type=_parse_targets)
    data_cmd.add_argument("--output", required=True, type=Path)
    data_cmd.add_argument("--image-size", default=128, type=int)
    data_cmd.add_argument("--max-chars", default=None, type=int)
    data_cmd.add_argument("--candidate-limit", default=None, type=int)
    data_cmd.add_argument("--readability-weight", default=2.0, type=float)
    data_cmd.add_argument("--ink-weight", default=0.4, type=float)
    data_cmd.add_argument("--target-weight", default=2.5, type=float)
    data_cmd.add_argument("--topology-weight", default=3.0, type=float)

    train_cmd = sub.add_parser("train", help="Train the U-Net mask generator.")
    train_cmd.add_argument("--dataset", required=True, type=Path)
    train_cmd.add_argument("--output", required=True, type=Path)
    train_cmd.add_argument("--epochs", default=10, type=int)
    train_cmd.add_argument("--batch-size", default=16, type=int)
    train_cmd.add_argument("--learning-rate", default=1e-3, type=float)
    train_cmd.add_argument("--val-split", default=0.15, type=float)
    train_cmd.add_argument("--device", default="auto")
    train_cmd.add_argument("--base-channels", default=32, type=int)
    train_cmd.add_argument("--seed", default=42, type=int)

    ocr_cmd = sub.add_parser("train-ocr", help="Train a local OCR-surrogate glyph recognizer.")
    ocr_cmd.add_argument("--font", required=True, type=Path)
    ocr_cmd.add_argument("--output", required=True, type=Path)
    ocr_cmd.add_argument("--language", default="chr", choices=["ko", "chr", "mixed"])
    ocr_cmd.add_argument("--text", default=None)
    ocr_cmd.add_argument("--image-size", default=96, type=int)
    ocr_cmd.add_argument("--samples-per-char", default=32, type=int)
    ocr_cmd.add_argument("--epochs", default=8, type=int)
    ocr_cmd.add_argument("--batch-size", default=64, type=int)
    ocr_cmd.add_argument("--learning-rate", default=1e-3, type=float)
    ocr_cmd.add_argument("--device", default="auto")
    ocr_cmd.add_argument("--seed", default=7, type=int)

    infer_cmd = sub.add_parser("infer", help="Run model or rule inference on a font.")
    infer_cmd.add_argument("--font", required=True, type=Path)
    infer_cmd.add_argument("--output", required=True, type=Path)
    infer_cmd.add_argument("--checkpoint", default=None, type=Path)
    infer_cmd.add_argument("--ocr-checkpoint", default=None, type=Path)
    infer_cmd.add_argument("--method", default="model", choices=["model", "rules", "ocr-rules"])
    infer_cmd.add_argument("--language", default="ko", choices=["ko", "chr", "mixed"])
    infer_cmd.add_argument("--text", default=None)
    infer_cmd.add_argument("--target-saving", default=0.25, type=float)
    infer_cmd.add_argument("--image-size", default=128, type=int)
    infer_cmd.add_argument("--threshold", default=0.5, type=float)
    infer_cmd.add_argument("--device", default="auto")
    infer_cmd.add_argument("--max-chars", default=None, type=int)
    infer_cmd.add_argument("--candidate-limit", default=None, type=int)
    infer_cmd.add_argument("--ocr-weight", default=1.0, type=float)
    infer_cmd.add_argument("--ocr-target-weight", default=3.0, type=float)
    infer_cmd.add_argument("--ocr-ink-weight", default=0.2, type=float)
    infer_cmd.add_argument("--outline-reward-weight", default=0.35, type=float)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-font":
        chars = characters_for_language(args.language, args.text)
        report = inspect_font(args.font, chars)
        print(
            json.dumps(
                {
                    "font": str(report.path),
                    "total_codepoints": report.total_codepoints,
                    "requested_count": report.requested_count,
                    "supported_count": report.supported_count,
                    "missing_count": report.missing_count,
                    "supported_preview": "".join(report.supported_chars[:50]),
                    "missing_preview": "".join(report.missing_chars[:50]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "build-dataset":
        weights = RuleWeights(
            readability_weight=args.readability_weight,
            ink_weight=args.ink_weight,
            target_weight=args.target_weight,
            topology_weight=args.topology_weight,
        )
        summary = build_dataset(
            DatasetBuildConfig(
                fonts=args.font,
                output=args.output,
                language=args.language,
                text=args.text,
                targets=args.targets,
                image_size=args.image_size,
                max_chars=args.max_chars,
                candidate_limit=args.candidate_limit,
                weights=weights,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return

    if args.command == "train":
        summary = train_model(
            dataset_dir=args.dataset,
            output=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            device=args.device,
            base_channels=args.base_channels,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "train-ocr":
        summary = train_ocr_surrogate(
            OCRTrainConfig(
                font=args.font,
                output=args.output,
                language=args.language,
                text=args.text,
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
        summary = infer_font(
            font_path=args.font,
            output=args.output,
            checkpoint=args.checkpoint,
            ocr_checkpoint=args.ocr_checkpoint,
            method=args.method,
            language=args.language,
            text=args.text,
            target_saving=args.target_saving,
            image_size=args.image_size,
            threshold=args.threshold,
            device=args.device,
            max_chars=args.max_chars,
            candidate_limit=args.candidate_limit,
            ocr_weight=args.ocr_weight,
            ocr_target_weight=args.ocr_target_weight,
            ocr_ink_weight=args.ocr_ink_weight,
            outline_reward_weight=args.outline_reward_weight,
        )
        print(json.dumps(summary["average_metrics"], ensure_ascii=False, indent=2))
        return

    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
