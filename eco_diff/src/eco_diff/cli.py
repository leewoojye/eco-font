from __future__ import annotations

import argparse
from pathlib import Path

from .config import chars_from_args, parse_float_list, read_charset
from .dataset import build_dataset
from .infer import run_inference
from .sample_diffusion import sample_diffusion
from .train import train_from_config
from .train_diffusion import train_diffusion_from_config


def _add_charset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chars", default=None, help="Characters to process, e.g. ABCD")
    parser.add_argument("--charset-file", default=None, help="UTF-8 file containing characters")


def build_dataset_cmd(args: argparse.Namespace) -> None:
    chars = chars_from_args(args.chars, args.charset_file)
    summary = build_dataset(
        fonts=args.font or [],
        fonts_dir=args.fonts_dir,
        chars=chars,
        out_dir=args.out_dir,
        target_savings=parse_float_list(args.target_savings),
        image_size=args.image_size,
        font_size=args.font_size,
        padding=args.padding,
        max_records=args.max_records,
    )
    print(f"records={summary.records} skipped={summary.skipped}")
    print(f"manifest={summary.manifest}")


def train_cmd(args: argparse.Namespace) -> None:
    best = train_from_config(args.config)
    print(f"best_checkpoint={best}")


def train_diffusion_cmd(args: argparse.Namespace) -> None:
    best = train_diffusion_from_config(args.config)
    print(f"best_diffusion_checkpoint={best}")


def infer_cmd(args: argparse.Namespace) -> None:
    chars = chars_from_args(args.chars, args.charset_file)
    manifest = run_inference(
        checkpoint=args.checkpoint,
        font=args.font,
        chars=chars,
        out_dir=args.out_dir,
        target_saving=args.target_saving,
        image_size=args.image_size,
        font_size=args.font_size,
        threshold=args.threshold,
        force_saving=args.force_saving,
        device_name=args.device,
        export_ttf=args.export_ttf,
    )
    print(f"manifest={manifest}")


def sample_diffusion_cmd(args: argparse.Namespace) -> None:
    chars = chars_from_args(args.chars, args.charset_file)
    manifest = sample_diffusion(
        checkpoint=args.checkpoint,
        font=args.font,
        chars=chars,
        out_dir=args.out_dir,
        target_saving=args.target_saving,
        image_size=args.image_size,
        font_size=args.font_size,
        num_candidates=args.num_candidates,
        sample_steps=args.sample_steps,
        force_ink_budget=args.force_ink_budget,
        allow_outline_shift=args.allow_outline_shift,
        ocr_lang=args.ocr_lang,
        template_ocr=not args.no_template_ocr,
        device_name=args.device,
        export_ttf=args.export_ttf,
    )
    print(f"manifest={manifest}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eco-diff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="Render fonts and build rule-based eco-mask labels")
    build.add_argument("--font", action="append", help="TTF/OTF path. Can be passed multiple times.")
    build.add_argument("--fonts-dir", default=None, help="Directory recursively scanned for TTF/OTF/TTC")
    _add_charset_args(build)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--target-savings", default="0.15,0.25,0.35")
    build.add_argument("--image-size", type=int, default=96)
    build.add_argument("--font-size", type=int, default=76)
    build.add_argument("--padding", type=int, default=4)
    build.add_argument("--max-records", type=int, default=None)
    build.set_defaults(func=build_dataset_cmd)

    train = subparsers.add_parser("train", help="Train EcoMask U-Net")
    train.add_argument("--config", default="configs/default.yaml")
    train.set_defaults(func=train_cmd)

    train_diff = subparsers.add_parser("train-diffusion", help="Train conditional diffusion eco glyph generator")
    train_diff.add_argument("--config", default="configs/diffusion.yaml")
    train_diff.set_defaults(func=train_diffusion_cmd)

    infer = subparsers.add_parser("infer", help="Predict eco masks for a font")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--font", required=True)
    _add_charset_args(infer)
    infer.add_argument("--out-dir", required=True)
    infer.add_argument("--target-saving", type=float, default=0.25)
    infer.add_argument("--image-size", type=int, default=96)
    infer.add_argument("--font-size", type=int, default=76)
    infer.add_argument("--threshold", type=float, default=None)
    infer.add_argument(
        "--force-saving",
        action="store_true",
        help="Binarize predicted probabilities so cut ink area approximately matches --target-saving.",
    )
    infer.add_argument("--device", default="auto")
    infer.add_argument("--export-ttf", default=None)
    infer.set_defaults(func=infer_cmd)

    sample_diff = subparsers.add_parser("sample-diffusion", help="Generate diffusion eco glyph candidates and select the best")
    sample_diff.add_argument("--checkpoint", required=True)
    sample_diff.add_argument("--font", required=True)
    _add_charset_args(sample_diff)
    sample_diff.add_argument("--out-dir", required=True)
    sample_diff.add_argument("--target-saving", type=float, default=0.4)
    sample_diff.add_argument("--image-size", type=int, default=96)
    sample_diff.add_argument("--font-size", type=int, default=76)
    sample_diff.add_argument("--num-candidates", type=int, default=4)
    sample_diff.add_argument("--sample-steps", type=int, default=None)
    sample_diff.add_argument("--force-ink-budget", action="store_true", help="Project each candidate to the requested ink budget.")
    sample_diff.add_argument("--allow-outline-shift", type=int, default=2)
    sample_diff.add_argument("--ocr-lang", default=None, help="Optional Tesseract language code for OCR-based selection.")
    sample_diff.add_argument("--no-template-ocr", action="store_true", help="Disable built-in template OCR fallback.")
    sample_diff.add_argument("--device", default="auto")
    sample_diff.add_argument("--export-ttf", default=None)
    sample_diff.set_defaults(func=sample_diffusion_cmd)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
