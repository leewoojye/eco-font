from __future__ import annotations

import argparse

from .config import chars_from_args, parse_float_list
from .data import build_dataset
from .infer import run_inference
from .report import contact_sheet, summarize_outputs
from .train import train_from_config


def _charset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chars", default=None)
    parser.add_argument("--charset-file", default=None)


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
        max_records=args.max_records,
        target_style=args.target_style,
    )
    print(f"records={summary.records} skipped={summary.skipped}")
    print(f"manifest={summary.manifest}")


def train_cmd(args: argparse.Namespace) -> None:
    path = train_from_config(args.config)
    print(f"best_checkpoint={path}")


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
        device_name=args.device,
        export_font=args.export_ttf,
        force_budget=not args.no_force_budget,
        target_style=args.target_style,
        ocr_engine=args.ocr_engine,
        ocr_lang=args.ocr_lang,
        ocr_psm=args.ocr_psm,
    )
    print(f"manifest={manifest}")


def report_cmd(args: argparse.Namespace) -> None:
    chars = chars_from_args(args.chars, args.charset_file)
    folders = [item.strip() for item in args.folders.split(",") if item.strip()]
    labels = [item.strip() for item in args.labels.split(",") if item.strip()] if args.labels else folders
    summarize_outputs(args.root, folders)
    output = contact_sheet(args.root, folders, labels, chars, args.output, font_path=args.label_font)
    print(f"contact_sheet={output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ryman-font")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset")
    build.add_argument("--font", action="append")
    build.add_argument("--fonts-dir", default=None)
    _charset_args(build)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--target-savings", default="0.35,0.45,0.55")
    build.add_argument("--image-size", type=int, default=96)
    build.add_argument("--font-size", type=int, default=76)
    build.add_argument("--max-records", type=int, default=None)
    build.add_argument("--target-style", choices=["contour", "distinct", "canonical"], default="contour")
    build.set_defaults(func=build_dataset_cmd)

    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/hangul.yaml")
    train.set_defaults(func=train_cmd)

    infer = sub.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--font", required=True)
    _charset_args(infer)
    infer.add_argument("--out-dir", required=True)
    infer.add_argument("--target-saving", type=float, default=0.45)
    infer.add_argument("--image-size", type=int, default=96)
    infer.add_argument("--font-size", type=int, default=76)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--export-ttf", default=None)
    infer.add_argument("--no-force-budget", action="store_true")
    infer.add_argument("--target-style", choices=["contour", "distinct", "canonical"], default=None)
    infer.add_argument("--ocr-engine", choices=["tesseract", "template", "both", "none"], default="tesseract")
    infer.add_argument("--ocr-lang", default="kor")
    infer.add_argument("--ocr-psm", type=int, default=10)
    infer.set_defaults(func=infer_cmd)

    report = sub.add_parser("report")
    report.add_argument("--root", required=True)
    report.add_argument("--folders", required=True)
    report.add_argument("--labels", default=None)
    _charset_args(report)
    report.add_argument("--output", required=True)
    report.add_argument("--label-font", default=None)
    report.set_defaults(func=report_cmd)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
