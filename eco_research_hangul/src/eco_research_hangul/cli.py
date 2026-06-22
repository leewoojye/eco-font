from __future__ import annotations

import argparse

from .candidate_preview import generate_candidate_preview
from .data_build import build_dataset_from_config
from .guided import run_guided_inference_from_config
from .infer import run_inference_from_config
from .report import make_report
from .train import train_from_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eco-research-hangul")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset")
    build.add_argument("--config", default="configs/smoke.yaml")

    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/smoke.yaml")

    infer = sub.add_parser("infer")
    infer.add_argument("--config", default="configs/smoke.yaml")

    guided = sub.add_parser("guided-infer")
    guided.add_argument("--config", default="configs/guided.yaml")

    candidates = sub.add_parser("candidate-preview")
    candidates.add_argument("--config", default="configs/guided_jua.yaml")
    candidates.add_argument("--output-dir", default=None)
    candidates.add_argument("--chars", default=None)
    candidates.add_argument("--max-chars", type=int, default=None)

    report = sub.add_parser("report")
    report.add_argument("--root", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--label-font", default="/usr/share/fonts/truetype/nanum/NanumGothic.ttf")

    args = parser.parse_args(argv)
    if args.command == "build-dataset":
        build_dataset_from_config(args.config)
    elif args.command == "train":
        train_from_config(args.config)
    elif args.command == "infer":
        run_inference_from_config(args.config)
    elif args.command == "guided-infer":
        run_guided_inference_from_config(args.config)
    elif args.command == "candidate-preview":
        generate_candidate_preview(
            args.config,
            output_dir=args.output_dir,
            chars=args.chars,
            max_chars=args.max_chars,
        )
    elif args.command == "report":
        make_report(args.root, args.output, label_font=args.label_font)


if __name__ == "__main__":
    main()
