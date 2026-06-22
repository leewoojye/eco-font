from __future__ import annotations

import argparse
import json
from pathlib import Path

from .elements import write_default_elements
from .flux_pipeline import FluxConfig, run_flux
from .proxy import ProxyConfig, run_proxy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fontcrafter-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    elements = sub.add_parser("make-elements")
    elements.add_argument("--out-dir", required=True, type=Path)
    elements.add_argument("--size", default=512, type=int)

    proxy = sub.add_parser("proxy-sample")
    proxy.add_argument("--font", required=True, type=Path)
    proxy.add_argument("--chars", required=True)
    proxy.add_argument("--out-dir", required=True, type=Path)
    proxy.add_argument("--element-kind", default="blue_stone")
    proxy.add_argument("--size", default=512, type=int)
    proxy.add_argument("--seed", default=7, type=int)
    proxy.add_argument("--no-edge-repaint", action="store_true")

    flux = sub.add_parser("flux-sample")
    flux.add_argument("--font", required=True, type=Path)
    flux.add_argument("--chars", required=True)
    flux.add_argument("--element-image", required=True, type=Path)
    flux.add_argument("--out-dir", required=True, type=Path)
    flux.add_argument("--prompt", default="a stylized glyph made of the reference element, pure black background")
    flux.add_argument("--model-id", default="black-forest-labs/FLUX.1-Fill-dev")
    flux.add_argument("--size", default=512, type=int)
    flux.add_argument("--steps", default=28, type=int)
    flux.add_argument("--guidance-scale", default=30.0, type=float)
    flux.add_argument("--seed", default=0, type=int)
    flux.add_argument("--device", default="cuda")
    flux.add_argument("--hf-home", default=Path(".hf_cache"), type=Path)
    flux.add_argument("--no-edge-repaint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "make-elements":
        paths = write_default_elements(args.out_dir, args.size)
        print(json.dumps({"elements": [str(path) for path in paths]}, indent=2))
        return
    if args.command == "proxy-sample":
        summary = run_proxy(
            ProxyConfig(
                font=args.font,
                chars=args.chars,
                out_dir=args.out_dir,
                element_kind=args.element_kind,
                size=args.size,
                seed=args.seed,
                edge_repaint=not args.no_edge_repaint,
            )
        )
        print(json.dumps({"output": str(args.out_dir), "glyphs": len(summary["glyphs"])}, ensure_ascii=False, indent=2))
        return
    if args.command == "flux-sample":
        summary = run_flux(
            FluxConfig(
                font=args.font,
                chars=args.chars,
                element_image=args.element_image,
                out_dir=args.out_dir,
                prompt=args.prompt,
                model_id=args.model_id,
                size=args.size,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                device=args.device,
                hf_home=args.hf_home,
                edge_repaint=not args.no_edge_repaint,
            )
        )
        print(json.dumps({"output": str(args.out_dir), "glyphs": len(summary["glyphs"])}, ensure_ascii=False, indent=2))
        return
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
