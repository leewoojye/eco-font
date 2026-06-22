#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cherokee_style_fusion.font_bank import download_font_sources


DEMO_NAMES = {
    "NotoSansCherokee-Regular",
    "Digohweli",
    "Tsulehisanvhi",
    "Donisiladv",
    "Anowelisgv",
    "AboriginalSans",
    "AboriginalSerif",
    "Alitsoi",
    "Diquetlusdi",
    "MunchChanceryCherokee",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Cherokee font bank into this folder.")
    parser.add_argument("--sources", default=str(ROOT / "data" / "font_sources.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "fonts"))
    parser.add_argument("--preset", choices=["demo", "all"], default="demo")
    args = parser.parse_args()

    names = DEMO_NAMES if args.preset == "demo" else None
    paths = download_font_sources(args.sources, args.out_dir, names=names)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
