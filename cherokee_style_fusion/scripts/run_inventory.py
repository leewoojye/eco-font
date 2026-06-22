#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cherokee_style_fusion.font_bank import info_to_dict, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Cherokee Unicode coverage of font files.")
    parser.add_argument("--fonts-dir", action="append", default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    font_dirs = args.fonts_dir or [str(ROOT / "data" / "fonts")]
    infos = [info_to_dict(info) for info in inventory(font_dirs)]
    text = json.dumps(infos, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
