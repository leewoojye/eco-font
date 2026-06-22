#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cherokee_style_fusion.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cherokee style-fusion eco-font candidates.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    out = run(args.config, args.out_dir or None)
    print(out)


if __name__ == "__main__":
    main()
