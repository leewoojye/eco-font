#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder trainer for TinyStyleFusionAutoEncoder.")
    parser.add_argument("--help-design", action="store_true", help="Print the intended training setup.")
    args = parser.parse_args()
    _ = args
    print(
        "Training design: render (content font, style font, char) triples, "
        "input content/style/target-saving maps to TinyStyleFusionAutoEncoder, "
        "optimize BCE+L1+ink-target+template-readability losses. "
        "The deterministic generator is the default path until enough licensed Cherokee fonts are curated."
    )


if __name__ == "__main__":
    main()
