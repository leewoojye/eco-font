#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FONT="../assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-Regular.ttf"
ELEMENT="assets/elements/blue_stone.png"

PYTHONPATH=src python -m fontcrafter_lab.cli make-elements \
  --out-dir assets/elements \
  --size 512

HF_HOME="$ROOT/.hf_cache" PYTHONPATH=src python -m fontcrafter_lab.cli flux-sample \
  --font "$FONT" \
  --chars "ᎣᏏᏲᏣᎳᎩ" \
  --element-image "$ELEMENT" \
  --out-dir outputs/cherokee_blue_stone_flux \
  --size 512 \
  --steps 28 \
  --guidance-scale 30 \
  --seed 17 \
  --hf-home "$ROOT/.hf_cache"
