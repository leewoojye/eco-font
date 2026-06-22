#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FONT="../assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-Regular.ttf"

PYTHONPATH=src python -m fontcrafter_lab.cli proxy-sample \
  --font "$FONT" \
  --chars "ᎣᏏᏲᏣᎳᎩ" \
  --element-kind blue_stone \
  --size 512 \
  --seed 17 \
  --out-dir outputs/cherokee_blue_stone_proxy
