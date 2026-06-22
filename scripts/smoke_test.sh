#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FONT="${1:-/usr/share/fonts/truetype/nanum/NanumGothic.ttf}"

python -m ecofont.cli build-dataset \
  --font "$FONT" \
  --language ko \
  --text "가나다라마바사아자차카타파하" \
  --targets 0.20 \
  --max-chars 8 \
  --image-size 96 \
  --candidate-limit 40 \
  --output data/smoke

python -m ecofont.cli train \
  --dataset data/smoke \
  --output runs/smoke/model.pt \
  --epochs 1 \
  --batch-size 4 \
  --device cpu

python -m ecofont.cli infer \
  --font "$FONT" \
  --checkpoint runs/smoke/model.pt \
  --language ko \
  --text "가나다라" \
  --output outputs/smoke \
  --target-saving 0.20 \
  --device cpu
