#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/woojye2020/decs_jupyter_lab/eco-font"
CF_DIR="$ROOT/cf_font"
FONT_GLOB="$ROOT/assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-*.ttf"
REGULAR_FONT="$ROOT/assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-Regular.ttf"

cd "$CF_DIR"

PYTHONPATH=src python -m cf_font.cli build-dataset \
  --fonts-glob "$FONT_GLOB" \
  --chars-file charsets/cherokee.txt \
  --out-dir data/cherokee_smoke \
  --styles contour,centerline,edge,diagonal \
  --target-savings 0.45,0.60 \
  --image-size 64 \
  --limit-chars 24 \
  --ref-count 8

PYTHONPATH=src python -m cf_font.cli train \
  --dataset data/cherokee_smoke \
  --out runs/cherokee_smoke/model.pt \
  --base-epochs 2 \
  --cf-epochs 2 \
  --batch-size 48 \
  --base-channels 16 \
  --style-dim 64 \
  --basis-count 4 \
  --cfm-temperature 0.18 \
  --device auto

PYTHONPATH=src python -m cf_font.cli train-ocr \
  --dataset data/cherokee_smoke \
  --out runs/cherokee_smoke/ocr.pt \
  --samples-per-char 64 \
  --epochs 8 \
  --batch-size 64 \
  --device auto

PYTHONPATH=src python -m cf_font.cli infer \
  --checkpoint runs/cherokee_smoke/model.pt \
  --font "$REGULAR_FONT" \
  --chars "ᎣᏏᏲᏣᎳᎩᎠᎡᎢᎣ" \
  --out-dir outputs/cherokee_smoke \
  --style auto \
  --target-saving 0.60 \
  --ocr-checkpoint runs/cherokee_smoke/ocr.pt \
  --ocr-threshold 0.45 \
  --isr-steps 4 \
  --save-candidates
