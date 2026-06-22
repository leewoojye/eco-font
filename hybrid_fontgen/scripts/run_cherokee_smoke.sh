#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FONT="../assets/fonts/NotoSansCherokee-v2.001/NotoSansCherokee/hinted/ttf/NotoSansCherokee-Regular.ttf"
TRAIN_CHARS="ᎠᎡᎢᎣᎤᎥᎦᎧᎨᎩᎪᎫᎬᎭᎮᎯᎰᎱᎲᎳᎴᎵᎶᎷ"
TEST_CHARS="ᎣᏏᏲᏣᎳᎩ"

PYTHONPATH=src python -m hybrid_fontgen.cli build-dataset \
  --font "$FONT" \
  --chars "$TRAIN_CHARS" \
  --styles contour,centerline,edge,diagonal \
  --target-savings 0.45,0.60 \
  --image-size 96 \
  --out-dir data/chr_smoke

PYTHONPATH=src python -m hybrid_fontgen.cli train \
  --dataset data/chr_smoke \
  --out runs/chr_smoke/model.pt \
  --epochs 5 \
  --batch-size 16 \
  --device cpu

PYTHONPATH=src python -m hybrid_fontgen.cli train-ocr \
  --font "$FONT" \
  --chars "$TEST_CHARS" \
  --out runs/chr_smoke/ocr.pt \
  --image-size 96 \
  --samples-per-char 80 \
  --epochs 8 \
  --batch-size 48 \
  --device cpu

PYTHONPATH=src python -m hybrid_fontgen.cli infer \
  --checkpoint runs/chr_smoke/model.pt \
  --font "$FONT" \
  --chars "$TEST_CHARS" \
  --style auto \
  --target-saving 0.60 \
  --ocr-checkpoint runs/chr_smoke/ocr.pt \
  --image-size 96 \
  --out-dir outputs/chr_smoke \
  --export-ttf outputs/chr_smoke/hybrid_cherokee.ttf \
  --device cpu
