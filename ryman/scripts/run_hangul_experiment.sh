#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

CHARS="가나다라마바사아자차카타파하한글에코폰트"
LABEL_FONT="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

rm -rf data/hangul runs/hangul outputs/hangul

python -m ryman_font.cli build-dataset \
  --font /usr/share/fonts/truetype/nanum/NanumGothic.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumGothicBold.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumSquareR.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf \
  --chars "$CHARS" \
  --out-dir data/hangul \
  --image-size 96 \
  --font-size 76 \
  --target-savings 0.35,0.45,0.55

python -m ryman_font.cli train --config configs/hangul.yaml

python -m ryman_font.cli infer \
  --checkpoint runs/hangul/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul/saving_035 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.35

python -m ryman_font.cli infer \
  --checkpoint runs/hangul/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul/saving_045 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.45 \
  --export-ttf outputs/hangul/ryman_hangul_045.ttf

python -m ryman_font.cli infer \
  --checkpoint runs/hangul/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul/saving_055 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.55

python -m ryman_font.cli report \
  --root outputs/hangul \
  --folders saving_035,saving_045,saving_055 \
  --labels "Ryman-like 0.35,Ryman-like 0.45,Ryman-like 0.55" \
  --chars "$CHARS" \
  --output outputs/hangul/contact_sheet.png \
  --label-font "$LABEL_FONT"

echo "Ryman hangul experiment complete: outputs/hangul"
