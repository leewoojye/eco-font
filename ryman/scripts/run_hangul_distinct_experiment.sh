#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

CHARS="가나다라마바사아자차카타파하한글에코폰트"
LABEL_FONT="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

rm -rf data/hangul_distinct runs/hangul_distinct outputs/hangul_distinct

python -m ryman_font.cli build-dataset \
  --font /usr/share/fonts/truetype/nanum/NanumGothic.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumGothicBold.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumSquareR.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf \
  --chars "$CHARS" \
  --out-dir data/hangul_distinct \
  --image-size 96 \
  --font-size 76 \
  --target-savings 0.52,0.62,0.72 \
  --target-style distinct

python -m ryman_font.cli train --config configs/hangul_distinct.yaml

python -m ryman_font.cli infer \
  --checkpoint runs/hangul_distinct/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul_distinct/saving_052 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.52

python -m ryman_font.cli infer \
  --checkpoint runs/hangul_distinct/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul_distinct/saving_062 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.62 \
  --export-ttf outputs/hangul_distinct/ryman_distinct_hangul_062.ttf

python -m ryman_font.cli infer \
  --checkpoint runs/hangul_distinct/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul_distinct/saving_072 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.72

python -m ryman_font.cli report \
  --root outputs/hangul_distinct \
  --folders saving_052,saving_062,saving_072 \
  --labels "Distinct 0.52,Distinct 0.62,Distinct 0.72" \
  --chars "$CHARS" \
  --output outputs/hangul_distinct/contact_sheet.png \
  --label-font "$LABEL_FONT"

echo "Distinct Ryman hangul experiment complete: outputs/hangul_distinct"
