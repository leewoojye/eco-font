#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

CHARS="가나다라마바사아자차카타파하한글에코폰트"
LABEL_FONT="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

rm -rf data/hangul_canonical runs/hangul_canonical outputs/hangul_canonical

python -m ryman_font.cli build-dataset \
  --font /usr/share/fonts/truetype/nanum/NanumGothic.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumGothicBold.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumSquareR.ttf \
  --font /usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf \
  --chars "$CHARS" \
  --out-dir data/hangul_canonical \
  --image-size 96 \
  --font-size 76 \
  --target-savings 0.50,0.60,0.70 \
  --target-style canonical

python -m ryman_font.cli train --config configs/hangul_canonical.yaml

python -m ryman_font.cli infer \
  --checkpoint runs/hangul_canonical/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul_canonical/saving_050 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.50

python -m ryman_font.cli infer \
  --checkpoint runs/hangul_canonical/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul_canonical/saving_060 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.60 \
  --export-ttf outputs/hangul_canonical/ryman_canonical_hangul_060.ttf

python -m ryman_font.cli infer \
  --checkpoint runs/hangul_canonical/checkpoint_best.pt \
  --font "$LABEL_FONT" \
  --chars "$CHARS" \
  --out-dir outputs/hangul_canonical/saving_070 \
  --image-size 96 \
  --font-size 76 \
  --target-saving 0.70

python -m ryman_font.cli report \
  --root outputs/hangul_canonical \
  --folders saving_050,saving_060,saving_070 \
  --labels "Canonical 0.50,Canonical 0.60,Canonical 0.70" \
  --chars "$CHARS" \
  --output outputs/hangul_canonical/contact_sheet.png \
  --label-font "$LABEL_FONT"

echo "Canonical Ryman hangul experiment complete: outputs/hangul_canonical"
