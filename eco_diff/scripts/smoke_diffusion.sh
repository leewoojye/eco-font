#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

FONT="$(python - <<'PY'
from pathlib import Path
candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
for item in candidates:
    if Path(item).exists():
        print(item)
        break
else:
    for root in [Path("/usr/share/fonts"), Path.home() / ".fonts"]:
        if root.exists():
            for path in root.rglob("*.ttf"):
                print(path)
                raise SystemExit
    raise SystemExit("No system TTF font found")
PY
)"

rm -rf data/diffusion_smoke runs/diffusion_smoke outputs/diffusion_smoke

python -m eco_diff.cli build-dataset \
  --font "$FONT" \
  --chars "ECOFONT123" \
  --out-dir data/diffusion_smoke \
  --image-size 64 \
  --font-size 52 \
  --target-savings 0.25,0.40,0.55

python -m eco_diff.cli train-diffusion --config configs/diffusion.yaml

python -m eco_diff.cli sample-diffusion \
  --checkpoint runs/diffusion_smoke/diffusion_best.pt \
  --font "$FONT" \
  --chars "ECOFONT123" \
  --out-dir outputs/diffusion_smoke \
  --image-size 64 \
  --font-size 52 \
  --target-saving 0.40 \
  --num-candidates 4 \
  --sample-steps 24 \
  --force-ink-budget \
  --export-ttf outputs/diffusion_smoke/eco_diffusion.ttf

echo "Diffusion smoke complete: outputs/diffusion_smoke"
