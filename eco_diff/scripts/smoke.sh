#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

FONT="$(python - <<'PY'
from pathlib import Path
candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
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

rm -rf data/smoke runs/smoke outputs/smoke

python -m eco_diff.cli build-dataset \
  --font "$FONT" \
  --chars "ABCDEabcde0123" \
  --out-dir data/smoke \
  --image-size 64 \
  --font-size 52 \
  --target-savings 0.15,0.25 \
  --max-records 20

python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
cfg["data"]["manifest"] = "data/smoke/manifest.jsonl"
cfg["data"]["image_size"] = 64
cfg["training"]["output_dir"] = "runs/smoke"
cfg["training"]["epochs"] = 1
cfg["training"]["batch_size"] = 4
cfg["model"]["base_channels"] = 8
cfg["model"]["depth"] = 3
Path("runs").mkdir(exist_ok=True)
Path("runs/smoke_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
PY

python -m eco_diff.cli train --config runs/smoke_config.yaml

python -m eco_diff.cli infer \
  --checkpoint runs/smoke/checkpoint_best.pt \
  --font "$FONT" \
  --chars "ABCDE" \
  --out-dir outputs/smoke \
  --image-size 64 \
  --font-size 52 \
  --target-saving 0.25

echo "Smoke test complete: outputs/smoke"
