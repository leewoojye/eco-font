#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-../.venv/bin/python}"

"$PY" -m eco_research_hangul.cli build-dataset --config configs/smoke.yaml
"$PY" -m eco_research_hangul.cli train --config configs/smoke.yaml
"$PY" -m eco_research_hangul.cli infer --config configs/smoke.yaml
"$PY" -m eco_research_hangul.cli report --root outputs/smoke --output outputs/smoke/contact_sheet.png

