#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m venv .venv
PIP_CACHE_DIR="$ROOT/.cache/pip" .venv/bin/python -m pip install --upgrade pip
PIP_CACHE_DIR="$ROOT/.cache/pip" .venv/bin/python -m pip install -r requirements.txt
