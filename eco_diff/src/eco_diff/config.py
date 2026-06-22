from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping config in {path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_float_list(value: str) -> list[float]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise ValueError("Expected at least one float value")
    return [float(v) for v in items]


def read_charset(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    chars: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for ch in line:
            if ch.isspace() or ch in seen:
                continue
            seen.add(ch)
            chars.append(ch)
    if not chars:
        raise ValueError(f"No characters found in charset file: {path}")
    return chars


def chars_from_args(chars: str | None, charset_file: str | None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    if chars:
        for ch in chars:
            if ch.isspace() or ch in seen:
                continue
            seen.add(ch)
            values.append(ch)
    if charset_file:
        for ch in read_charset(charset_file):
            if ch in seen:
                continue
            seen.add(ch)
            values.append(ch)
    if not values:
        raise ValueError("Provide --chars or --charset-file")
    return values
