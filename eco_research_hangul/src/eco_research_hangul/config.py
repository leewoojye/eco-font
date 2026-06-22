from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_chars(chars: str | None = None, charset_file: str | Path | None = None) -> list[str]:
    values: list[str] = []
    if chars:
        values.extend([ch for ch in chars if not ch.isspace()])
    if charset_file:
        text = Path(charset_file).read_text(encoding="utf-8")
        values.extend([ch for ch in text if not ch.isspace()])
    seen: set[str] = set()
    unique: list[str] = []
    for ch in values:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    return unique


def project_path(base: str | Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(base) / p

