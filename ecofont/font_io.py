"""Font inspection and glyph coverage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont


@dataclass(frozen=True)
class FontReport:
    path: Path
    total_codepoints: int
    requested_count: int
    supported_count: int
    missing_count: int
    supported_chars: list[str]
    missing_chars: list[str]


def supported_codepoints(font_path: str | Path) -> set[int]:
    """Read Unicode codepoints supported by a TTF/OTF font."""
    path = Path(font_path)
    font = TTFont(str(path), lazy=True)
    codepoints: set[int] = set()
    for table in font["cmap"].tables:
        codepoints.update(table.cmap.keys())
    font.close()
    return codepoints


def filter_supported_chars(font_path: str | Path, chars: list[str]) -> tuple[list[str], list[str]]:
    """Split characters into supported and missing groups."""
    cmap = supported_codepoints(font_path)
    supported = [ch for ch in chars if ord(ch) in cmap]
    missing = [ch for ch in chars if ord(ch) not in cmap]
    return supported, missing


def inspect_font(font_path: str | Path, chars: list[str]) -> FontReport:
    """Create a coverage report for a target character set."""
    cmap = supported_codepoints(font_path)
    supported = [ch for ch in chars if ord(ch) in cmap]
    missing = [ch for ch in chars if ord(ch) not in cmap]
    return FontReport(
        path=Path(font_path),
        total_codepoints=len(cmap),
        requested_count=len(chars),
        supported_count=len(supported),
        missing_count=len(missing),
        supported_chars=supported,
        missing_chars=missing,
    )
