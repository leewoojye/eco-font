from __future__ import annotations

import unicodedata
from pathlib import Path

from fontTools.ttLib import TTFont

from .schemas import CodepointSetName, ScriptName


CHEROKEE_RANGES = ((0x13A0, 0x13FF), (0xAB70, 0xABBF))
HANGUL_SYLLABLE_RANGE = (0xAC00, 0xD7A3)


def _font_cmap(font_path: str | Path) -> dict[int, str]:
    font = TTFont(str(font_path), lazy=True)
    cmap: dict[int, str] = {}
    for table in font["cmap"].tables:
        cmap.update(table.cmap)
    return cmap


def _count_range(cmap: dict[int, str], ranges: tuple[tuple[int, int], ...]) -> int:
    count = 0
    for start, end in ranges:
        count += sum(1 for codepoint in cmap if start <= codepoint <= end)
    return count


def font_script_counts(font_path: str | Path) -> dict[str, int]:
    """Return script coverage counts inferred from the font cmap table."""
    cmap = _font_cmap(font_path)
    hangul_start, hangul_end = HANGUL_SYLLABLE_RANGE
    return {
        "cherokee": _count_range(cmap, CHEROKEE_RANGES),
        "hangul": sum(1 for codepoint in cmap if hangul_start <= codepoint <= hangul_end),
    }


def detect_font_script(font_path: str | Path) -> ScriptName | None:
    """Infer the primary supported script from a font's Unicode cmap coverage."""
    counts = font_script_counts(font_path)
    if counts["cherokee"] > 0:
        return "cherokee"
    if counts["hangul"] > 0:
        return "hangul"
    return None


def _assigned_cherokee_chars() -> list[str]:
    chars: list[str] = []
    for start, end in CHEROKEE_RANGES:
        for codepoint in range(start, end + 1):
            ch = chr(codepoint)
            try:
                name = unicodedata.name(ch)
            except ValueError:
                continue
            if name.startswith("CHEROKEE "):
                chars.append(ch)
    return chars


def _uploaded_cherokee_chars(font_path: str | Path) -> list[str]:
    cmap = _font_cmap(font_path)
    chars: list[str] = []
    for start, end in CHEROKEE_RANGES:
        for codepoint in range(start, end + 1):
            if codepoint in cmap:
                chars.append(chr(codepoint))
    return chars


def _hangul_full_chars() -> list[str]:
    start, end = HANGUL_SYLLABLE_RANGE
    return [chr(codepoint) for codepoint in range(start, end + 1)]


def chars_for_codepoint_set(
    script: ScriptName,
    codepoint_set: CodepointSetName,
    font_path: str | Path,
) -> tuple[list[str], list[str]]:
    """Return requested chars and missing chars for a font/codepoint-set pair."""
    cmap = _font_cmap(font_path)
    if script == "cherokee":
        if codepoint_set == "cherokee_full":
            requested = _assigned_cherokee_chars()
        elif codepoint_set == "uploaded_cherokee":
            requested = _uploaded_cherokee_chars(font_path)
        else:
            raise ValueError(f"codepoint_set '{codepoint_set}' is not valid for Cherokee")
    elif script == "hangul":
        if codepoint_set == "hangul_full":
            requested = _hangul_full_chars()
        elif codepoint_set == "hangul_subset":
            start, end = HANGUL_SYLLABLE_RANGE
            requested = [chr(codepoint) for codepoint in range(start, end + 1) if codepoint in cmap]
        else:
            raise ValueError(f"codepoint_set '{codepoint_set}' is not valid for Hangul")
    else:
        raise ValueError(f"unsupported script: {script}")

    missing = [ch for ch in requested if ord(ch) not in cmap]
    present = [ch for ch in requested if ord(ch) in cmap]
    if not present:
        raise ValueError(f"uploaded font does not contain any glyphs for {script}/{codepoint_set}")
    return present, missing
