from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont


CHEROKEE_MAIN = set(range(0x13A0, 0x1400))
CHEROKEE_SUPPLEMENT = set(range(0xAB70, 0xABC0))


@dataclass(frozen=True)
class FontInfo:
    path: Path
    family: str
    subfamily: str
    full_name: str
    cherokee_main: int
    cherokee_supplement: int
    size_bytes: int

    @property
    def usable(self) -> bool:
        return self.cherokee_main >= 80


def read_sources(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def download_font_sources(source_json: str | Path, out_dir: str | Path, names: set[str] | None = None) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for item in read_sources(source_json):
        if names and item["name"] not in names:
            continue
        target = out / item["filename"]
        if not target.exists():
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                target.write_bytes(response.read())
        downloaded.append(target)
    return downloaded


def find_font_files(font_dirs: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for root in font_dirs:
        path = Path(root)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}:
            files.append(path)
            continue
        for item in path.rglob("*"):
            if item.suffix.lower() in {".ttf", ".otf"}:
                files.append(item)
    return sorted(dict.fromkeys(files))


def _name(font: TTFont, ids: set[int]) -> str:
    values: list[str] = []
    for record in font["name"].names:
        if record.nameID not in ids:
            continue
        try:
            values.append(record.toUnicode())
        except UnicodeDecodeError:
            continue
    return values[0] if values else ""


def inspect_font(path: str | Path) -> FontInfo:
    path = Path(path)
    font = TTFont(str(path), lazy=True)
    codepoints: set[int] = set()
    if "cmap" in font:
        for table in font["cmap"].tables:
            codepoints.update(table.cmap.keys())
    return FontInfo(
        path=path,
        family=_name(font, {1}) or path.stem,
        subfamily=_name(font, {2}),
        full_name=_name(font, {4}) or path.stem,
        cherokee_main=len(codepoints & CHEROKEE_MAIN),
        cherokee_supplement=len(codepoints & CHEROKEE_SUPPLEMENT),
        size_bytes=path.stat().st_size,
    )


def inventory(font_dirs: list[str | Path]) -> list[FontInfo]:
    infos: list[FontInfo] = []
    for path in find_font_files(font_dirs):
        try:
            infos.append(inspect_font(path))
        except Exception:
            continue
    return infos


def select_base_font(fonts: list[FontInfo], preferences: list[str]) -> FontInfo:
    usable = [font for font in fonts if font.usable]
    if not usable:
        raise RuntimeError("No usable Cherokee Unicode font found")
    for preference in preferences:
        for font in usable:
            if font.path.name == preference or preference in font.full_name:
                return font
    for font in usable:
        if "NotoSansCherokee-Regular" in font.path.name or font.full_name == "Noto Sans Cherokee Regular":
            return font
    return sorted(usable, key=lambda f: (0 if "Noto" in f.family else 1, f.path.name))[0]


def select_style_fonts(fonts: list[FontInfo], base: FontInfo, limit: int) -> list[FontInfo]:
    usable = [font for font in fonts if font.usable and font.path != base.path]
    # Prefer distinct families first, then allow extra styles from the same family.
    selected: list[FontInfo] = []
    seen: set[str] = set()
    for font in sorted(usable, key=lambda f: (f.family == base.family, f.family, f.path.name)):
        key = font.family.strip().lower()
        if key in seen:
            continue
        selected.append(font)
        seen.add(key)
        if len(selected) >= limit:
            return selected
    for font in usable:
        if font not in selected:
            selected.append(font)
        if len(selected) >= limit:
            break
    return selected


def info_to_dict(info: FontInfo) -> dict:
    return {
        "path": str(info.path),
        "family": info.family,
        "subfamily": info.subfamily,
        "full_name": info.full_name,
        "cherokee_main": info.cherokee_main,
        "cherokee_supplement": info.cherokee_supplement,
        "size_bytes": info.size_bytes,
        "usable": info.usable,
    }
