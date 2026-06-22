# -*- coding: utf-8 -*-
"""Language-specific glyph and preview text presets."""

from __future__ import annotations

from collections import OrderedDict


KOREAN_TEXT = (
    "가나다라마바사아자차카타파하"
    "거너더러머버서어저처커터퍼허"
    "고노도로모보소오조초코토포호"
    "구누두루무부수우주추쿠투푸후"
    "기니디리미비시이지치키티피히"
    "한글에코폰트대한민국잉크절약가독성"
)

CHEROKEE_TEXT = (
    "ᎠᎡᎢᎣᎤᎥᎦᎧᎨᎩᎪᎫᎬᎭᎮᎯᎰᎱᎲᎳᎴᎵᎶᎷᎸ"
    "ᎹᎺᎻᎼᎽᎾᏀᏁᏂᏃᏄᏅᏆᏇᏈᏉᏊᏋᏌᏍᏎᏏᏐᏑᏒ"
    "ᏓᏔᏕᏖᏗᏘᏙᏚᏛᏜᏝᏞᏟᏠᏡᏢᏣᏤᏥᏦᏧᏨᏩᏪᏫᏬᏭᏮ"
    "ᏯᏰᏱᏲᏳᏴ"
)

PREVIEW_TEXT = {
    "ko": "한글 에코폰트 테스트",
    "chr": "ᎣᏏᏲ ᏣᎳᎩ",
    "mixed": "한글 ᎣᏏᏲ EcoFont",
}

LANGUAGE_TEXT = {
    "ko": KOREAN_TEXT,
    "chr": CHEROKEE_TEXT,
    "mixed": KOREAN_TEXT + CHEROKEE_TEXT,
}


def unique_characters(text: str) -> list[str]:
    """Return stable unique non-whitespace characters."""
    return list(OrderedDict((ch, None) for ch in text if not ch.isspace()).keys())


def characters_for_language(language: str, custom_text: str | None = None) -> list[str]:
    """Return candidate characters for a language preset or custom text."""
    if custom_text:
        return unique_characters(custom_text)
    try:
        return unique_characters(LANGUAGE_TEXT[language])
    except KeyError as exc:
        valid = ", ".join(sorted(LANGUAGE_TEXT))
        raise ValueError(f"Unknown language '{language}'. Valid presets: {valid}") from exc


def default_preview_text(language: str) -> str:
    """Return a short preview string for a language preset."""
    return PREVIEW_TEXT.get(language, PREVIEW_TEXT["ko"])
