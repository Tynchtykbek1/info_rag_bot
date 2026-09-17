"""Deterministic language segmentation for normalized Telegram posts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


LanguageCode = Literal["ru", "en", "und"]

_RU_MARKER = "\U0001f1f7\U0001f1fa"
_EN_MARKER = "\U0001f1ec\U0001f1e7"
_SEPARATOR = "\u2014"
_FOOTER = re.compile(
    r"(?:\n[ \t]*)*t\.me/MessinaInfo[ \t]*\|[ \t]*"
    r"instagram\.com/messinainfo[ \t]*$"
)


@dataclass(frozen=True)
class LanguageSection:
    language: LanguageCode
    title: str
    text: str


def _without_footer(text: str) -> str:
    return _FOOTER.sub("", text)


def _clean_text(text: str) -> str:
    """Remove structural separators and collapse runs of blank lines."""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if line.strip() == _SEPARATOR:
            continue
        if not line.strip():
            if cleaned and not blank:
                cleaned.append("")
            blank = True
            continue
        cleaned.append(line.rstrip())
        blank = False
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned)


def _informative(text: str) -> bool:
    return sum(character.isalnum() for character in text) >= 20


def _marker_line(lines: list[str], marker: str, start: int = 0) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip() == marker:
            return index
    return None


def _section(language: LanguageCode, title: str, body: str) -> LanguageSection | None:
    clean_body = _clean_text(body)
    if not _informative(clean_body):
        return None
    final_text = f"{title}\n\n{clean_body}" if title else clean_body
    return LanguageSection(language=language, title=title, text=final_text)


def split_language_sections(text: str) -> list[LanguageSection]:
    """Split a normalized Telegram message without losing meaningful content."""

    without_footer = _without_footer(text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = without_footer.split("\n")
    ru_index = _marker_line(lines, _RU_MARKER)
    en_index = (
        _marker_line(lines, _EN_MARKER, ru_index + 1)
        if ru_index is not None
        else None
    )

    if ru_index is None or en_index is None:
        clean_text = _clean_text(without_footer)
        if not _informative(clean_text):
            return []
        return [LanguageSection(language="und", title="", text=clean_text)]

    title = _clean_text("\n".join(lines[:ru_index]))
    ru_body = "\n".join(lines[ru_index + 1 : en_index])
    en_body = "\n".join(lines[en_index + 1 :])
    sections = (
        _section("ru", title, ru_body),
        _section("en", title, en_body),
    )
    return [section for section in sections if section is not None]
