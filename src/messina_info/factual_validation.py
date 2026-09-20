"""Deterministic checks for concrete facts in cited source text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Sequence

from .context import ContextSource


@dataclass(frozen=True)
class FactualValidationResult:
    valid: bool
    unsupported_facts: tuple[str, ...]


@dataclass(frozen=True)
class _Fact:
    category: str
    value: str
    display: str


_MONTHS = {
    1: "январь января january gennaio", 2: "февраль февраля february febbraio",
    3: "март марта march marzo", 4: "апрель апреля april aprile",
    5: "май мая may maggio", 6: "июнь июня june giugno",
    7: "июль июля july luglio", 8: "август августа august agosto",
    9: "сентябрь сентября september settembre", 10: "октябрь октября october ottobre",
    11: "ноябрь ноября november novembre", 12: "декабрь декабря december dicembre",
}
_MONTH_NUMBER = {name: number for number, names in _MONTHS.items() for name in names.split()}
_MONTH_PATTERN = "|".join(sorted(_MONTH_NUMBER, key=len, reverse=True))

_URL = re.compile(r"https?://[^\s<>()\[\]{}]+", re.I)
_EMAIL = re.compile(r"(?<![\w@])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w@])", re.I)
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
_DAY_MONTH = re.compile(rf"(?<!\d)(\d{{1,2}})\s+({_MONTH_PATTERN})\s*,?\s*(\d{{4}})(?!\d)", re.I)
_MONTH_DAY = re.compile(rf"(?<!\w)({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)", re.I)
_ACADEMIC_YEAR = re.compile(r"(?<!\d)(\d{4})\s*[/\-–—]\s*(\d{4})(?!\d)")
_NUMBER = r"(?:\d{1,3}(?:[ ,.]\d{3})+|\d+)(?:[.,]\d{1,2})?"
_CURRENCY = r"(?:€|eur\b|euros?\b|евро\b|\$|usd\b|dollars?\b|£|gbp\b|pounds?\b)"
_MONEY_PREFIX = re.compile(rf"(?<!\w)({_CURRENCY})\s*({_NUMBER})(?!\w)", re.I)
_MONEY_SUFFIX = re.compile(rf"(?<!\w)({_NUMBER})\s*({_CURRENCY})(?!\w)", re.I)
_PERCENT = re.compile(rf"(?<!\w)({_NUMBER})\s*(%|percent\b|процент(?:а|ов)?\b|per\s+cento\b)(?!\w)", re.I)
_TIME = re.compile(r"(?<![\d.])(\d{1,2})[:.](\d{2})(?![\d.])")
_PHONE = re.compile(r"(?<![\w])\+?\d[\d ()\-]{5,}\d(?!\w)")


def _decimal(raw: str) -> str | None:
    value = raw.replace(" ", "")
    for separator in (",", "."):
        if value.count(separator) == 1 and len(value.rsplit(separator, 1)[1]) in (1, 2):
            whole, fraction = value.rsplit(separator, 1)
            value = whole.replace(",", "").replace(".", "") + "." + fraction
            break
    else:
        value = value.replace(",", "").replace(".", "")
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return None


def _currency(raw: str) -> str:
    value = raw.casefold()
    if value in {"€", "eur", "euro", "euros", "евро"}:
        return "EUR"
    if value in {"$", "usd", "dollar", "dollars"}:
        return "USD"
    return "GBP"


def _extract(text: str) -> tuple[_Fact, ...]:
    occupied: list[tuple[int, int]] = []
    facts: list[_Fact] = []

    def add(match: re.Match[str], category: str, value: str | None) -> None:
        if value is None or any(match.start() < end and match.end() > start for start, end in occupied):
            return
        occupied.append(match.span())
        facts.append(_Fact(category, value, match.group(0)))

    for match in _URL.finditer(text):
        add(match, "url", match.group(0).rstrip(".,;!?").casefold())
    for match in _EMAIL.finditer(text):
        add(match, "email", match.group(0).casefold())

    def calendar(day: str, month: int, year: str) -> str | None:
        try:
            return date(int(year), month, int(day)).isoformat()
        except ValueError:
            return None

    for match in _NUMERIC_DATE.finditer(text):
        add(match, "date", calendar(match[1], int(match[2]), match[3]))
    for match in _DAY_MONTH.finditer(text):
        add(match, "date", calendar(match[1], _MONTH_NUMBER[match[2].casefold()], match[3]))
    for match in _MONTH_DAY.finditer(text):
        add(match, "date", calendar(match[2], _MONTH_NUMBER[match[1].casefold()], match[3]))
    for match in _ACADEMIC_YEAR.finditer(text):
        add(match, "academic_year", f"{match[1]}/{match[2]}")
    for match in _MONEY_PREFIX.finditer(text):
        number = _decimal(match[2])
        add(match, "money", f"{_currency(match[1])}:{number}" if number else None)
    for match in _MONEY_SUFFIX.finditer(text):
        number = _decimal(match[1])
        add(match, "money", f"{_currency(match[2])}:{number}" if number else None)
    for match in _PERCENT.finditer(text):
        add(match, "percent", _decimal(match[1]))
    for match in _TIME.finditer(text):
        hour, minute = int(match[1]), int(match[2])
        add(match, "time", f"{hour:02d}:{minute:02d}" if hour < 24 and minute < 60 else None)
    for match in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        add(match, "phone", digits if 7 <= len(digits) <= 15 else None)
    return tuple(facts)


def validate_cited_facts(
    answer: str, cited_sources: Sequence[ContextSource],
) -> FactualValidationResult:
    """Require every protected answer value in at least one cited, visible source."""
    supported = {
        (fact.category, fact.value)
        for source in cited_sources
        for fact in _extract(source.context_text)
    }
    unsupported = tuple(
        fact.display for fact in _extract(answer)
        if (fact.category, fact.value) not in supported
    )
    return FactualValidationResult(not unsupported, unsupported)
