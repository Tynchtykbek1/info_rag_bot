"""Local, deterministic routing and domain alias normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from .rag import Language


class MessageRoute(str, Enum):
    SMALL_TALK = "SMALL_TALK"
    DOMAIN_QUERY = "DOMAIN_QUERY"


_TALK = {
    "привет": "greeting", "здравствуйте": "greeting", "добрый день": "greeting",
    "hello": "greeting", "hi": "greeting", "hey": "greeting",
    "ciao": "greeting", "buongiorno": "greeting", "salve": "greeting",
    "спасибо": "thanks", "благодарю": "thanks", "thanks": "thanks",
    "thank you": "thanks", "grazie": "thanks",
    "пока": "farewell", "до свидания": "farewell", "bye": "farewell",
    "goodbye": "farewell", "arrivederci": "farewell",
}

_REPLIES = {
    "ru": {"greeting": "Привет! Чем могу помочь по UniME или ERSU?",
           "thanks": "Пожалуйста! Если появятся вопросы о UniME или ERSU — пишите.",
           "farewell": "До свидания! Обращайтесь, если появятся вопросы."},
    "en": {"greeting": "Hello! How can I help with UniME or ERSU?",
           "thanks": "You're welcome! Ask anytime about UniME or ERSU.",
           "farewell": "Goodbye! Feel free to ask again."},
    "it": {"greeting": "Ciao! Come posso aiutarti con UniME o ERSU?",
           "thanks": "Prego! Scrivimi se hai domande su UniME o ERSU.",
           "farewell": "Arrivederci! Scrivimi quando vuoi."},
}

_ALIASES = {
    "доки": "документы", "стипуха": "стипендия", "стипа": "стипендия",
    "общага": "общежитие", "общаге": "общежитие", "общагу": "общежитие",
    "общаги": "общежитие", "общагой": "общежитие", "общагою": "общежитие",
    "универ": "университет", "пермессо": "permesso di soggiorno",
    "ричевута": "ricevuta", "исее": "ISEE", "docs": "documents",
    "dorm": "student accommodation", "dorms": "student accommodation",
}
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class RoutedMessage:
    original: str
    normalized: str
    route: MessageRoute
    small_talk_kind: str | None = None


def route_message(message: str, language: Language) -> RoutedMessage:
    """Classify only complete small-talk phrases; preserve the original input."""
    words = _WORD.findall(unicodedata.normalize("NFC", message).casefold())
    phrase = " ".join(words)
    kind = _TALK.get(phrase)
    if kind is not None:
        return RoutedMessage(message, message, MessageRoute.SMALL_TALK, kind)

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.casefold() == "uni":
            return "università" if language == "it" else "university"
        return _ALIASES.get(word.casefold(), word)

    normalized = _WORD.sub(replace, unicodedata.normalize("NFC", message))
    return RoutedMessage(message, normalized, MessageRoute.DOMAIN_QUERY)


def direct_reply(kind: str, language: Language) -> str:
    return _REPLIES[language][kind]
