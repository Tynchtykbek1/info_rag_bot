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
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    UNCLEAR = "UNCLEAR"


class SmallTalkKind(str, Enum):
    GREETING = "GREETING"
    THANKS = "THANKS"
    FAREWELL = "FAREWELL"


_TALK = {
    "привет": "greeting", "здравствуйте": "greeting", "добрый день": "greeting",
    "hello": "greeting", "hi": "greeting", "hey": "greeting",
    "ciao": "greeting", "buongiorno": "greeting", "salve": "greeting",
    "спасибо": "thanks", "благодарю": "thanks", "thanks": "thanks",
    "thank you": "thanks", "grazie": "thanks",
    "пока": "farewell", "до свидания": "farewell", "bye": "farewell",
    "goodbye": "farewell", "arrivederci": "farewell",
    "спасибо большое": "thanks", "большое спасибо": "thanks", "спс": "thanks",
    "приветик": "greeting", "здарова": "greeting", "добрый вечер": "greeting",
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
_NEUTRAL = {
    "ru": "Если появятся вопросы об UniME или ERSU — пишите.",
    "en": "Feel free to ask about UniME or ERSU.",
    "it": "Scrivimi se hai domande su UniME o ERSU.",
}
_FILLERS = {"окей", "ок", "ладно", "хорошо", "okay", "ok", "va", "bene", "большое"}
_THANKS = {"спасибо", "благодарю", "спс", "thanks", "thank", "you", "grazie"}
_FAREWELLS = {"пока", "bye", "goodbye", "arrivederci"}
_GREETINGS = {"привет", "приветик", "здарова", "hello", "hi", "hey", "ciao", "salve", "buongiorno"}

_ALIASES = {"стипа": "стипендия", "пермессо": "permesso di soggiorno",
            "ричевута": "ricevuta", "исее": "ISEE", "docs": "documents",
            "dorm": "student accommodation", "dorms": "student accommodation"}
_RU_FORMS = (
    (re.compile(r"док(?:и|ов)\Z"), "документы"),
    (re.compile(r"стипух(?:а|у|е|и)\Z"), "стипендия"),
    (re.compile(r"общаг(?:а|у|е|и|ой|ою)\Z"), "общежитие"),
    (re.compile(r"универ(?:а|е|у)?\Z"), "университет"),
)
_WORD = re.compile(r"\w+", re.UNICODE)
_DOMAIN = re.compile(
    r"(?<!\w)(?:unime|ersu|messina|мессин\w*|университет\w*|стипенд\w*|"
    r"общежит\w*|документ\w*|выплат\w*|universit\w*|scholarship\w*|"
    r"student accommodation|documents|permesso|ricevuta|isee)(?!\w)", re.I,
)
_FOLLOWUP = re.compile(r"(?:^|\W)(?:а|это|она|он|там|эта|его|ее|её|it|that|and|e|quello|капнула)(?:\W|$)", re.I)


@dataclass(frozen=True)
class RoutedMessage:
    original: str
    normalized: str
    route: MessageRoute
    small_talk_kind: SmallTalkKind | None = None


def route_message(message: str, language: Language) -> RoutedMessage:
    """Classify only complete small-talk phrases; preserve the original input."""
    words = _WORD.findall(unicodedata.normalize("NFC", message).casefold())
    phrase = " ".join(words)
    kind = _TALK.get(phrase)
    if kind is None and words and set(words) <= (_FILLERS | _THANKS | _FAREWELLS | _GREETINGS):
        tokens = set(words)
        if tokens & (_THANKS - {"you"}) and ("thank" not in tokens or "you" in tokens):
            kind = "thanks"
        elif tokens & _FAREWELLS:
            kind = "farewell"
        elif tokens & _GREETINGS:
            kind = "greeting"
    if kind is not None:
        return RoutedMessage(message, message, MessageRoute.SMALL_TALK,
                             SmallTalkKind(kind.upper()))

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.casefold() == "uni":
            return "università" if language == "it" else "university"
        for pattern, replacement in _RU_FORMS:
            if pattern.fullmatch(word.casefold()):
                return replacement
        return _ALIASES.get(word.casefold(), word)

    normalized = _WORD.sub(replace, unicodedata.normalize("NFC", message))
    return RoutedMessage(message, normalized, MessageRoute.DOMAIN_QUERY)


def direct_reply(kind: SmallTalkKind | None, language: Language) -> str:
    return _REPLIES[language][kind.value.lower()] if kind is not None else _NEUTRAL[language]


def needs_interpretation(routed: RoutedMessage) -> bool:
    """Only uncertain text reaches the optional interpreter."""
    if routed.route == MessageRoute.SMALL_TALK:
        return False
    if "капнула" in _WORD.findall(routed.normalized.casefold()):
        return True
    if _DOMAIN.search(routed.normalized):
        return False
    return True
