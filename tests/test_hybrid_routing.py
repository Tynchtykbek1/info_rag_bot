from pathlib import Path

import pytest

from messina_info.chat import ChatService
from messina_info.conversations import get_recent_messages
from messina_info.database import connect_database, initialize_database
from messina_info.interpreter import Interpretation
from messina_info.interpreter import GeminiMessageInterpreter
from messina_info.message_routing import MessageRoute, needs_interpretation, route_message
from messina_info.rag import FALLBACKS, RAGAnswer


class FakeRAG:
    def __init__(self, *, no_evidence=False):
        self.calls = []
        self.no_evidence = no_evidence

    def answer_contextual(self, query, language, *, top_k):
        self.calls.append(query)
        if self.no_evidence:
            return RAGAnswer("fallback", FALLBACKS[language], (), "quality", (), "insufficient_evidence")
        return RAGAnswer("answered", "Grounded answer", (), "quality", (), "supported")


class FakeInterpreter:
    def __init__(self, intent=MessageRoute.DOMAIN_QUERY, standalone="Standalone question?", error=None):
        self.intent, self.standalone, self.error, self.calls = intent, standalone, error, []

    def interpret(self, message, language, history):
        self.calls.append((message, language, tuple(history)))
        if self.error:
            raise self.error
        return Interpretation(intent=self.intent, standalone_query=self.standalone, reason="test")


def reply(service, query, language="ru"):
    return service.reply(platform="telegram", external_chat_id="1",
                         external_user_id=None, query=query, language=language)


@pytest.mark.parametrize("language,query,route,normalized", [
    ("ru", "спасибо большое!", MessageRoute.SMALL_TALK, "спасибо большое!"),
    ("ru", "большое спасибо", MessageRoute.SMALL_TALK, "большое спасибо"),
    ("ru", "спс", MessageRoute.SMALL_TALK, "спс"),
    ("ru", "приветик", MessageRoute.SMALL_TALK, "приветик"),
    ("ru", "здарова", MessageRoute.SMALL_TALK, "здарова"),
    ("ru", "добрый вечер", MessageRoute.SMALL_TALK, "добрый вечер"),
    ("en", "Thank you!", MessageRoute.SMALL_TALK, "Thank you!"),
    ("en", "Hi", MessageRoute.SMALL_TALK, "Hi"),
    ("it", "Grazie!", MessageRoute.SMALL_TALK, "Grazie!"),
    ("it", "Buongiorno", MessageRoute.SMALL_TALK, "Buongiorno"),
    ("ru", "Спасибо, а когда выплата?", MessageRoute.DOMAIN_QUERY, "Спасибо, а когда выплата?"),
    ("en", "Hello, when is the ERSU deadline?", MessageRoute.DOMAIN_QUERY, "Hello, when is the ERSU deadline?"),
    ("it", "Ciao, quando scade ERSU?", MessageRoute.DOMAIN_QUERY, "Ciao, quando scade ERSU?"),
    ("ru", "стипуха", MessageRoute.DOMAIN_QUERY, "стипендия"),
    ("ru", "стипуху", MessageRoute.DOMAIN_QUERY, "стипендия"),
    ("ru", "стипухе", MessageRoute.DOMAIN_QUERY, "стипендия"),
    ("ru", "стипухи", MessageRoute.DOMAIN_QUERY, "стипендия"),
    ("ru", "общага", MessageRoute.DOMAIN_QUERY, "общежитие"),
    ("ru", "общагу", MessageRoute.DOMAIN_QUERY, "общежитие"),
    ("ru", "общаге", MessageRoute.DOMAIN_QUERY, "общежитие"),
    ("ru", "общаги", MessageRoute.DOMAIN_QUERY, "общежитие"),
    ("ru", "универ", MessageRoute.DOMAIN_QUERY, "университет"),
    ("ru", "универа", MessageRoute.DOMAIN_QUERY, "университет"),
    ("ru", "универе", MessageRoute.DOMAIN_QUERY, "университет"),
    ("ru", "универу", MessageRoute.DOMAIN_QUERY, "университет"),
    ("ru", "доки", MessageRoute.DOMAIN_QUERY, "документы"),
    ("ru", "доков", MessageRoute.DOMAIN_QUERY, "документы"),
    ("ru", "пермессо", MessageRoute.DOMAIN_QUERY, "permesso di soggiorno"),
    ("ru", "ричевута", MessageRoute.DOMAIN_QUERY, "ricevuta"),
    ("ru", "исее", MessageRoute.DOMAIN_QUERY, "ISEE"),
    ("en", "permesso", MessageRoute.DOMAIN_QUERY, "permesso"),
    ("it", "ricevuta", MessageRoute.DOMAIN_QUERY, "ricevuta"),
    ("en", "ISEE", MessageRoute.DOMAIN_QUERY, "ISEE"),
    ("en", "docs", MessageRoute.DOMAIN_QUERY, "documents"),
    ("it", "uni", MessageRoute.DOMAIN_QUERY, "università"),
    ("en", "dorms", MessageRoute.DOMAIN_QUERY, "student accommodation"),
    ("ru", "капнула?", MessageRoute.DOMAIN_QUERY, "капнула?"),
    ("ru", "Блин, когда стипуха?", MessageRoute.DOMAIN_QUERY, "Блин, когда стипендия?"),
    ("en", "documentations", MessageRoute.DOMAIN_QUERY, "documentations"),
    ("ru", "университетский", MessageRoute.DOMAIN_QUERY, "университетский"),
])
def test_local_patterns(language, query, route, normalized):
    result = route_message(query, language)
    assert (result.route, result.normalized) == (route, normalized)


def test_clear_domain_skips_interpreter_and_uses_rag(tmp_path):
    interpreter, rag = FakeInterpreter(), FakeRAG()
    service = ChatService(tmp_path / "chat.db", rag, interpreter=interpreter)
    for query in ("Когда дедлайн ERSU?", "Где документы UniME?", "Черт, когда стипуха?"):
        assert reply(service, query).route == MessageRoute.DOMAIN_QUERY
    assert interpreter.calls == []
    assert len(rag.calls) == 3


@pytest.mark.parametrize("intent", [MessageRoute.OUT_OF_DOMAIN, MessageRoute.UNCLEAR,
                                    MessageRoute.SMALL_TALK])
def test_interpreted_direct_reply_skips_rag_and_persists(tmp_path, intent):
    interpreter, rag = FakeInterpreter(intent), FakeRAG()
    database = tmp_path / "chat.db"
    result = reply(ChatService(database, rag, interpreter=interpreter), "???")
    assert result.route == intent
    assert result.rag_answer.sources == ()
    assert rag.calls == []
    with connect_database(database) as connection:
        rows = connection.execute("SELECT role, content, intent FROM conversation_messages ORDER BY id").fetchall()
    assert rows[0][1] == "???" and rows[0][2] == intent.value
    assert rows[1][1] == result.rag_answer.answer


def test_ambiguous_followup_is_rewritten_then_retrieved(tmp_path):
    interpreter = FakeInterpreter(standalone="Какова общая сумма всей стипендии за учебный год, а не размер первой выплаты?")
    rag = FakeRAG(no_evidence=True)
    service = ChatService(tmp_path / "chat.db", rag, interpreter=interpreter)
    reply(service, "Когда первая выплата ERSU?")
    result = reply(service, "а вся часть?")
    assert len(interpreter.calls) == 1
    assert [item.role for item in interpreter.calls[0][2]] == ["user", "assistant"]
    assert result.contextual_query.original_query == interpreter.standalone
    assert rag.calls[-1].retrieval_query == interpreter.standalone
    assert result.rag_answer.technical_reason == "insufficient_evidence"
    assert result.rag_answer.answer == FALLBACKS["ru"]


def test_capnula_uses_history_and_one_interpretation(tmp_path):
    interpreter = FakeInterpreter(standalone="Уже была ли выплачена стипендия ERSU?")
    rag = FakeRAG()
    service = ChatService(tmp_path / "chat.db", rag, interpreter=interpreter)
    reply(service, "Стипендия ERSU")
    result = reply(service, "уже капнула?")
    assert len(interpreter.calls) == 1
    assert result.contextual_query.retrieval_query == interpreter.standalone


@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("429"), ValueError("invalid JSON")])
def test_interpreter_failure_degrades_safely(tmp_path, error):
    interpreter, rag = FakeInterpreter(error=error), FakeRAG()
    service = ChatService(tmp_path / "chat.db", rag, interpreter=interpreter)
    assert reply(service, "а это когда?").route == MessageRoute.UNCLEAR
    assert rag.calls == []
    reply(service, "Когда первая выплата ERSU?")
    assert reply(service, "а вся часть?").route == MessageRoute.DOMAIN_QUERY
    assert len(rag.calls) == 2


def test_non_domain_history_excluded_after_reopen(tmp_path):
    database = tmp_path / "chat.db"
    rag = FakeRAG()
    interpreter = FakeInterpreter(MessageRoute.OUT_OF_DOMAIN)
    service = ChatService(database, rag, interpreter=interpreter)
    reply(service, "Как приготовить пиццу?")
    reply(service, "спс")
    reply(service, "Когда ERSU?")
    reopened = ChatService(database, rag, interpreter=interpreter)
    result = reply(reopened, "А когда стипендия?")
    assert "пиццу" not in result.contextual_query.retrieval_query
    assert "спс" not in result.contextual_query.retrieval_query


def test_topic_switch_does_not_restore_old_subject(tmp_path):
    interpreter, rag = FakeInterpreter(), FakeRAG()
    service = ChatService(tmp_path / "chat.db", rag, interpreter=interpreter)
    reply(service, "Когда стипендия ERSU?")
    result = reply(service, "Где документы UniME?")
    assert not result.contextual_query.contextualized
    assert "стипендия" not in result.contextual_query.retrieval_query
    assert interpreter.calls == []


def test_interpretation_schema_and_bounded_sdk_request(monkeypatch):
    from types import SimpleNamespace
    import sys

    calls = []
    class Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(parsed={"intent": "DOMAIN_QUERY",
                                           "standalone_query": "Когда стипендия ERSU?",
                                           "reason": "follow-up"})
    provider = SimpleNamespace(model="same-model", max_retries=0,
                               _load_client=lambda: SimpleNamespace(models=Models()))
    fake_types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "google.genai", SimpleNamespace(types=fake_types))
    result = GeminiMessageInterpreter(provider, max_history_chars=8).interpret("а когда?", "ru", [])
    assert result.intent == MessageRoute.DOMAIN_QUERY
    assert calls[0]["model"] == "same-model"
    assert calls[0]["config"].response_json_schema == Interpretation.model_json_schema()
    assert len(calls) == 1


def test_invalid_interpretation_structure_falls_back(tmp_path):
    class InvalidInterpreter:
        def interpret(self, *args):
            return {"intent": "CONFIDENT_GUESS", "standalone_query": "x", "reason": "x"}
    rag = FakeRAG()
    result = reply(ChatService(tmp_path / "chat.db", rag,
                               interpreter=InvalidInterpreter()), "???")
    assert result.route == MessageRoute.UNCLEAR
    assert rag.calls == []


def test_legacy_database_migrates_without_data_loss(tmp_path):
    database = tmp_path / "legacy.db"
    initialize_database(database)
    with connect_database(database) as connection:
        connection.execute("ALTER TABLE conversation_messages RENAME TO old_messages")
        connection.execute("CREATE TABLE conversation_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at INTEGER NOT NULL)")
        connection.execute("INSERT INTO conversations VALUES ('c', 'telegram', '1', NULL, 'ru', 1, 1)")
        connection.execute("INSERT INTO conversation_messages VALUES (1, 'c', 'user', 'legacy', 1)")
        connection.execute("DROP TABLE old_messages")
    initialize_database(database)
    with connect_database(database) as connection:
        messages = get_recent_messages(connection, conversation_id="c", limit=5)
    assert messages[0].content == "legacy" and messages[0].intent is None
