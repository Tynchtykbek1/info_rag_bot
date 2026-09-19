import sqlite3
from pathlib import Path

import pytest

from messina_info.chat import ChatService
from messina_info.conversations import get_recent_messages
from messina_info.database import connect_database
from messina_info.rag import FALLBACKS, RAGAnswer


def _answer(text: str = "Answer", *, fallback: bool = False) -> RAGAnswer:
    return RAGAnswer(
        "fallback" if fallback else "answered",
        text,
        (),
        "quality",
        (),
        "insufficient_evidence" if fallback else "supported",
    )


class FakeRAG:
    def __init__(self, answer: RAGAnswer | None = None, error: Exception | None = None):
        self.answer = answer or _answer()
        self.error = error
        self.calls = []

    def answer_contextual(self, query, language, *, top_k):
        self.calls.append((query, language, top_k))
        if self.error:
            raise self.error
        return self.answer


def _messages(database: Path, chat_id: str):
    with connect_database(database) as connection:
        row = connection.execute(
            "SELECT id FROM conversations WHERE external_chat_id = ?", (chat_id,)
        ).fetchone()
        if row is None:
            return ()
        return get_recent_messages(connection, conversation_id=row[0], limit=100)


def test_first_question_is_standalone_and_messages_are_ordered(tmp_path: Path) -> None:
    database = tmp_path / "chat.db"
    rag = FakeRAG()
    service = ChatService(database, rag)  # type: ignore[arg-type]

    result = service.reply(platform="telegram", external_chat_id="1",
                           external_user_id="u1", query=" First question ", language="en")

    assert result.contextual_query.contextualized is False
    assert result.contextual_query.original_query == "First question"
    assert [(m.role, m.content) for m in _messages(database, "1")] == [
        ("user", "First question"), ("assistant", "Answer")
    ]


def test_followup_uses_only_previous_user_question(tmp_path: Path) -> None:
    database = tmp_path / "chat.db"
    rag = FakeRAG()
    service = ChatService(database, rag)  # type: ignore[arg-type]
    service.reply(platform="telegram", external_chat_id="1", external_user_id=None,
                  query="Scholarship deadline?", language="en")

    result = service.reply(platform="telegram", external_chat_id="1",
                           external_user_id=None, query="And the time?", language="en")

    assert result.contextual_query.contextualized is True
    assert "Scholarship deadline?" in result.contextual_query.retrieval_query
    assert "Answer" not in result.contextual_query.retrieval_query
    assert len(result.contextual_query.history_messages_used) == 1


def test_chat_ids_are_isolated_and_metadata_updates(tmp_path: Path) -> None:
    database = tmp_path / "chat.db"
    service = ChatService(database, FakeRAG())  # type: ignore[arg-type]
    service.reply(platform="telegram", external_chat_id="1", external_user_id="old",
                  query="One", language="en")
    service.reply(platform="telegram", external_chat_id="2", external_user_id="two",
                  query="Two", language="it")
    service.reply(platform="telegram", external_chat_id="1", external_user_id="new",
                  query="Follow", language="ru")

    assert [m.content for m in _messages(database, "2")] == ["Two", "Answer"]
    with connect_database(database) as connection:
        row = connection.execute(
            "SELECT external_user_id, language FROM conversations WHERE external_chat_id='1'"
        ).fetchone()
    assert tuple(row) == ("new", "ru")


def test_provider_fallback_is_saved(tmp_path: Path) -> None:
    database = tmp_path / "chat.db"
    fallback = _answer(FALLBACKS["en"], fallback=True)
    service = ChatService(database, FakeRAG(fallback))  # type: ignore[arg-type]

    service.reply(platform="telegram", external_chat_id="1", external_user_id=None,
                  query="Unknown", language="en")

    assert _messages(database, "1")[-1].content == FALLBACKS["en"]


def test_unexpected_rag_error_keeps_only_user_message(tmp_path: Path) -> None:
    database = tmp_path / "chat.db"
    service = ChatService(database, FakeRAG(error=RuntimeError("reranker failed")))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="reranker failed"):
        service.reply(platform="telegram", external_chat_id="1", external_user_id=None,
                      query="Question", language="en")

    assert [(m.role, m.content) for m in _messages(database, "1")] == [
        ("user", "Question")
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": " ", "platform": "telegram", "external_chat_id": "1", "language": "en"},
        {"query": "q", "platform": " ", "external_chat_id": "1", "language": "en"},
        {"query": "q", "platform": "telegram", "external_chat_id": " ", "language": "en"},
        {"query": "q", "platform": "telegram", "external_chat_id": "1", "language": "de"},
    ],
)
def test_invalid_reply_does_not_write(tmp_path: Path, kwargs: dict) -> None:
    database = tmp_path / "chat.db"
    service = ChatService(database, FakeRAG())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        service.reply(external_user_id=None, **kwargs)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("history_limit", 0), ("max_user_messages", True),
     ("max_query_chars", -1), ("top_k", False)],
)
def test_invalid_constructor_settings(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ChatService(tmp_path / "chat.db", FakeRAG(), **{field: value})  # type: ignore[arg-type]


def test_limits_and_top_k_are_forwarded(tmp_path: Path) -> None:
    database = tmp_path / "chat.db"
    rag = FakeRAG()
    service = ChatService(database, rag, history_limit=2, max_user_messages=1,
                          max_query_chars=80, top_k=3)  # type: ignore[arg-type]
    for query in ("first", "second", "third"):
        service.reply(platform="telegram", external_chat_id="1", external_user_id=None,
                      query=query, language="en")

    contextual, _, top_k = rag.calls[-1]
    assert contextual.history_messages_used
    assert len(contextual.history_messages_used) == 1
    assert len(contextual.retrieval_query) <= 80
    assert top_k == 3


def test_database_connections_are_closed_before_rag(tmp_path: Path, monkeypatch) -> None:
    import messina_info.chat as chat_module

    database = tmp_path / "chat.db"
    real_connect = chat_module.connect_database
    trackers = []

    class TrackingConnection:
        def __init__(self, inner): self.inner, self.closed = inner, False
        def __getattr__(self, name): return getattr(self.inner, name)
        def __enter__(self): self.inner.__enter__(); return self
        def __exit__(self, *args): return self.inner.__exit__(*args)
        def close(self): self.inner.close(); self.closed = True

    def tracked_connect(path):
        tracker = TrackingConnection(real_connect(path)); trackers.append(tracker); return tracker

    class AssertClosedRAG(FakeRAG):
        def answer_contextual(self, query, language, *, top_k):
            assert trackers and all(item.closed for item in trackers)
            return super().answer_contextual(query, language, top_k=top_k)

    service = ChatService(database, AssertClosedRAG())  # type: ignore[arg-type]
    monkeypatch.setattr(chat_module, "connect_database", tracked_connect)
    service.reply(platform="telegram", external_chat_id="1", external_user_id=None,
                  query="Question", language="en")
    assert all(item.closed for item in trackers)


def test_one_reply_does_not_duplicate_user_message(tmp_path: Path) -> None:
    database = tmp_path / "chat.db"
    service = ChatService(database, FakeRAG())  # type: ignore[arg-type]
    service.reply(platform="telegram", external_chat_id="1", external_user_id=None,
                  query="Once", language="en")
    assert [m.role for m in _messages(database, "1")].count("user") == 1
