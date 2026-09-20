"""Offline checks for protected facts in cited Telegram text."""

import hashlib
from types import SimpleNamespace

import pytest

from messina_info.chat import ChatService
from messina_info.context import ContextSource
from messina_info.database import connect_database
from messina_info.factual_validation import validate_cited_facts
from messina_info.followup import ContextualQuery
from messina_info.interpreter import Interpretation
from messina_info.llm import StructuredResponse
from messina_info.message_routing import MessageRoute
from messina_info.rag import FALLBACKS, RAGService, SYSTEM_INSTRUCTION
from messina_info.retrieval import RetrievalDocument, SearchResult
from messina_info.telegram_bot import format_chat_result


def source(text: str, source_id: str = "tg-1-1", url: str = "https://example.test/source") -> ContextSource:
    return ContextSource(source_id, 1, int(source_id.rsplit("-", 1)[1]), None,
                         "ru", url, text, text, False)


@pytest.mark.parametrize("answer,evidence", [
    ("18.08.2026", "18 августа 2026"),
    ("18 августа 2026", "August 18, 2026"),
    ("August 18, 2026", "18 agosto 2026"),
    ("18 agosto 2026", "18.08.2026"),
    ("14:00", "14.00"),
    ("€2076", "€2,076"),
    ("2 076 EUR", "2076 евро"),
    ("20%", "20 percent"),
    ("20 процентов", "20 per cento"),
    ("2026/2027", "2026–2027"),
    ("INFO@UNIME.IT", "info@unime.it"),
    ("+39 333 123 4567", "+39 (333) 123-4567"),
    ("https://unime.it/guide", "https://unime.it/guide"),
    ("The application opens soon.", "A source without concrete facts."),
])
def test_equivalent_supported_facts(answer, evidence):
    assert validate_cited_facts(answer, [source(evidence)]).valid


@pytest.mark.parametrize("answer,evidence", [
    ("18.08.2026", "19 августа 2026"),
    ("14:00", "15.00"),
    ("€2076", "€2077"),
    ("€2076", "$2076"),
    ("20%", "21 percent"),
    ("2026/2027", "2025–2026"),
    ("contact@unime.it", "info@unime.it"),
    ("+39 333 123 4567", "+39 333 123 4568"),
    ("https://unime.it/guide", "https://unime.it/other"),
])
def test_unsupported_facts_are_reported(answer, evidence):
    result = validate_cited_facts(answer, [source(evidence)])
    assert not result.valid and result.unsupported_facts


def test_only_cited_source_text_counts_and_second_citation_can_support():
    first = source("Scholarship information", "tg-1-1")
    second = source("Payment: €2076", "tg-1-2")
    assert not validate_cited_facts("Payment: €2076", [first]).valid
    assert validate_cited_facts("Payment: €2076", [first, second]).valid
    hidden = ContextSource("tg-1-3", 1, 3, None, "ru", "https://example.test/3",
                           "Payment: €2076", "Payment details omitted", True)
    assert not validate_cited_facts("Payment: €2076", [hidden]).valid


class FakeLLM:
    def __init__(self, answer: str, citations=("tg-1-1",), answerable=True):
        self.answer, self.citations, self.answerable, self.calls = answer, citations, answerable, 0

    def generate(self, system_instruction, user_prompt):
        self.calls += 1
        return StructuredResponse(answerable=self.answerable, answer=self.answer,
                                  cited_source_ids=list(self.citations), reason="synthetic")


class FakeIndex:
    def __init__(self, texts):
        self.results = []
        for i, text in enumerate(texts, 1):
            document = RetrievalDocument(str(i), 1, i, None, "ru", 0,
                                         f"https://example.test/{i}", text,
                                         hashlib.sha256(str(i).encode()).hexdigest())
            self.results.append(SearchResult(document, 0.9, 0.0, 0.9))

    def search(self, query, *, k, recency_weight, embedding_provider=None):
        return self.results[:k]


@pytest.mark.parametrize("language", ["ru", "en", "it"])
def test_unsupported_answer_uses_localized_fallback_without_sources(language):
    llm = FakeLLM("The payment is €2077.")
    result = RAGService(FakeIndex(["The payment is €2076."]), llm).answer("payment", language)
    assert result.answer == FALLBACKS[language]
    assert result.technical_reason == "unsupported_factual_claim"
    assert result.sources == () and result.message_ids == ()
    assert format_chat_result(SimpleNamespace(rag_answer=result), language) == FALLBACKS[language]
    assert llm.calls == 1


def test_citation_validation_precedes_fact_check_and_only_citations_count():
    index = FakeIndex(["No amount here.", "Payment: €2076."])
    unsupported = RAGService(index, FakeLLM("Payment: €2076.")).answer("payment", "en")
    supported = RAGService(index, FakeLLM("Payment: €2076.", ("tg-1-1", "tg-1-2"))).answer("payment", "en")
    invalid = RAGService(index, FakeLLM("Payment: €2076.", ("unknown",))).answer("payment", "en")
    assert unsupported.technical_reason == "unsupported_factual_claim"
    assert supported.status == "answered" and len(supported.sources) == 2
    assert invalid.technical_reason == "invalid_citations"


def test_answer_without_protected_facts_and_telegram_source_link():
    result = RAGService(FakeIndex(["Documents are available."]),
                        FakeLLM("Documents are available.")).answer("documents", "en")
    assert result.status == "answered"
    formatted = format_chat_result(SimpleNamespace(rag_answer=result), "en")
    assert "Sources:" in formatted and "https://example.test/1" in formatted


def test_chat_persists_visible_guard_fallback(tmp_path, monkeypatch):
    index = FakeIndex(["Payment: €2076."])
    llm = FakeLLM("Payment: €2077.")
    rag = RAGService(index, llm, reranker=SimpleNamespace())
    monkeypatch.setattr("messina_info.rag.contextual_rerank_search",
                        lambda *args, **kwargs: index.results)
    database = tmp_path / "chat.db"
    result = ChatService(database, rag).reply(
        platform="telegram", external_chat_id="1", external_user_id=None,
        query="Когда выплата ERSU?", language="ru",
    )
    with connect_database(database) as connection:
        rows = connection.execute("SELECT role, content FROM conversation_messages ORDER BY id").fetchall()
    assert result.rag_answer.technical_reason == "unsupported_factual_claim"
    assert [tuple(row) for row in rows] == [("user", "Когда выплата ERSU?"),
                                            ("assistant", FALLBACKS["ru"])]


def test_direct_routes_bypass_fact_guard(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("fact guard called")
    monkeypatch.setattr("messina_info.rag.validate_cited_facts", forbidden)
    rag = RAGService(FakeIndex([]), FakeLLM("unused"))
    service = ChatService(tmp_path / "chat.db", rag)
    small = service.reply(platform="telegram", external_chat_id="1", external_user_id=None,
                          query="спасибо", language="ru")
    assert small.route == MessageRoute.SMALL_TALK

    class OOD:
        def interpret(self, *args):
            return Interpretation(intent=MessageRoute.OUT_OF_DOMAIN,
                                  standalone_query="unrelated", reason="test")
    service.interpreter = OOD()
    ood = service.reply(platform="telegram", external_chat_id="1", external_user_id=None,
                        query="как приготовить пиццу?", language="ru")
    assert ood.route == MessageRoute.OUT_OF_DOMAIN


def test_system_instruction_limits_irrelevant_years():
    assert "another academic year" in SYSTEM_INSTRUCTION
    assert "comparison" in SYSTEM_INSTRUCTION
    assert "only the question asked" in SYSTEM_INSTRUCTION
