import hashlib

from messina_info.context import TRUNCATION_MARKER, build_context
from messina_info.retrieval import RetrievalDocument, SearchResult


def _result(message_id: int, text: str, language: str = "en") -> SearchResult:
    doc = RetrievalDocument(
        section_key=f"100:{message_id}:0:{language}", channel_id=100,
        message_id=message_id, date="2025-01-01T00:00:00+00:00",
        language=language, section_index=0,
        source_url=f"https://example.test/{message_id}", text=text,
        fingerprint=hashlib.sha256(text.encode()).hexdigest(),
    )
    return SearchResult(doc, 0.9, 0.0, 0.9)


def test_context_preserves_start_and_all_metadata() -> None:
    bundle = build_context([_result(1, "Introduction before FAQ\nFAQ: details")])
    source = bundle.sources[0]
    assert source.context_text.startswith("Introduction before FAQ")
    assert source.source_id == "tg-100-1"
    assert (source.channel_id, source.message_id, source.language) == (100, 1, "en")
    assert source.date and source.source_url == "https://example.test/1"
    assert "UNTRUSTED DATA" in bundle.text


def test_context_deduplicates_messages_in_relevance_order() -> None:
    bundle = build_context([_result(2, "best"), _result(2, "duplicate", "ru"), _result(3, "next")])
    assert [source.message_id for source in bundle.sources] == [2, 3]
    assert bundle.sources[0].text == "best"


def test_context_limit_marks_truncation_without_losing_start() -> None:
    bundle = build_context([_result(1, "BEGIN-" + "x" * 500)], max_chars=400)
    assert bundle.sources[0].context_text.startswith("BEGIN-")
    assert bundle.sources[0].context_text.endswith(TRUNCATION_MARKER)
    assert bundle.sources[0].truncated
    assert len(bundle.text) <= 400


def test_empty_context_is_safe() -> None:
    assert build_context([]).sources == ()
    assert build_context([]).text == ""
