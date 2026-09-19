import hashlib

import numpy as np
import pytest

from messina_info.contextual_retrieval import contextual_rerank_search
from messina_info.followup import ContextualQuery
from messina_info.retrieval import RetrievalDocument, SearchResult


class FakeIndex:
    def __init__(self, results: dict[str, list[SearchResult]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, object, float]] = []

    def search(
        self,
        query: str,
        *,
        k: int,
        embedding_provider: object = None,
        recency_weight: float,
    ) -> list[SearchResult]:
        self.calls.append((query, k, embedding_provider, recency_weight))
        return list(self.results[query][:k])


class FakeReranker:
    model_name = "fake-reranker"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def score(
        self, pairs: list[tuple[str, str]], *, batch_size: int
    ) -> np.ndarray:
        self.calls.append((list(pairs), batch_size))
        return np.asarray([self.scores[text] for _, text in pairs], dtype=np.float32)


def _document(
    key: str, message_id: int, text: str, *, language: str = "en"
) -> RetrievalDocument:
    return RetrievalDocument(
        section_key=key,
        channel_id=100,
        message_id=message_id,
        date=None,
        language=language,  # type: ignore[arg-type]
        section_index=0,
        source_url=f"https://example.test/{message_id}",
        text=text,
        fingerprint=hashlib.sha256(key.encode()).hexdigest(),
    )


def _result(document: RetrievalDocument, score: float) -> SearchResult:
    return SearchResult(document, score, 0.0, score)


def _query(contextualized: bool = True) -> ContextualQuery:
    return ContextualQuery(
        original_query="current",
        retrieval_query="previous and current" if contextualized else "current",
        history_messages_used=(1,) if contextualized else (),
        contextualized=contextualized,
    )


def _fixture() -> tuple[FakeIndex, FakeReranker]:
    a = _document("a", 1, "alpha")
    b = _document("b", 2, "bravo")
    c = _document("c", 3, "charlie")
    index = FakeIndex(
        {
            "current": [_result(a, 0.9), _result(b, 0.8)],
            "previous and current": [_result(b, 0.95), _result(c, 0.7)],
        }
    )
    return index, FakeReranker({"alpha": 0.1, "bravo": 0.9, "charlie": 0.5})


def test_standalone_runs_one_dense_search_and_one_reranker_call() -> None:
    index, reranker = _fixture()

    results = contextual_rerank_search(
        index, _query(False), reranker, candidate_k=2  # type: ignore[arg-type]
    )

    assert [call[0] for call in index.calls] == ["current"]
    assert len(reranker.calls) == 1
    assert all(result.contextual_dense_rank is None for result in results)


def test_contextualized_runs_two_searches_and_merges_section_keys() -> None:
    index, reranker = _fixture()

    results = contextual_rerank_search(
        index, _query(), reranker, candidate_k=3  # type: ignore[arg-type]
    )

    assert [call[0] for call in index.calls] == ["current", "previous and current"]
    assert {result.document.section_key for result in results} == {"a", "b", "c"}
    bravo = next(result for result in results if result.document.section_key == "b")
    assert bravo.original_dense_rank == 2
    assert bravo.contextual_dense_rank == 1
    assert bravo.original_semantic_score == pytest.approx(0.8)
    assert bravo.contextual_semantic_score == pytest.approx(0.95)


def test_candidates_from_only_one_list_are_preserved_and_rrf_is_correct() -> None:
    index, reranker = _fixture()

    results = contextual_rerank_search(
        index, _query(), reranker, candidate_k=3, rrf_constant=60  # type: ignore[arg-type]
    )

    by_key = {result.document.section_key: result for result in results}
    assert by_key["a"].contextual_dense_rank is None
    assert by_key["c"].original_dense_rank is None
    assert by_key["b"].fusion_score == pytest.approx(1 / 62 + 1 / 61)
    assert by_key["a"].fusion_score == pytest.approx(1 / 61)


def test_fusion_pool_is_limited_and_contextual_query_and_batch_are_forwarded() -> None:
    index, reranker = _fixture()
    provider = object()

    contextual_rerank_search(
        index,
        _query(),
        reranker,
        candidate_k=2,
        batch_size=7,
        embedding_provider=provider,  # type: ignore[arg-type]
    )  # type: ignore[arg-type]

    pairs, batch = reranker.calls[0]
    assert len(pairs) == 2
    assert {query for query, _ in pairs} == {"previous and current"}
    assert batch == 7
    assert all(call[2] is provider and call[3] == 0.0 for call in index.calls)


def test_reranker_changes_order() -> None:
    index, reranker = _fixture()

    results = contextual_rerank_search(
        index, _query(), reranker, candidate_k=3  # type: ignore[arg-type]
    )

    assert [result.document.section_key for result in results] == ["b", "c", "a"]
    assert [result.reranked_rank for result in results] == [1, 2, 3]


def test_deduplication_occurs_after_reranking() -> None:
    english = _document("en", 10, "english", language="en")
    russian = _document("ru", 10, "russian", language="ru")
    other = _document("other", 11, "other")
    index = FakeIndex(
        {
            "current": [
                _result(english, 0.9),
                _result(russian, 0.8),
                _result(other, 0.7),
            ]
        }
    )
    reranker = FakeReranker({"english": 0.1, "russian": 0.9, "other": 0.5})

    results = contextual_rerank_search(
        index, _query(False), reranker, k=2, candidate_k=3  # type: ignore[arg-type]
    )

    assert len(reranker.calls[0][0]) == 3
    assert [(result.document.message_id, result.document.language) for result in results] == [
        (10, "ru"),
        (11, "en"),
    ]


def test_deterministic_ties_use_fusion_dense_rank_then_section_key() -> None:
    a = _document("a", 1, "alpha")
    b = _document("b", 2, "bravo")
    index = FakeIndex({"current": [_result(b, 0.9), _result(a, 0.8)]})
    reranker = FakeReranker({"alpha": 1.0, "bravo": 1.0})

    results = contextual_rerank_search(
        index, _query(False), reranker, deduplicate=False  # type: ignore[arg-type]
    )

    assert [result.document.section_key for result in results] == ["b", "a"]


def test_section_key_breaks_complete_tie_deterministically() -> None:
    a = _document("a", 1, "alpha")
    b = _document("b", 2, "bravo")
    index = FakeIndex(
        {
            "current": [_result(a, 0.9), _result(b, 0.8)],
            "previous and current": [_result(b, 0.9), _result(a, 0.8)],
        }
    )
    reranker = FakeReranker({"alpha": 1.0, "bravo": 1.0})

    results = contextual_rerank_search(
        index, _query(), reranker, deduplicate=False  # type: ignore[arg-type]
    )

    assert [result.document.section_key for result in results] == ["a", "b"]


def test_k_zero_calls_nothing() -> None:
    index, reranker = _fixture()

    assert contextual_rerank_search(  # type: ignore[arg-type]
        index, _query(), reranker, k=0
    ) == []
    assert index.calls == []
    assert reranker.calls == []


@pytest.mark.parametrize("k", [-1, True, 1.5])
def test_invalid_k_is_rejected(k: object) -> None:
    index, reranker = _fixture()
    with pytest.raises(ValueError, match="k must"):
        contextual_rerank_search(index, _query(), reranker, k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_k", 0),
        ("candidate_k", True),
        ("batch_size", 0),
        ("batch_size", False),
        ("rrf_constant", 0),
        ("rrf_constant", -1),
    ],
)
def test_invalid_positive_options_are_rejected(field: str, value: object) -> None:
    index, reranker = _fixture()
    options = {field: value}
    with pytest.raises(ValueError, match=field):
        contextual_rerank_search(  # type: ignore[arg-type]
            index, _query(), reranker, **options
        )


@pytest.mark.parametrize(
    "query",
    [
        ContextualQuery("", "retrieval", (), False),
        ContextualQuery("original", " ", (1,), True),
    ],
)
def test_invalid_manual_contextual_query_is_rejected(query: ContextualQuery) -> None:
    index, reranker = _fixture()
    with pytest.raises(ValueError, match="must not be empty"):
        contextual_rerank_search(index, query, reranker)  # type: ignore[arg-type]


def test_invalid_reranker_scores_are_rejected() -> None:
    class InvalidReranker:
        model_name = "invalid"

        def score(self, pairs: object, *, batch_size: int) -> np.ndarray:
            return np.asarray([np.nan])

    index, _ = _fixture()
    with pytest.raises(ValueError, match="invalid scores"):
        contextual_rerank_search(  # type: ignore[arg-type]
            index, _query(False), InvalidReranker()
        )


def test_inputs_are_not_modified() -> None:
    index, reranker = _fixture()
    query = _query()
    original_lists = {key: list(value) for key, value in index.results.items()}

    contextual_rerank_search(index, query, reranker, candidate_k=3)  # type: ignore[arg-type]

    assert query == _query()
    assert index.results == original_lists
