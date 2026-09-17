import hashlib
from pathlib import Path

import numpy as np
import pytest

from messina_info.evaluation import RetrievalCase, evaluate_retrieval
from messina_info.reranking import ONNXCrossEncoderReranker, rerank_search
from messina_info.retrieval import (
    IndexManifest,
    RetrievalDocument,
    RetrievalIndex,
)


class FakeEmbeddingProvider:
    model_name = "fake-e5"
    dimension = 2

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


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


class FailingReranker:
    model_name = "failing"

    def score(self, pairs: object, *, batch_size: int) -> np.ndarray:
        raise RuntimeError("provider failed")


def _document(key: str, message_id: int, language: str, text: str) -> RetrievalDocument:
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


def _index() -> RetrievalIndex:
    documents = [
        _document("a", 1, "en", "first section"),
        _document("b", 2, "en", "second english"),
        _document("c", 2, "ru", "second russian"),
        _document("d", 3, "it", "third section"),
    ]
    return RetrievalIndex(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]),
        documents,
        IndexManifest(1, "fake-e5", 2, 4, "synthetic", "2024-01-01T00:00:00Z"),
        FakeEmbeddingProvider(),
    )


def test_reranking_batches_candidates_and_preserves_dense_metadata() -> None:
    reranker = FakeReranker(
        {
            "first section": 0.1,
            "second english": 0.8,
            "second russian": 0.7,
            "third section": 0.2,
        }
    )

    results = rerank_search(
        _index(), "raw query", reranker, k=3, candidate_k=4, batch_size=8
    )

    assert len(reranker.calls) == 1
    assert reranker.calls[0][1] == 8
    assert [result.document.message_id for result in results] == [2, 3, 1]
    assert results[0].document.language == "en"
    assert results[0].dense_rank == 2
    assert results[0].semantic_score > results[1].semantic_score
    assert results[0].reranker_score == pytest.approx(0.8)
    assert [result.reranked_rank for result in results] == [1, 2, 3]


def test_deduplication_keeps_highest_scoring_language_section() -> None:
    reranker = FakeReranker(
        {
            "first section": 0.1,
            "second english": 0.4,
            "second russian": 0.9,
            "third section": 0.2,
        }
    )

    result = rerank_search(_index(), "query", reranker, candidate_k=4)[0]

    assert result.document.message_id == 2
    assert result.document.language == "ru"
    assert result.dense_rank == 3


def test_equal_scores_use_dense_rank_deterministically() -> None:
    reranker = FakeReranker(
        {
            "first section": 1.0,
            "second english": 1.0,
            "second russian": 1.0,
            "third section": 1.0,
        }
    )

    results = rerank_search(
        _index(), "query", reranker, k=4, candidate_k=4, deduplicate=False
    )

    assert [result.document.section_key for result in results] == ["a", "b", "c", "d"]


@pytest.mark.parametrize(("candidate_k", "expected"), [(2, 2), (20, 4)])
def test_candidate_k_is_bounded_by_corpus(
    candidate_k: int, expected: int
) -> None:
    reranker = FakeReranker(
        {
            "first section": 0.4,
            "second english": 0.3,
            "second russian": 0.2,
            "third section": 0.1,
        }
    )

    rerank_search(
        _index(),
        "query",
        reranker,
        k=20,
        candidate_k=candidate_k,
        deduplicate=False,
    )

    assert len(reranker.calls[0][0]) == expected


def test_empty_query_and_provider_errors_are_reported() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        rerank_search(_index(), " ", FakeReranker({}))
    with pytest.raises(RuntimeError, match="provider failed"):
        rerank_search(_index(), "query", FailingReranker())


def test_onnx_provider_is_lazy_and_batches_once(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model:
        def __init__(self) -> None:
            self.calls: list[tuple[object, int]] = []

        def predict(
            self,
            pairs: object,
            *,
            batch_size: int,
            show_progress_bar: bool,
            convert_to_numpy: bool,
        ) -> np.ndarray:
            self.calls.append((pairs, batch_size))
            return np.asarray([0.2, 0.1])

    model = Model()
    provider = ONNXCrossEncoderReranker("synthetic-model")
    monkeypatch.setattr(provider, "_create_model", lambda: model)

    assert provider.loaded is False
    scores = provider.score([("q", "one"), ("q", "two")], batch_size=8)

    assert provider.loaded is True
    assert scores.tolist() == pytest.approx([0.2, 0.1])
    assert len(model.calls) == 1
    assert model.calls[0][1] == 8


def test_missing_local_model_path_is_rejected(tmp_path: Path) -> None:
    provider = ONNXCrossEncoderReranker(str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        provider.score([("query", "text")], batch_size=8)


def test_dense_only_search_does_not_load_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ONNXCrossEncoderReranker("synthetic-model")
    monkeypatch.setattr(
        provider,
        "_create_model",
        lambda: pytest.fail("reranker should remain unloaded"),
    )

    assert _index().search("query", k=1)[0].document.message_id == 1
    assert provider.loaded is False


def test_evaluation_can_use_reranking() -> None:
    case = RetrievalCase(
        case_id="synthetic",
        query="raw query",
        language="en",
        category="test",
        answerable=True,
        relevant_message_ids=(2,),
        preferred_message_id=2,
        time_sensitive=False,
        notes="synthetic",
    )
    reranker = FakeReranker(
        {
            "first section": 0.1,
            "second english": 0.9,
            "second russian": 0.8,
            "third section": 0.2,
        }
    )

    report = evaluate_retrieval(
        _index(), [case], reranker=reranker, candidate_k=4, reranker_batch_size=8
    )

    assert report["overall"]["hit_at_1"] == 1.0
