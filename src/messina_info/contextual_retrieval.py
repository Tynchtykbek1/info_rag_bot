"""Dual dense retrieval with reciprocal-rank fusion and one reranker pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingProvider
from .followup import ContextualQuery
from .reranking import Reranker
from .retrieval import RetrievalDocument, RetrievalIndex, SearchResult


@dataclass(frozen=True)
class ContextualRerankedResult:
    document: RetrievalDocument
    original_dense_rank: int | None
    contextual_dense_rank: int | None
    original_semantic_score: float | None
    contextual_semantic_score: float | None
    fusion_score: float
    reranker_score: float
    reranked_rank: int


@dataclass(frozen=True)
class _FusedCandidate:
    document: RetrievalDocument
    original_dense_rank: int | None = None
    contextual_dense_rank: int | None = None
    original_semantic_score: float | None = None
    contextual_semantic_score: float | None = None
    fusion_score: float = 0.0

    @property
    def best_dense_rank(self) -> int:
        ranks = tuple(
            rank
            for rank in (self.original_dense_rank, self.contextual_dense_rank)
            if rank is not None
        )
        return min(ranks)


def _positive_integer(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _validate_query(query: ContextualQuery) -> None:
    if not isinstance(query, ContextualQuery):
        raise ValueError("query must be a ContextualQuery")
    if not isinstance(query.original_query, str) or not query.original_query.strip():
        raise ValueError("original_query must not be empty")
    if not isinstance(query.retrieval_query, str) or not query.retrieval_query.strip():
        raise ValueError("retrieval_query must not be empty")


def _merge_results(
    original: list[SearchResult],
    contextual: list[SearchResult],
    rrf_constant: int,
) -> list[_FusedCandidate]:
    by_key: dict[str, _FusedCandidate] = {}
    for source, is_contextual in ((original, False), (contextual, True)):
        for rank, result in enumerate(source, start=1):
            key = result.document.section_key
            previous = by_key.get(key)
            if previous is None:
                previous = _FusedCandidate(document=result.document)
            contribution = 1.0 / (rrf_constant + rank)
            if is_contextual:
                candidate = _FusedCandidate(
                    document=previous.document,
                    original_dense_rank=previous.original_dense_rank,
                    contextual_dense_rank=rank,
                    original_semantic_score=previous.original_semantic_score,
                    contextual_semantic_score=result.semantic_score,
                    fusion_score=previous.fusion_score + contribution,
                )
            else:
                candidate = _FusedCandidate(
                    document=previous.document,
                    original_dense_rank=rank,
                    contextual_dense_rank=previous.contextual_dense_rank,
                    original_semantic_score=result.semantic_score,
                    contextual_semantic_score=previous.contextual_semantic_score,
                    fusion_score=previous.fusion_score + contribution,
                )
            by_key[key] = candidate
    return sorted(
        by_key.values(),
        key=lambda candidate: (
            -candidate.fusion_score,
            candidate.best_dense_rank,
            candidate.document.section_key,
        ),
    )


def contextual_rerank_search(
    index: RetrievalIndex,
    query: ContextualQuery,
    reranker: Reranker,
    *,
    k: int = 5,
    candidate_k: int = 15,
    batch_size: int = 8,
    rrf_constant: int = 60,
    deduplicate: bool = True,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[ContextualRerankedResult]:
    """Fuse original/contextual dense searches, then rerank one bounded pool."""

    _validate_query(query)
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a non-negative integer")
    _positive_integer(candidate_k, "candidate_k")
    _positive_integer(batch_size, "batch_size")
    _positive_integer(rrf_constant, "rrf_constant")
    if k == 0:
        return []

    original = index.search(
        query.original_query,
        k=candidate_k,
        embedding_provider=embedding_provider,
        recency_weight=0.0,
    )
    contextual: list[SearchResult] = []
    if query.contextualized:
        contextual = index.search(
            query.retrieval_query,
            k=candidate_k,
            embedding_provider=embedding_provider,
            recency_weight=0.0,
        )
    fused = _merge_results(original, contextual, rrf_constant)[:candidate_k]
    reranker_query = (
        query.retrieval_query if query.contextualized else query.original_query
    )
    pairs = [(reranker_query, candidate.document.text) for candidate in fused]
    scores = np.asarray(
        reranker.score(pairs, batch_size=batch_size), dtype=np.float32
    ).reshape(-1)
    if len(scores) != len(fused) or not np.isfinite(scores).all():
        raise ValueError("reranker returned invalid scores")

    ranked = list(zip(fused, (float(score) for score in scores), strict=True))
    ranked.sort(
        key=lambda item: (
            -item[1],
            -item[0].fusion_score,
            item[0].best_dense_rank,
            item[0].document.section_key,
        )
    )
    if deduplicate:
        seen: set[tuple[int, int]] = set()
        unique: list[tuple[_FusedCandidate, float]] = []
        for item in ranked:
            message_key = (item[0].document.channel_id, item[0].document.message_id)
            if message_key not in seen:
                seen.add(message_key)
                unique.append(item)
        ranked = unique

    return [
        ContextualRerankedResult(
            document=candidate.document,
            original_dense_rank=candidate.original_dense_rank,
            contextual_dense_rank=candidate.contextual_dense_rank,
            original_semantic_score=candidate.original_semantic_score,
            contextual_semantic_score=candidate.contextual_semantic_score,
            fusion_score=candidate.fusion_score,
            reranker_score=score,
            reranked_rank=rank,
        )
        for rank, (candidate, score) in enumerate(ranked[:k], start=1)
    ]
