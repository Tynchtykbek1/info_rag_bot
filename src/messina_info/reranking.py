"""Optional second-stage reranking for dense retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .embeddings import EmbeddingProvider
from .retrieval import RetrievalDocument, RetrievalIndex


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_ONNX_FILE = "onnx/model_O3.onnx"


@runtime_checkable
class Reranker(Protocol):
    """Batch scoring interface for raw query and section-text pairs."""

    @property
    def model_name(self) -> str: ...

    def score(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class RerankedSearchResult:
    document: RetrievalDocument
    dense_rank: int
    semantic_score: float
    reranker_score: float
    reranked_rank: int


class ONNXCrossEncoderReranker:
    """Lazily loaded Sentence Transformers CrossEncoder reranker."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANKER_MODEL,
        *,
        backend: str = "onnx",
        provider: str = "CPUExecutionProvider",
        file_name: str | None = DEFAULT_ONNX_FILE,
    ) -> None:
        if not model_name_or_path.strip():
            raise ValueError("reranker model name or path must not be empty")
        if backend not in {"onnx", "torch"}:
            raise ValueError("reranker backend must be 'onnx' or 'torch'")
        self._model_name = model_name_or_path
        self.backend = backend
        self.provider = provider
        self.file_name = file_name
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _create_model(self) -> object:
        candidate = Path(self._model_name)
        if (candidate.is_absolute() or self._model_name.startswith(".")) and not candidate.exists():
            raise FileNotFoundError(f"reranker model path does not exist: {candidate}")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers with ONNX support is required for reranking"
            ) from exc

        if self.backend == "onnx":
            model_kwargs: dict[str, str] = {"provider": self.provider}
            if self.file_name:
                model_kwargs["file_name"] = self.file_name
            return CrossEncoder(
                self._model_name,
                backend="onnx",
                device="cpu",
                model_kwargs=model_kwargs,
                processor_kwargs={"fix_mistral_regex": True},
            )
        return CrossEncoder(
            self._model_name,
            backend="torch",
            device="cpu",
            processor_kwargs={"fix_mistral_regex": True},
        )

    def _load_model(self) -> object:
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def score(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("reranker batch_size must be positive")
        if not pairs:
            return np.empty((0,), dtype=np.float32)
        for index, (query, text) in enumerate(pairs):
            if not query.strip() or not text.strip():
                raise ValueError(f"reranker pair at position {index} contains empty text")
        model = self._load_model()
        scores = model.predict(  # type: ignore[attr-defined]
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(values) != len(pairs) or not np.isfinite(values).all():
            raise ValueError("reranker returned invalid scores")
        return values


def rerank_search(
    index: RetrievalIndex,
    query: str,
    reranker: Reranker,
    *,
    k: int = 5,
    candidate_k: int = 15,
    batch_size: int = 8,
    deduplicate: bool = True,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RerankedSearchResult]:
    """Rerank dense candidates and optionally deduplicate Telegram messages."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if k < 0:
        raise ValueError("k must not be negative")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if batch_size <= 0:
        raise ValueError("reranker batch_size must be positive")
    if k == 0:
        return []

    dense_results = index.search(
        query,
        k=candidate_k,
        embedding_provider=embedding_provider,
        recency_weight=0.0,
    )
    pairs = [(query, result.document.text) for result in dense_results]
    scores = reranker.score(pairs, batch_size=batch_size)
    if len(scores) != len(dense_results) or not np.isfinite(scores).all():
        raise ValueError("reranker returned invalid scores")

    ranked = [
        (dense_rank, result, float(score))
        for dense_rank, (result, score) in enumerate(
            zip(dense_results, scores, strict=True), start=1
        )
    ]
    ranked.sort(
        key=lambda item: (
            -item[2],
            item[0],
            item[1].document.section_key,
        )
    )
    if deduplicate:
        seen: set[tuple[int, int]] = set()
        unique = []
        for item in ranked:
            message_key = (
                item[1].document.channel_id,
                item[1].document.message_id,
            )
            if message_key not in seen:
                seen.add(message_key)
                unique.append(item)
        ranked = unique

    return [
        RerankedSearchResult(
            document=result.document,
            dense_rank=dense_rank,
            semantic_score=result.semantic_score,
            reranker_score=reranker_score,
            reranked_rank=reranked_rank,
        )
        for reranked_rank, (dense_rank, result, reranker_score) in enumerate(
            ranked[: min(k, len(ranked))], start=1
        )
    ]
