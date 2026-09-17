"""Embedding providers used by multilingual retrieval."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_ENV = "MESSINA_EMBEDDING_MODEL"


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal batch embedding interface accepted by the retrieval index."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def prepare_e5_queries(texts: Sequence[str]) -> list[str]:
    """Validate and add the E5 query prefix."""

    return _prepare_e5(texts, "query")


def prepare_e5_passages(texts: Sequence[str]) -> list[str]:
    """Validate and add the E5 passage prefix."""

    return _prepare_e5(texts, "passage")


def _prepare_e5(texts: Sequence[str], kind: str) -> list[str]:
    prepared: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{kind} text at position {index} must not be empty")
        prepared.append(f"{kind}: {text.strip()}")
    return prepared


def normalize_vectors(vectors: np.ndarray, *, expected_rows: int | None = None) -> np.ndarray:
    """Return finite, row-normalized float32 vectors."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("embedding vectors must be a non-empty two-dimensional matrix")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError(
            f"embedding provider returned {array.shape[0]} rows; expected {expected_rows}"
        )
    if not np.isfinite(array).all():
        raise ValueError("embedding vectors must contain only finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must not contain zero-length rows")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class SentenceTransformerEmbeddingProvider:
    """Lazy sentence-transformers provider for multilingual E5 embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        selected_model = model_name or os.getenv(
            EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL
        )
        if not selected_model.strip():
            raise ValueError("embedding model name must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model_name = selected_model
        self._batch_size = batch_size
        self._device = device
        self._model: object | None = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._load_model()
        assert self._dimension is not None
        return self._dimension

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for production embeddings; "
                    "install the 'retrieval' optional dependency"
                ) from exc
            kwargs = {"device": self._device} if self._device else {}
            model = SentenceTransformer(self._model_name, **kwargs)
            dimension_method = getattr(model, "get_embedding_dimension", None)
            if dimension_method is None:
                dimension_method = model.get_sentence_embedding_dimension
            dimension = dimension_method()
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError("embedding model reported an invalid dimension")
            self._model = model
            self._dimension = dimension
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        prepared: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"text at position {index} must not be empty")
            prepared.append(text.strip())
        model = self._load_model()
        vectors = model.encode(  # type: ignore[attr-defined]
            prepared,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        normalized = normalize_vectors(vectors, expected_rows=len(prepared))
        if normalized.shape[1] != self.dimension:
            raise ValueError(
                "embedding provider returned vectors with an inconsistent dimension"
            )
        return normalized
