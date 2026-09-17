"""Persistent exact vector retrieval for segmented Telegram messages."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    normalize_vectors,
    prepare_e5_passages,
    prepare_e5_queries,
)
from .segmentation import LanguageCode, split_language_sections


INDEX_FORMAT_VERSION = 1
VECTORS_FILE = "vectors.npz"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"


class RetrievalIndexError(ValueError):
    """Raised when an index is corrupt or incompatible."""


@dataclass(frozen=True)
class RetrievalDocument:
    section_key: str
    channel_id: int
    message_id: int
    date: str | None
    language: LanguageCode
    section_index: int
    source_url: str
    text: str
    fingerprint: str


@dataclass(frozen=True)
class IndexManifest:
    format_version: int
    embedding_model: str
    vector_dimension: int
    document_count: int
    corpus_fingerprint: str
    built_at: str


@dataclass(frozen=True)
class SearchResult:
    document: RetrievalDocument
    semantic_score: float
    recency_bonus: float
    final_score: float


def _read_only_connection(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path).resolve()
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def build_retrieval_documents(database_path: str | Path) -> list[RetrievalDocument]:
    """Read messages without mutation and split them into stable index documents."""

    documents: dict[str, RetrievalDocument] = {}
    connection = _read_only_connection(database_path)
    try:
        rows = connection.execute(
            """SELECT channel_id, message_id, published_at, text, source_url,
                      content_hash
               FROM messages
               ORDER BY channel_id, message_id"""
        ).fetchall()
    finally:
        connection.close()

    for row in rows:
        sections = split_language_sections(row["text"])
        for section_index, section in enumerate(sections):
            text = section.text.strip()
            if not text:
                continue
            section_key = (
                f"{row['channel_id']}:{row['message_id']}:{section_index}:"
                f"{section.language}"
            )
            fingerprint_source = json.dumps(
                {
                    "section_key": section_key,
                    "message_content_hash": row["content_hash"],
                    "text": text,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            fingerprint = hashlib.sha256(
                fingerprint_source.encode("utf-8")
            ).hexdigest()
            published_at = row["published_at"]
            date = None
            if isinstance(published_at, int):
                date = datetime.fromtimestamp(
                    published_at, tz=timezone.utc
                ).isoformat()
            documents[section_key] = RetrievalDocument(
                section_key=section_key,
                channel_id=row["channel_id"],
                message_id=row["message_id"],
                date=date,
                language=section.language,
                section_index=section_index,
                source_url=row["source_url"],
                text=text,
                fingerprint=fingerprint,
            )
    return [documents[key] for key in sorted(documents)]


def _corpus_fingerprint(documents: Sequence[RetrievalDocument]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.section_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.fingerprint.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_vectors(path: Path, vectors: np.ndarray) -> None:
    handle = tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, vectors=vectors)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_retrieval_index(
    database_path: str | Path,
    index_path: str | Path,
    embedding_provider: EmbeddingProvider | None = None,
    *,
    model_name: str | None = None,
) -> "RetrievalIndex":
    """Build and atomically persist an exact-search index."""

    provider = embedding_provider or SentenceTransformerEmbeddingProvider(model_name)
    if model_name is not None and provider.model_name != model_name:
        raise ValueError("embedding provider model does not match model_name")
    documents = build_retrieval_documents(database_path)
    if not documents:
        raise ValueError("database contains no indexable language sections")
    vectors = provider.encode(prepare_e5_passages([doc.text for doc in documents]))
    vectors = normalize_vectors(vectors, expected_rows=len(documents))
    if vectors.shape[1] != provider.dimension:
        raise ValueError("embedding provider dimension does not match encoded vectors")

    destination = Path(index_path)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = IndexManifest(
        format_version=INDEX_FORMAT_VERSION,
        embedding_model=provider.model_name,
        vector_dimension=vectors.shape[1],
        document_count=len(documents),
        corpus_fingerprint=_corpus_fingerprint(documents),
        built_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_vectors(destination / VECTORS_FILE, vectors)
    _atomic_json(destination / METADATA_FILE, [asdict(doc) for doc in documents])
    _atomic_json(destination / MANIFEST_FILE, asdict(manifest))
    return RetrievalIndex(vectors, documents, manifest, provider)


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as input_file:
            return json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalIndexError(f"cannot read valid index file: {path.name}") from exc


def _load_manifest(value: object) -> IndexManifest:
    if not isinstance(value, dict):
        raise RetrievalIndexError("manifest must be a JSON object")
    try:
        manifest = IndexManifest(**value)
    except (TypeError, ValueError) as exc:
        raise RetrievalIndexError("manifest has an invalid schema") from exc
    if manifest.format_version != INDEX_FORMAT_VERSION:
        raise RetrievalIndexError(
            f"unsupported index format version {manifest.format_version}"
        )
    if manifest.vector_dimension <= 0 or manifest.document_count < 0:
        raise RetrievalIndexError("manifest contains invalid dimensions or counts")
    return manifest


def _load_documents(value: object) -> list[RetrievalDocument]:
    if not isinstance(value, list):
        raise RetrievalIndexError("metadata must be a JSON array")
    try:
        documents = [RetrievalDocument(**item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RetrievalIndexError("metadata has an invalid schema") from exc
    if len({document.section_key for document in documents}) != len(documents):
        raise RetrievalIndexError("metadata contains duplicate section keys")
    return documents


def load_retrieval_index(
    index_path: str | Path,
    embedding_provider: EmbeddingProvider | None = None,
    *,
    expected_model: str | None = None,
) -> "RetrievalIndex":
    """Load and validate a persistent retrieval index."""

    source = Path(index_path)
    manifest = _load_manifest(_load_json(source / MANIFEST_FILE))
    documents = _load_documents(_load_json(source / METADATA_FILE))
    try:
        with np.load(source / VECTORS_FILE, allow_pickle=False) as archive:
            if set(archive.files) != {"vectors"}:
                raise RetrievalIndexError("vectors archive has unexpected entries")
            vectors = normalize_vectors(archive["vectors"])
    except (OSError, ValueError) as exc:
        if isinstance(exc, RetrievalIndexError):
            raise
        raise RetrievalIndexError("cannot read valid vectors.npz") from exc

    if len(documents) != manifest.document_count or vectors.shape[0] != len(documents):
        raise RetrievalIndexError("index document counts do not match")
    if vectors.shape[1] != manifest.vector_dimension:
        raise RetrievalIndexError("index vector dimension does not match manifest")
    if _corpus_fingerprint(documents) != manifest.corpus_fingerprint:
        raise RetrievalIndexError("index corpus fingerprint does not match metadata")
    if expected_model is not None and expected_model != manifest.embedding_model:
        raise RetrievalIndexError("index embedding model does not match expected model")
    if embedding_provider is not None:
        if embedding_provider.model_name != manifest.embedding_model:
            raise RetrievalIndexError("embedding provider model does not match index")
        if embedding_provider.dimension != manifest.vector_dimension:
            raise RetrievalIndexError("embedding provider dimension does not match index")
    return RetrievalIndex(vectors, documents, manifest, embedding_provider)


class RetrievalIndex:
    """In-memory exact cosine index with backend-neutral documents."""

    def __init__(
        self,
        vectors: np.ndarray,
        documents: Sequence[RetrievalDocument],
        manifest: IndexManifest,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.vectors = normalize_vectors(vectors, expected_rows=len(documents))
        self.documents = tuple(documents)
        self.manifest = manifest
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        embedding_provider: EmbeddingProvider | None = None,
        recency_weight: float = 0.0,
        half_life_days: float = 180.0,
        as_of: datetime | None = None,
    ) -> list[SearchResult]:
        """Return deterministic exact cosine matches for a non-empty query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if k < 0:
            raise ValueError("k must not be negative")
        if k == 0 or not self.documents:
            return []
        if not 0.0 <= recency_weight <= 0.25:
            raise ValueError("recency_weight must be between 0 and 0.25")
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        provider = embedding_provider or self.embedding_provider
        if provider is None:
            raise ValueError("an embedding provider is required for search")
        if provider.model_name != self.manifest.embedding_model:
            raise ValueError("embedding provider model does not match index")
        query_vector = provider.encode(prepare_e5_queries([query]))
        query_vector = normalize_vectors(query_vector, expected_rows=1)
        if query_vector.shape[1] != self.manifest.vector_dimension:
            raise ValueError("query vector dimension does not match index")
        semantic_scores = self.vectors @ query_vector[0]

        reference = as_of or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        candidates: list[SearchResult] = []
        for document, semantic in zip(self.documents, semantic_scores, strict=True):
            bonus = _recency_bonus(
                document.date, reference, recency_weight, half_life_days
            )
            candidates.append(
                SearchResult(
                    document=document,
                    semantic_score=float(semantic),
                    recency_bonus=bonus,
                    final_score=float(semantic) + bonus,
                )
            )
        candidates.sort(
            key=lambda result: (
                -result.final_score,
                -result.semantic_score,
                result.document.section_key,
            )
        )
        return candidates[: min(k, len(candidates))]


def _recency_bonus(
    date: str | None,
    as_of: datetime,
    recency_weight: float,
    half_life_days: float,
) -> float:
    if recency_weight == 0.0 or not date:
        return 0.0
    try:
        published = datetime.fromisoformat(date)
    except ValueError:
        return 0.0
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (as_of - published).total_seconds() / 86_400.0)
    return min(recency_weight, recency_weight * math.exp(-age_days / half_life_days))
