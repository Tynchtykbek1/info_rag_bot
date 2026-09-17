import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from messina_info import initialize_database
from messina_info.retrieval import (
    INDEX_FORMAT_VERSION,
    MANIFEST_FILE,
    IndexManifest,
    RetrievalDocument,
    RetrievalIndex,
    RetrievalIndexError,
    build_retrieval_documents,
    build_retrieval_index,
    load_retrieval_index,
)


class FakeEmbeddingProvider:
    def __init__(
        self,
        mapping: dict[str, list[float]] | None = None,
        *,
        model_name: str = "fake-e5",
        dimension: int = 3,
    ) -> None:
        self._mapping = mapping or {}
        self._model_name = model_name
        self._dimension = dimension
        self.seen_batches: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        self.seen_batches.append(list(texts))
        rows = []
        for text in texts:
            if text in self._mapping:
                rows.append(self._mapping[text])
                continue
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([float(digest[index] + 1) for index in range(self.dimension)])
        return np.asarray(rows, dtype=np.float32)


def _insert_message(
    database: Path,
    message_id: int,
    text: str,
    *,
    published_at: int = 1_700_000_000,
) -> None:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.execute(
            """INSERT INTO messages (
                   channel_id, message_id, channel_name, channel_username,
                   published_at, edited_at, text, source_url, content_hash,
                   raw_json, indexed_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                100,
                message_id,
                "Synthetic",
                "synthetic",
                published_at,
                None,
                text,
                f"https://t.me/synthetic/{message_id}",
                content_hash,
                "{}",
                None,
                published_at,
                published_at,
            ),
        )


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "messages.db"
    initialize_database(database)
    bilingual = (
        "Shared title\n\n🇷🇺\n\nПодробная информация для русскоязычных "
        "студентов университета.\n\n—\n\n🇬🇧\n\nDetailed information for "
        "English-speaking university students."
    )
    _insert_message(database, 2, bilingual)
    _insert_message(database, 1, "A sufficiently informative monolingual notice.")
    return database


def _document(
    key: str,
    message_id: int,
    *,
    date: str | None = "2024-01-01T00:00:00+00:00",
    language: str = "en",
) -> RetrievalDocument:
    return RetrievalDocument(
        section_key=key,
        channel_id=100,
        message_id=message_id,
        date=date,
        language=language,  # type: ignore[arg-type]
        section_index=0,
        source_url=f"https://t.me/synthetic/{message_id}",
        text=f"Document {message_id}",
        fingerprint=hashlib.sha256(key.encode()).hexdigest(),
    )


def _index(
    vectors: np.ndarray,
    documents: list[RetrievalDocument],
    provider: FakeEmbeddingProvider,
) -> RetrievalIndex:
    manifest = IndexManifest(
        format_version=INDEX_FORMAT_VERSION,
        embedding_model=provider.model_name,
        vector_dimension=provider.dimension,
        document_count=len(documents),
        corpus_fingerprint="synthetic",
        built_at="2024-01-01T00:00:00+00:00",
    )
    return RetrievalIndex(vectors, documents, manifest, provider)


def test_builds_documents_from_sqlite_with_multilingual_metadata(tmp_path: Path) -> None:
    documents = build_retrieval_documents(_database(tmp_path))

    assert len(documents) == 3
    assert [document.language for document in documents] == ["und", "ru", "en"]
    assert len({document.section_key for document in documents}) == 3
    assert all(document.date and document.source_url for document in documents)
    assert all(len(document.fingerprint) == 64 for document in documents)


def test_build_save_load_and_repeat_without_duplicates(tmp_path: Path) -> None:
    database = _database(tmp_path)
    index_path = tmp_path / "index"
    provider = FakeEmbeddingProvider()

    first = build_retrieval_index(database, index_path, provider)
    second = build_retrieval_index(database, index_path, provider)
    loaded = load_retrieval_index(index_path, provider)

    assert first.manifest.document_count == second.manifest.document_count == 3
    assert len(loaded.documents) == len({doc.section_key for doc in loaded.documents})
    assert provider.seen_batches[0][0].startswith("passage: ")
    assert (index_path / "vectors.npz").exists()
    assert (index_path / "metadata.json").exists()
    assert (index_path / "manifest.json").exists()


def test_cosine_ranking_top_k_and_query_prefix() -> None:
    provider = FakeEmbeddingProvider(
        {"query: deadline": [1.0, 0.0, 0.0]}
    )
    index = _index(
        np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0]]),
        [_document("a", 1), _document("b", 2), _document("c", 3)],
        provider,
    )

    results = index.search("deadline", k=2)

    assert [result.document.message_id for result in results] == [1, 2]
    assert provider.seen_batches[-1] == ["query: deadline"]
    assert results[0].semantic_score == pytest.approx(1.0)
    assert all(result.final_score == result.semantic_score for result in results)


def test_ties_are_resolved_by_stable_section_key() -> None:
    provider = FakeEmbeddingProvider({"query: same": [1.0, 0.0, 0.0]})
    index = _index(
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        [_document("z", 2), _document("a", 1)],
        provider,
    )

    assert [r.document.section_key for r in index.search("same", k=10)] == ["a", "z"]


def test_empty_query_and_negative_k_are_rejected() -> None:
    provider = FakeEmbeddingProvider()
    index = _index(np.array([[1.0, 0.0, 0.0]]), [_document("a", 1)], provider)

    with pytest.raises(ValueError, match="query must not be empty"):
        index.search("  ")
    with pytest.raises(ValueError, match="k must not be negative"):
        index.search("query", k=-1)


def test_recency_bonus_is_bounded_and_missing_dates_are_safe() -> None:
    provider = FakeEmbeddingProvider({"query: recent": [1.0, 0.0, 0.0]})
    documents = [
        _document("future", 1, date="2030-01-01T00:00:00+00:00"),
        _document("missing", 2, date=None),
    ]
    index = _index(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), documents, provider)

    results = index.search(
        "recent",
        recency_weight=0.1,
        half_life_days=30,
        as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    assert results[0].document.section_key == "future"
    assert results[0].recency_bonus == pytest.approx(0.1)
    assert results[1].recency_bonus == 0.0
    assert all(0.0 <= result.recency_bonus <= 0.1 for result in results)


def test_corrupted_and_incompatible_indexes_are_rejected(tmp_path: Path) -> None:
    index_path = tmp_path / "index"
    provider = FakeEmbeddingProvider()
    build_retrieval_index(_database(tmp_path), index_path, provider)
    manifest_path = index_path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RetrievalIndexError, match="unsupported index format"):
        load_retrieval_index(index_path, provider)

    build_retrieval_index(_database(tmp_path / "second"), index_path, provider)
    (index_path / "vectors.npz").write_bytes(b"not a numpy archive")
    with pytest.raises(RetrievalIndexError, match="cannot read valid vectors"):
        load_retrieval_index(index_path, provider)


def test_model_and_dimension_mismatches_are_rejected(tmp_path: Path) -> None:
    index_path = tmp_path / "index"
    provider = FakeEmbeddingProvider()
    build_retrieval_index(_database(tmp_path), index_path, provider)

    with pytest.raises(RetrievalIndexError, match="model does not match"):
        load_retrieval_index(
            index_path, FakeEmbeddingProvider(model_name="different-model")
        )
    with pytest.raises(RetrievalIndexError, match="dimension does not match"):
        load_retrieval_index(index_path, FakeEmbeddingProvider(dimension=4))
