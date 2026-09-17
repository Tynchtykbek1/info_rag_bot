"""Messina Info package."""

from .ingestion import (
    TelegramExport,
    TelegramExportError,
    TelegramMessage,
    load_telegram_export,
)
from .database import initialize_database
from .importer import ImportStats, import_telegram_export
from .segmentation import LanguageCode, LanguageSection, split_language_sections
from .evaluation import (
    RetrievalCase,
    evaluate_retrieval,
    load_retrieval_cases,
    validate_retrieval_cases,
)
from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from .retrieval import (
    RetrievalDocument,
    RetrievalIndex,
    RetrievalIndexError,
    SearchResult,
    build_retrieval_documents,
    build_retrieval_index,
    load_retrieval_index,
)

__all__ = [
    "TelegramExport",
    "TelegramExportError",
    "TelegramMessage",
    "load_telegram_export",
    "ImportStats",
    "import_telegram_export",
    "initialize_database",
    "LanguageCode",
    "LanguageSection",
    "split_language_sections",
    "RetrievalCase",
    "load_retrieval_cases",
    "validate_retrieval_cases",
    "evaluate_retrieval",
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "RetrievalDocument",
    "RetrievalIndex",
    "RetrievalIndexError",
    "SearchResult",
    "build_retrieval_documents",
    "build_retrieval_index",
    "load_retrieval_index",
]
