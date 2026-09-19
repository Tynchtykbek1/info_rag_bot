"""Messina Info package."""

from .ingestion import (
    TelegramExport,
    TelegramExportError,
    TelegramMessage,
    load_telegram_export,
)
from .database import connect_database, initialize_database
from .conversations import (
    Conversation,
    ConversationMessage,
    append_message,
    clear_conversation,
    get_or_create_conversation,
    get_recent_messages,
)
from .followup import ContextualQuery, build_contextual_query
from .contextual_retrieval import (
    ContextualRerankedResult,
    contextual_rerank_search,
)
from .chat import ChatResult, ChatService
from .bot_settings import BotSettings, create_chat_service, load_bot_settings
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
from .reranking import (
    DEFAULT_RERANKER_MODEL,
    ONNXCrossEncoderReranker,
    RerankedSearchResult,
    Reranker,
    rerank_search,
)

__all__ = [
    "TelegramExport",
    "TelegramExportError",
    "TelegramMessage",
    "load_telegram_export",
    "ImportStats",
    "import_telegram_export",
    "initialize_database",
    "connect_database",
    "Conversation",
    "ConversationMessage",
    "get_or_create_conversation",
    "append_message",
    "get_recent_messages",
    "clear_conversation",
    "ContextualQuery",
    "build_contextual_query",
    "ContextualRerankedResult",
    "contextual_rerank_search",
    "ChatResult",
    "ChatService",
    "BotSettings",
    "load_bot_settings",
    "create_chat_service",
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
    "DEFAULT_RERANKER_MODEL",
    "ONNXCrossEncoderReranker",
    "RerankedSearchResult",
    "Reranker",
    "rerank_search",
]
