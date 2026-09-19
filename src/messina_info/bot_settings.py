"""Environment settings and dependency factory for the local Telegram bot."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .chat import ChatService
from .embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL_ENV, SentenceTransformerEmbeddingProvider
from .llm import DEFAULT_GEMINI_MODEL, GEMINI_MODEL_ENV, GeminiProvider
from .rag import RAGService
from .reranking import DEFAULT_ONNX_FILE, DEFAULT_RERANKER_MODEL, ONNXCrossEncoderReranker
from .retrieval import load_retrieval_index


@dataclass(frozen=True)
class BotSettings:
    telegram_token: str = field(repr=False)
    database_path: Path
    index_path: Path
    embedding_model: str
    gemini_model: str
    gemini_timeout_seconds: float
    reranker_model: str
    reranker_backend: str
    reranker_provider: str
    reranker_file: str | None
    candidate_k: int
    reranker_batch_size: int


def _text(values: Mapping[str, str], name: str, default: str | None = None) -> str:
    value = values.get(name, default)
    if not isinstance(value, str):
        raise ValueError(f"{name} must not be empty")
    if not value.strip() and default is not None:
        value = default
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default))
    if not isinstance(raw, str) or re.fullmatch(r"[0-9]+", raw.strip()) is None:
        raise ValueError(f"{name} must be a positive integer")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name, str(default))
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a positive number")
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def load_bot_settings(environ: Mapping[str, str] | None = None) -> BotSettings:
    values = os.environ if environ is None else environ
    token = _text(values, "TELEGRAM_BOT_TOKEN")
    backend = _text(values, "MESSINA_RERANKER_BACKEND", "onnx")
    if backend not in {"onnx", "torch"}:
        raise ValueError("MESSINA_RERANKER_BACKEND must be onnx or torch")
    reranker_file_value = values.get("MESSINA_RERANKER_FILE", DEFAULT_ONNX_FILE)
    if not isinstance(reranker_file_value, str):
        raise ValueError("MESSINA_RERANKER_FILE must be a string")
    reranker_file = reranker_file_value.strip() or None
    return BotSettings(
        telegram_token=token,
        database_path=Path(_text(values, "MESSINA_DATABASE_PATH", "messina.db")),
        index_path=Path(_text(values, "MESSINA_INDEX_PATH", ".retrieval-index")),
        embedding_model=_text(values, EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL),
        gemini_model=_text(values, GEMINI_MODEL_ENV, DEFAULT_GEMINI_MODEL),
        gemini_timeout_seconds=_positive_float(values, "MESSINA_GEMINI_TIMEOUT_SECONDS", 30.0),
        reranker_model=_text(values, "MESSINA_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        reranker_backend=backend,
        reranker_provider=_text(values, "MESSINA_RERANKER_PROVIDER", "CPUExecutionProvider"),
        reranker_file=reranker_file,
        candidate_k=_positive_int(values, "MESSINA_CANDIDATE_K", 15),
        reranker_batch_size=_positive_int(values, "MESSINA_RERANKER_BATCH_SIZE", 8),
    )


def create_chat_service(settings: BotSettings) -> ChatService:
    embedding = SentenceTransformerEmbeddingProvider(settings.embedding_model)
    try:
        index = load_retrieval_index(
            settings.index_path, embedding, expected_model=settings.embedding_model
        )
    except Exception as exc:
        raise RuntimeError(f"Cannot load retrieval index from {settings.index_path}") from exc
    reranker = ONNXCrossEncoderReranker(
        settings.reranker_model,
        backend=settings.reranker_backend,
        provider=settings.reranker_provider,
        file_name=settings.reranker_file,
    )
    llm = GeminiProvider(
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    rag = RAGService(
        index,
        llm,
        reranker=reranker,
        candidate_k=settings.candidate_k,
        reranker_batch_size=settings.reranker_batch_size,
    )
    return ChatService(settings.database_path, rag)
