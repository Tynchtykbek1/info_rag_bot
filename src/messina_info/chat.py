"""Application-level orchestration for persistent conversational RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .conversations import (
    append_message,
    get_or_create_conversation,
    get_recent_messages,
)
from .database import connect_database, initialize_database
from .followup import ContextualQuery, build_contextual_query
from .rag import FALLBACKS, Language, RAGAnswer, RAGService


@dataclass(frozen=True)
class ChatResult:
    conversation_id: str
    contextual_query: ContextualQuery
    rag_answer: RAGAnswer


def _positive_integer(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _required(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value.strip()


class ChatService:
    def __init__(
        self,
        database_path: str | Path,
        rag_service: RAGService,
        *,
        history_limit: int = 10,
        max_user_messages: int = 2,
        max_query_chars: int = 1200,
        top_k: int = 5,
    ) -> None:
        _positive_integer(history_limit, "history_limit")
        _positive_integer(max_user_messages, "max_user_messages")
        _positive_integer(max_query_chars, "max_query_chars")
        _positive_integer(top_k, "top_k")
        self.database_path = Path(database_path)
        self.rag_service = rag_service
        self.history_limit = history_limit
        self.max_user_messages = max_user_messages
        self.max_query_chars = max_query_chars
        self.top_k = top_k
        initialize_database(self.database_path)

    def reply(
        self,
        *,
        platform: str,
        external_chat_id: str,
        external_user_id: str | None,
        query: str,
        language: Language,
    ) -> ChatResult:
        normalized_query = _required(query, "query")
        normalized_platform = _required(platform, "platform")
        normalized_chat_id = _required(external_chat_id, "external_chat_id")
        if language not in FALLBACKS:
            raise ValueError("language must be ru, en, or it")

        connection = connect_database(self.database_path)
        try:
            with connection:
                conversation = get_or_create_conversation(
                    connection,
                    platform=normalized_platform,
                    external_chat_id=normalized_chat_id,
                    external_user_id=external_user_id,
                    language=language,
                )
                history = get_recent_messages(
                    connection,
                    conversation_id=conversation.id,
                    limit=self.history_limit,
                )
                append_message(
                    connection,
                    conversation_id=conversation.id,
                    role="user",
                    content=normalized_query,
                )
        finally:
            connection.close()

        contextual_query = build_contextual_query(
            normalized_query,
            history,
            max_user_messages=self.max_user_messages,
            max_chars=self.max_query_chars,
        )
        rag_answer = self.rag_service.answer_contextual(
            contextual_query, language, top_k=self.top_k
        )

        connection = connect_database(self.database_path)
        try:
            with connection:
                append_message(
                    connection,
                    conversation_id=conversation.id,
                    role="assistant",
                    content=rag_answer.answer,
                )
        finally:
            connection.close()
        return ChatResult(conversation.id, contextual_query, rag_answer)
