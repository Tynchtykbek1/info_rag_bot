"""Persistent SQLite repository for conversations and their messages."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Literal, cast
from uuid import uuid4


MessageRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class Conversation:
    id: str
    platform: str
    external_chat_id: str
    external_user_id: str | None
    language: str | None
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class ConversationMessage:
    id: int
    conversation_id: str
    role: MessageRole
    content: str
    created_at: int
    intent: str | None = None


def _required(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value.strip()


def _optional(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or None")
    normalized = value.strip()
    return normalized or None


def _conversation(row: sqlite3.Row | tuple[object, ...]) -> Conversation:
    return Conversation(
        id=cast(str, row[0]),
        platform=cast(str, row[1]),
        external_chat_id=cast(str, row[2]),
        external_user_id=cast(str | None, row[3]),
        language=cast(str | None, row[4]),
        created_at=cast(int, row[5]),
        updated_at=cast(int, row[6]),
    )


def _message(row: sqlite3.Row | tuple[object, ...]) -> ConversationMessage:
    return ConversationMessage(
        id=cast(int, row[0]),
        conversation_id=cast(str, row[1]),
        role=cast(MessageRole, row[2]),
        content=cast(str, row[3]),
        created_at=cast(int, row[4]),
        intent=cast(str | None, row[5]),
    )


def get_or_create_conversation(
    connection: sqlite3.Connection,
    *,
    platform: str,
    external_chat_id: str,
    external_user_id: str | None = None,
    language: str | None = None,
) -> Conversation:
    """Return a conversation for a platform/chat pair, creating it if needed."""

    normalized_platform = _required(platform, "platform")
    normalized_chat_id = _required(external_chat_id, "external_chat_id")
    normalized_user_id = _optional(external_user_id, "external_user_id")
    normalized_language = _optional(language, "language")
    now = int(time.time())
    conversation_id = str(uuid4())
    row = connection.execute(
        """
        INSERT INTO conversations (
            id, platform, external_chat_id, external_user_id, language,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (platform, external_chat_id) DO UPDATE SET
            external_user_id = COALESCE(excluded.external_user_id, external_user_id),
            language = COALESCE(excluded.language, language),
            updated_at = excluded.updated_at
        RETURNING *
        """,
        (
            conversation_id,
            normalized_platform,
            normalized_chat_id,
            normalized_user_id,
            normalized_language,
            now,
            now,
        ),
    ).fetchone()
    if row is None:  # pragma: no cover - SQLite RETURNING guarantees a row
        raise RuntimeError("conversation upsert returned no row")
    return _conversation(row)


def append_message(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    role: MessageRole,
    content: str,
    intent: str | None = None,
) -> ConversationMessage:
    """Append one message and update the owning conversation timestamp."""

    normalized_conversation_id = _required(conversation_id, "conversation_id")
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must not be empty")
    now = int(time.time())
    row = connection.execute(
        """
        INSERT INTO conversation_messages (conversation_id, role, content, created_at, intent)
        SELECT id, ?, ?, ?, ? FROM conversations WHERE id = ?
        RETURNING *
        """,
        (role, content, now, intent, normalized_conversation_id),
    ).fetchone()
    if row is None:
        raise ValueError("unknown conversation_id")
    connection.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, normalized_conversation_id),
    )
    return _message(row)


def get_recent_messages(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    limit: int,
) -> tuple[ConversationMessage, ...]:
    """Return at most ``limit`` recent messages in chronological order."""

    normalized_conversation_id = _required(conversation_id, "conversation_id")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    rows = connection.execute(
        """
        SELECT * FROM (
            SELECT * FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
        )
        ORDER BY id ASC
        """,
        (normalized_conversation_id, limit),
    ).fetchall()
    return tuple(_message(row) for row in rows)


def clear_conversation(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
) -> int:
    """Delete only a known conversation's messages and return the row count."""

    normalized_conversation_id = _required(conversation_id, "conversation_id")
    exists = connection.execute(
        "SELECT 1 FROM conversations WHERE id = ?", (normalized_conversation_id,)
    ).fetchone()
    if exists is None:
        raise ValueError("unknown conversation_id")
    cursor = connection.execute(
        "DELETE FROM conversation_messages WHERE conversation_id = ?",
        (normalized_conversation_id,),
    )
    return cursor.rowcount
