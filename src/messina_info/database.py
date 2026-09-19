"""SQLite schema and connection helpers for imported messages and conversations."""

from __future__ import annotations

import sqlite3
from pathlib import Path


MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    channel_name TEXT NOT NULL,
    channel_username TEXT NOT NULL,
    published_at INTEGER NOT NULL,
    edited_at INTEGER,
    text TEXT NOT NULL,
    source_url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    indexed_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (channel_id, message_id)
);
"""

CONVERSATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    external_chat_id TEXT NOT NULL,
    external_user_id TEXT,
    language TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (platform, external_chat_id)
);
"""

CONVERSATION_MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""

CONVERSATION_MESSAGES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id_id
ON conversation_messages (conversation_id, id);
"""


def connect_database(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection without implicitly creating parent directories."""

    connection = sqlite3.connect(Path(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path) -> None:
    """Create the database directory and required tables if necessary."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect_database(path) as connection:
        connection.execute(MESSAGES_SCHEMA)
        connection.execute(CONVERSATIONS_SCHEMA)
        connection.execute(CONVERSATION_MESSAGES_SCHEMA)
        connection.execute(CONVERSATION_MESSAGES_INDEX)
