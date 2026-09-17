"""SQLite schema and connection helpers for imported messages."""

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


def connect_database(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection without implicitly creating parent directories."""

    connection = sqlite3.connect(Path(database_path))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database_path: str | Path) -> None:
    """Create the database directory and messages table if necessary."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect_database(path) as connection:
        connection.execute(MESSAGES_SCHEMA)
