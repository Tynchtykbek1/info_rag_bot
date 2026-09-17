"""Orchestration for importing Telegram JSON exports into SQLite."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .database import connect_database, initialize_database
from .ingestion import TelegramExportError, normalize_record, read_telegram_payload


@dataclass(frozen=True)
class ImportStats:
    source_total: int
    inserted: int
    updated: int
    unchanged: int
    skipped_empty: int
    skipped_service: int


def _channel_id(payload: Mapping[str, Any]) -> int:
    value = payload.get("id")
    if isinstance(value, bool):
        raise TelegramExportError("the export has no valid integer channel id")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TelegramExportError(
            "the export has no valid integer channel id"
        ) from exc


def import_telegram_export(
    json_path: str | Path,
    database_path: str | Path,
    channel_username: str,
) -> ImportStats:
    """Parse a Telegram export and persist normalized messages atomically."""

    username = channel_username.removeprefix("@")
    if not username:
        raise ValueError("channel_username must not be empty")

    payload = read_telegram_payload(json_path)
    records = payload["messages"]
    channel_id = _channel_id(payload)
    channel_name = payload["name"]

    normalized = []
    skipped_empty = 0
    skipped_service = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise TelegramExportError("every item in 'messages' must be an object")
        if record.get("type") != "message":
            skipped_service += 1
            continue
        item = normalize_record(record)
        if not item.message.text.strip():
            skipped_empty += 1
            continue
        normalized.append(item)

    initialize_database(database_path)
    inserted = 0
    updated = 0
    unchanged = 0
    now = int(time.time())

    with connect_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for item in normalized:
            message = item.message
            content_hash = hashlib.sha256(message.text.encode("utf-8")).hexdigest()
            existing = connection.execute(
                """SELECT content_hash FROM messages
                   WHERE channel_id = ? AND message_id = ?""",
                (channel_id, message.id),
            ).fetchone()
            values = (
                channel_name,
                username,
                item.published_at,
                item.edited_at,
                message.text,
                f"https://t.me/{username}/{message.id}",
                content_hash,
                item.raw_json,
                now,
                channel_id,
                message.id,
            )
            if existing is None:
                connection.execute(
                    """INSERT INTO messages (
                           channel_name, channel_username, published_at, edited_at,
                           text, source_url, content_hash, raw_json, created_at,
                           updated_at, channel_id, message_id
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values[:9] + (now,) + values[9:],
                )
                inserted += 1
            elif existing["content_hash"] == content_hash:
                unchanged += 1
            else:
                connection.execute(
                    """UPDATE messages SET
                           channel_name = ?, channel_username = ?, published_at = ?,
                           edited_at = ?, text = ?, source_url = ?, content_hash = ?,
                           raw_json = ?, updated_at = ?, indexed_at = NULL
                       WHERE channel_id = ? AND message_id = ?""",
                    values,
                )
                updated += 1

    return ImportStats(
        source_total=len(records),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        skipped_empty=skipped_empty,
        skipped_service=skipped_service,
    )
