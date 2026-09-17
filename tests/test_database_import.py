import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from messina_info import (
    TelegramExportError,
    import_telegram_export,
    initialize_database,
)


def _message(message_id: int, text: object = "Città di Messina") -> dict:
    return {
        "id": message_id,
        "type": "message",
        "date": "2024-01-02T09:30:00",
        "date_unixtime": "1704187800",
        "edited": "2024-01-02T10:00:00",
        "edited_unixtime": "1704189600",
        "from": "Synthetic Editor",
        "from_id": "channel1000",
        "text": text,
    }


def _write_export(path: Path, messages: list[object]) -> None:
    payload = {
        "name": "Synthetic Channel",
        "type": "public_channel",
        "id": 1000,
        "messages": messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _rows(database: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute("SELECT * FROM messages ORDER BY message_id").fetchall()
    finally:
        connection.close()


def test_first_import_inserts_and_second_is_unchanged(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    database = tmp_path / "nested" / "messages.db"
    rich_text = ["Visit ", {"type": "bold", "text": "Messina"}, "!"]
    _write_export(export_path, [_message(1, rich_text), _message(2)])

    first = import_telegram_export(export_path, database, "@MessinaInfo")
    second = import_telegram_export(export_path, database, "@MessinaInfo")

    assert database.exists()
    assert first.source_total == 2
    assert (first.inserted, first.updated, first.unchanged) == (2, 0, 0)
    assert (second.inserted, second.updated, second.unchanged) == (0, 0, 2)
    rows = _rows(database)
    assert len(rows) == 2
    assert rows[0]["text"] == "Visit Messina!"
    assert rows[0]["source_url"] == "https://t.me/MessinaInfo/1"
    assert rows[0]["channel_username"] == "MessinaInfo"
    assert rows[0]["published_at"] == 1704187800
    assert rows[0]["edited_at"] == 1704189600
    assert json.loads(rows[0]["raw_json"])["text"] == rich_text
    duplicates = sqlite3.connect(database).execute(
        """SELECT COUNT(*) FROM (
               SELECT channel_id, message_id FROM messages
               GROUP BY channel_id, message_id HAVING COUNT(*) > 1
           )"""
    ).fetchone()[0]
    assert duplicates == 0


def test_edited_text_updates_hash_and_resets_indexed_at(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    database = tmp_path / "messages.db"
    _write_export(export_path, [_message(1, "Original")])
    import_telegram_export(export_path, database, "MessinaInfo")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE messages SET indexed_at = 123 WHERE message_id = 1")

    _write_export(export_path, [_message(1, "Aggiornato 🎉")])
    stats = import_telegram_export(export_path, database, "MessinaInfo")

    row = _rows(database)[0]
    assert (stats.inserted, stats.updated, stats.unchanged) == (0, 1, 0)
    assert row["text"] == "Aggiornato 🎉"
    assert row["content_hash"] == hashlib.sha256(
        "Aggiornato 🎉".encode("utf-8")
    ).hexdigest()
    assert row["indexed_at"] is None


def test_empty_and_service_records_are_skipped(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    database = tmp_path / "messages.db"
    _write_export(
        export_path,
        [
            {"id": 1, "type": "service", "action": "create_channel"},
            _message(2, "  \n"),
            _message(3, "Published"),
        ],
    )

    stats = import_telegram_export(export_path, database, "MessinaInfo")

    assert stats.source_total == 3
    assert stats.inserted == 1
    assert stats.skipped_empty == 1
    assert stats.skipped_service == 1
    assert [row["message_id"] for row in _rows(database)] == [3]


def test_malformed_export_does_not_leave_partial_import(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    database = tmp_path / "messages.db"
    initialize_database(database)
    _write_export(export_path, [_message(1), _message(2) | {"date_unixtime": "bad"}])

    with pytest.raises(TelegramExportError, match="date_unixtime"):
        import_telegram_export(export_path, database, "MessinaInfo")

    assert _rows(database) == []


def test_initialize_database_creates_parent_and_required_table(tmp_path: Path) -> None:
    database = tmp_path / "one" / "two" / "messages.db"

    initialize_database(database)

    assert database.exists()
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(messages)")
        }
    assert columns == {
        "channel_id",
        "message_id",
        "channel_name",
        "channel_username",
        "published_at",
        "edited_at",
        "text",
        "source_url",
        "content_hash",
        "raw_json",
        "indexed_at",
        "created_at",
        "updated_at",
    }
