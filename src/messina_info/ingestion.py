"""Read and normalize Telegram Desktop JSON exports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class TelegramExportError(ValueError):
    """Raised when a Telegram export cannot be parsed or validated."""


@dataclass(frozen=True, slots=True)
class TelegramMessage:
    """A normalized message from a Telegram export."""

    id: int
    date: datetime
    text: str
    author: str | None = None
    author_id: str | None = None


@dataclass(frozen=True, slots=True)
class TelegramExport:
    """Normalized channel metadata and messages."""

    name: str
    chat_type: str | None
    chat_id: int | str | None
    messages: tuple[TelegramMessage, ...]


@dataclass(frozen=True, slots=True)
class NormalizedTelegramRecord:
    """A normalized message paired with persistence-specific export fields."""

    message: TelegramMessage
    published_at: int
    edited_at: int | None
    raw_json: str


def _text_content(value: Any) -> str:
    """Flatten Telegram's string-or-rich-text representation."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text_content(part) for part in value)
    if isinstance(value, Mapping):
        return _text_content(value.get("text", ""))
    return ""


def _parse_date(value: Any, message_id: Any) -> datetime:
    if not isinstance(value, str):
        raise TelegramExportError(f"message {message_id!r} has no valid date")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TelegramExportError(
            f"message {message_id!r} has an invalid date: {value!r}"
        ) from exc


def _parse_unix_timestamp(value: Any, field: str, message_id: int) -> int:
    if isinstance(value, bool):
        raise TelegramExportError(f"message {message_id!r} has no valid {field}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TelegramExportError(
            f"message {message_id!r} has no valid {field}"
        ) from exc


def _parse_messages(records: Iterable[Any]) -> tuple[TelegramMessage, ...]:
    messages: list[TelegramMessage] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TelegramExportError("every item in 'messages' must be an object")
        if record.get("type") != "message":
            continue

        message_id = record.get("id")
        if not isinstance(message_id, int) or isinstance(message_id, bool):
            raise TelegramExportError("a message has no valid integer id")

        messages.append(
            TelegramMessage(
                id=message_id,
                date=_parse_date(record.get("date"), message_id),
                text=_text_content(record.get("text", "")),
                author=record.get("from") if isinstance(record.get("from"), str) else None,
                author_id=(
                    record.get("from_id")
                    if isinstance(record.get("from_id"), str)
                    else None
                ),
            )
        )
    return tuple(messages)


def read_telegram_payload(path: str | Path) -> Mapping[str, Any]:
    """Read and validate the top-level structure of a Telegram export."""

    export_path = Path(path)
    try:
        with export_path.open(encoding="utf-8") as export_file:
            payload = json.load(export_file)
    except OSError as exc:
        raise TelegramExportError(f"cannot read export: {export_path}") from exc
    except json.JSONDecodeError as exc:
        raise TelegramExportError(f"invalid JSON in export: {export_path}") from exc

    if not isinstance(payload, Mapping):
        raise TelegramExportError("the export root must be an object")

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise TelegramExportError("the export has no valid chat name")

    records = payload.get("messages")
    if not isinstance(records, list):
        raise TelegramExportError("the export 'messages' field must be an array")
    return payload


def normalize_record(record: Any) -> NormalizedTelegramRecord:
    """Validate and normalize one ordinary Telegram message record."""

    if not isinstance(record, Mapping):
        raise TelegramExportError("every item in 'messages' must be an object")
    message_id = record.get("id")
    if not isinstance(message_id, int) or isinstance(message_id, bool):
        raise TelegramExportError("a message has no valid integer id")

    edited_value = record.get("edited_unixtime")
    edited_at = (
        None
        if edited_value in (None, "")
        else _parse_unix_timestamp(edited_value, "edited_unixtime", message_id)
    )
    message = TelegramMessage(
        id=message_id,
        date=_parse_date(record.get("date"), message_id),
        text=_text_content(record.get("text", "")),
        author=record.get("from") if isinstance(record.get("from"), str) else None,
        author_id=(
            record.get("from_id") if isinstance(record.get("from_id"), str) else None
        ),
    )
    return NormalizedTelegramRecord(
        message=message,
        published_at=_parse_unix_timestamp(
            record.get("date_unixtime"), "date_unixtime", message_id
        ),
        edited_at=edited_at,
        raw_json=json.dumps(record, ensure_ascii=False, separators=(",", ":")),
    )


def load_telegram_export(path: str | Path) -> TelegramExport:
    """Load a Telegram Desktop JSON export from *path*.

    The function performs no writes and retains naive timestamps as exported by
    Telegram. Timestamps containing a UTC offset remain offset-aware.
    """

    payload = read_telegram_payload(path)
    name = payload["name"]
    records = payload["messages"]

    chat_type = payload.get("type")
    if not isinstance(chat_type, str):
        chat_type = None
    chat_id = payload.get("id")
    if not isinstance(chat_id, (int, str)) or isinstance(chat_id, bool):
        chat_id = None

    return TelegramExport(
        name=name,
        chat_type=chat_type,
        chat_id=chat_id,
        messages=_parse_messages(records),
    )
