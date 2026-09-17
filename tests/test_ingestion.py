import json
from datetime import datetime
from pathlib import Path

import pytest

from messina_info.ingestion import TelegramExportError, load_telegram_export


FIXTURE = Path(__file__).parent / "fixtures" / "telegram_export.json"


def test_loads_export_and_normalizes_rich_text() -> None:
    export = load_telegram_export(FIXTURE)

    assert export.name == "Synthetic Messina Channel"
    assert export.chat_type == "public_channel"
    assert export.chat_id == 1_000_000_000
    assert len(export.messages) == 2
    assert export.messages[0].date == datetime(2024, 1, 2, 9, 30)
    assert export.messages[0].author == "Synthetic Editor"
    assert export.messages[0].author_id == "user100"
    assert export.messages[1].text == "Road closure on Via Roma."


def test_skips_service_records(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["messages"].insert(0, {"id": 0, "type": "service", "action": "create_group"})
    path = tmp_path / "export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert [message.id for message in load_telegram_export(path).messages] == [1, 2]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ([], "root must be an object"),
        ({"name": "Chat"}, "'messages' field must be an array"),
        ({"name": "Chat", "messages": [{"id": "1", "type": "message"}]}, "integer id"),
        ({"name": "Chat", "messages": [{"id": 1, "type": "message", "date": "bad"}]}, "invalid date"),
    ],
)
def test_rejects_malformed_exports(tmp_path: Path, payload: object, error: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TelegramExportError, match=error):
        load_telegram_export(path)


def test_wraps_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(TelegramExportError, match="invalid JSON"):
        load_telegram_export(path)

