import sqlite3
from pathlib import Path

import pytest

from messina_info import (
    append_message,
    clear_conversation,
    connect_database,
    get_or_create_conversation,
    get_recent_messages,
    initialize_database,
)


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "conversations.db"
    initialize_database(database)
    return database


def test_schema_is_idempotent_and_foreign_keys_are_enabled(tmp_path: Path) -> None:
    database = _database(tmp_path)

    initialize_database(database)

    with connect_database(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
    assert {"messages", "conversations", "conversation_messages"} <= tables
    assert "idx_conversation_messages_conversation_id_id" in indexes
    assert foreign_keys == 1


def test_conversation_creation_reuse_and_distinct_chats(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        first = get_or_create_conversation(
            connection, platform=" telegram ", external_chat_id=" 100 "
        )
        repeated = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="100"
        )
        other = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="101"
        )

    assert first.id == repeated.id
    assert first.platform == "telegram"
    assert first.external_chat_id == "100"
    assert other.id != first.id


def test_existing_conversation_updates_only_nonempty_metadata(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        original = get_or_create_conversation(
            connection,
            platform="telegram",
            external_chat_id="100",
            external_user_id="old-user",
            language="ru",
        )
        updated = get_or_create_conversation(
            connection,
            platform="telegram",
            external_chat_id="100",
            external_user_id=" new-user ",
            language="   ",
        )

    assert updated.id == original.id
    assert updated.external_user_id == "new-user"
    assert updated.language == "ru"
    assert updated.updated_at >= original.updated_at


@pytest.mark.parametrize(
    ("platform", "chat_id"),
    [("", "1"), ("   ", "1"), ("telegram", ""), ("telegram", "\n")],
)
def test_empty_identifiers_are_rejected(
    tmp_path: Path, platform: str, chat_id: str
) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        with pytest.raises(ValueError, match="must not be empty"):
            get_or_create_conversation(
                connection, platform=platform, external_chat_id=chat_id
            )


def test_user_and_assistant_messages_are_saved(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        conversation = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1"
        )
        user = append_message(
            connection,
            conversation_id=conversation.id,
            role="user",
            content="Question",
        )
        assistant = append_message(
            connection,
            conversation_id=conversation.id,
            role="assistant",
            content="Answer",
        )

    assert (user.role, user.content) == ("user", "Question")
    assert (assistant.role, assistant.content) == ("assistant", "Answer")
    assert assistant.id > user.id


def test_invalid_role_and_empty_content_are_rejected(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        conversation = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1"
        )
        with pytest.raises(ValueError, match="role"):
            append_message(
                connection,
                conversation_id=conversation.id,
                role="system",  # type: ignore[arg-type]
                content="content",
            )
        with pytest.raises(ValueError, match="content"):
            append_message(
                connection,
                conversation_id=conversation.id,
                role="user",
                content=" \n ",
            )


def test_recent_messages_are_limited_and_chronological(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        conversation = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1"
        )
        assert get_recent_messages(
            connection, conversation_id=conversation.id, limit=3
        ) == ()
        for content in ("one", "two", "three", "four"):
            append_message(
                connection,
                conversation_id=conversation.id,
                role="user",
                content=content,
            )
        recent = get_recent_messages(
            connection, conversation_id=conversation.id, limit=2
        )

    assert [message.content for message in recent] == ["three", "four"]
    assert [message.id for message in recent] == sorted(message.id for message in recent)


@pytest.mark.parametrize("limit", [0, -1, True])
def test_recent_limit_must_be_positive_integer(tmp_path: Path, limit: int) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        conversation = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1"
        )
        with pytest.raises(ValueError, match="positive integer"):
            get_recent_messages(
                connection, conversation_id=conversation.id, limit=limit
            )


def test_clear_removes_only_selected_conversation_messages(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        first = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1"
        )
        second = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="2"
        )
        append_message(connection, conversation_id=first.id, role="user", content="a")
        append_message(connection, conversation_id=first.id, role="assistant", content="b")
        append_message(connection, conversation_id=second.id, role="user", content="c")

        deleted = clear_conversation(connection, conversation_id=first.id)

        first_recent = get_recent_messages(connection, conversation_id=first.id, limit=10)
        second_recent = get_recent_messages(connection, conversation_id=second.id, limit=10)
        still_exists = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (first.id,)
        ).fetchone()
    assert deleted == 2
    assert first_recent == ()
    assert [message.content for message in second_recent] == ["c"]
    assert still_exists is not None


def test_deleting_conversation_cascades_to_messages(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        conversation = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1"
        )
        append_message(
            connection, conversation_id=conversation.id, role="user", content="hello"
        )
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation.id,))
        count = connection.execute(
            "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = ?",
            (conversation.id,),
        ).fetchone()[0]
    assert count == 0


def test_unknown_conversation_is_rejected(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        with pytest.raises(ValueError, match="unknown conversation_id"):
            append_message(
                connection,
                conversation_id="missing",
                role="user",
                content="hello",
            )
        with pytest.raises(ValueError, match="unknown conversation_id"):
            clear_conversation(connection, conversation_id="missing")


def test_data_persists_after_reopening_database(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with connect_database(database) as connection:
        conversation = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1", language="it"
        )
        append_message(
            connection,
            conversation_id=conversation.id,
            role="assistant",
            content="Persisted",
        )

    with connect_database(database) as connection:
        reopened = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1"
        )
        messages = get_recent_messages(
            connection, conversation_id=reopened.id, limit=10
        )
    assert reopened.id == conversation.id
    assert reopened.language == "it"
    assert [message.content for message in messages] == ["Persisted"]


def test_repository_supports_default_sqlite_tuple_rows(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with sqlite3.connect(database) as connection:
        conversation = get_or_create_conversation(
            connection, platform="telegram", external_chat_id="1"
        )
        message = append_message(
            connection,
            conversation_id=conversation.id,
            role="user",
            content="Tuple row",
        )
        recent = get_recent_messages(
            connection, conversation_id=conversation.id, limit=1
        )

    assert message == recent[0]
