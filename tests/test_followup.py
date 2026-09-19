from dataclasses import replace

import pytest

from messina_info import ConversationMessage, build_contextual_query


def _message(
    message_id: int,
    role: str,
    content: str,
) -> ConversationMessage:
    return ConversationMessage(
        id=message_id,
        conversation_id="conversation-1",
        role=role,  # type: ignore[arg-type]
        content=content,
        created_at=message_id,
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        build_contextual_query(query, [])


def test_no_history_returns_normalized_standalone_query() -> None:
    result = build_contextual_query("  When is the deadline?  ", [])

    assert result.original_query == "When is the deadline?"
    assert result.retrieval_query == result.original_query
    assert result.history_messages_used == ()
    assert result.contextualized is False


def test_assistant_only_history_is_excluded() -> None:
    result = build_contextual_query(
        "What documents are needed?",
        [_message(1, "assistant", "The deadline is August 18.")],
    )

    assert result.retrieval_query == result.original_query
    assert result.history_messages_used == ()
    assert result.contextualized is False


def test_one_previous_user_question_is_contextualized() -> None:
    result = build_contextual_query(
        "And what time?",
        [_message(10, "user", "When is the ERSU deadline?")],
    )

    assert result.retrieval_query == (
        "PREVIOUS USER CONTEXT:\nWhen is the ERSU deadline?\n\n"
        "CURRENT QUESTION:\nAnd what time?"
    )
    assert result.history_messages_used == (10,)
    assert result.contextualized is True


def test_mixed_turns_use_only_latest_user_messages_in_chronological_order() -> None:
    history = [
        _message(1, "user", "First user question"),
        _message(2, "assistant", "Generated first answer"),
        _message(3, "user", "Second user question"),
        _message(4, "assistant", "Generated second answer"),
        _message(5, "user", "Third user question"),
    ]

    result = build_contextual_query("Follow-up", history, max_user_messages=2)

    assert result.history_messages_used == (3, 5)
    assert "Second user question\nThird user question" in result.retrieval_query
    assert "First user question" not in result.retrieval_query
    assert "Generated" not in result.retrieval_query


def test_duplicate_current_last_user_message_is_not_repeated() -> None:
    history = [
        _message(1, "user", "Earlier topic"),
        _message(2, "assistant", "Generated answer"),
        _message(3, "user", "  Same current question  "),
    ]

    result = build_contextual_query("Same current question", history)

    assert result.history_messages_used == (1,)
    assert result.retrieval_query.count("Same current question") == 1


def test_max_chars_drops_old_messages_before_new_messages() -> None:
    history = [
        _message(1, "user", "oldest message that should be removed"),
        _message(2, "user", "newest message"),
    ]
    newest_only_length = len(
        "PREVIOUS USER CONTEXT:\nnewest message\n\nCURRENT QUESTION:\nCurrent"
    )

    result = build_contextual_query(
        "Current", history, max_user_messages=2, max_chars=newest_only_length
    )

    assert len(result.retrieval_query) <= newest_only_length
    assert result.history_messages_used == (2,)
    assert "oldest" not in result.retrieval_query
    assert "newest message" in result.retrieval_query


def test_history_ids_only_include_messages_that_fit() -> None:
    history = [
        _message(1, "user", "x" * 100),
        _message(2, "user", "also too long"),
    ]

    result = build_contextual_query("Current", history, max_chars=len("Current"))

    assert result.retrieval_query == "Current"
    assert result.history_messages_used == ()
    assert result.contextualized is False


def test_very_long_current_query_is_predictably_truncated() -> None:
    query = "абвгд" * 20

    result = build_contextual_query(
        query,
        [_message(1, "user", "Previous")],
        max_chars=17,
    )

    assert result.original_query == query
    assert result.retrieval_query == query[:17]
    assert len(result.retrieval_query) == 17
    assert result.history_messages_used == ()
    assert result.contextualized is False


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        ("Когда подача на общежитие?", "А точное время?"),
        ("When is the scholarship deadline?", "Which time zone?"),
        ("Quando scade la domanda?", "E a che ora?"),
    ],
)
def test_ru_en_it_unicode_is_preserved(previous: str, current: str) -> None:
    result = build_contextual_query(current, [_message(1, "user", previous)])

    assert previous in result.retrieval_query
    assert current in result.retrieval_query


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
def test_invalid_max_user_messages_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="max_user_messages"):
        build_contextual_query("Question", [], max_user_messages=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1200"])
def test_invalid_max_chars_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        build_contextual_query("Question", [], max_chars=value)  # type: ignore[arg-type]


def test_history_objects_are_not_modified() -> None:
    history = [
        _message(1, "user", " Previous question "),
        _message(2, "assistant", " Answer "),
    ]
    snapshots = [replace(message) for message in history]

    build_contextual_query(" Current question ", history)

    assert history == snapshots
