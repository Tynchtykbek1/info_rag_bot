"""Deterministic contextual retrieval-query construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .conversations import ConversationMessage


_PREVIOUS_HEADER = "PREVIOUS USER CONTEXT:"
_CURRENT_HEADER = "CURRENT QUESTION:"


@dataclass(frozen=True)
class ContextualQuery:
    original_query: str
    retrieval_query: str
    history_messages_used: tuple[int, ...]
    contextualized: bool


def _positive_integer(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _format_query(messages: Sequence[ConversationMessage], query: str) -> str:
    previous = "\n".join(message.content.strip() for message in messages)
    return (
        f"{_PREVIOUS_HEADER}\n{previous}\n\n"
        f"{_CURRENT_HEADER}\n{query}"
    )


def build_contextual_query(
    query: str,
    history: Sequence[ConversationMessage],
    *,
    max_user_messages: int = 2,
    max_chars: int = 1200,
) -> ContextualQuery:
    """Build a bounded retrieval query from current and previous user questions.

    Assistant answers are deliberately excluded because generated text may be wrong,
    and retrieval must not reinforce previously generated claims. Previous user
    questions are used only to restore the topic of the current question.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    _positive_integer(max_user_messages, "max_user_messages")
    _positive_integer(max_chars, "max_chars")
    original_query = query.strip()

    user_messages = [message for message in history if message.role == "user"]
    if user_messages and user_messages[-1].content.strip() == original_query:
        user_messages = user_messages[:-1]
    selected = user_messages[-max_user_messages:]

    while selected:
        candidate = _format_query(selected, original_query)
        if len(candidate) <= max_chars:
            return ContextualQuery(
                original_query=original_query,
                retrieval_query=candidate,
                history_messages_used=tuple(message.id for message in selected),
                contextualized=True,
            )
        selected = selected[1:]

    retrieval_query = original_query[:max_chars]
    return ContextualQuery(
        original_query=original_query,
        retrieval_query=retrieval_query,
        history_messages_used=(),
        contextualized=False,
    )
