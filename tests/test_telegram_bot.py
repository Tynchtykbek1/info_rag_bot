import asyncio
import logging
from types import SimpleNamespace

import pytest
from telegram.constants import ChatType
from telegram.ext import CommandHandler, MessageHandler

from messina_info.chat import ChatResult
from messina_info.chat import ChatService
from messina_info.followup import ContextualQuery
from messina_info.rag import AnswerSource, RAGAnswer
from messina_info.telegram_bot import (
    build_application,
    format_chat_result,
    handle_question,
    help_command,
    language_command,
    reset_command,
    split_message,
    start_command,
)


TOKEN = "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"


class FakeMessage:
    def __init__(self, text="question", error=None):
        self.text, self.error, self.replies = text, error, []
    async def reply_text(self, text):
        if self.error: raise self.error
        self.replies.append(text)


class FakeService:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.reply_calls, self.reset_calls = result or _result(), error, [], []
    def reply(self, **kwargs):
        self.reply_calls.append(kwargs)
        if self.error: raise self.error
        return self.result
    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        if self.error: raise self.error
        return 2


class FakeBot:
    def __init__(self): self.actions = []
    async def send_chat_action(self, **kwargs): self.actions.append(kwargs)


def _result(answer="Answer", sources=(), reason="supported"):
    rag = RAGAnswer("fallback" if not sources and reason != "supported" else "answered",
                    answer, tuple(sources), "quality", tuple(s.message_id for s in sources), reason)
    return ChatResult("conversation", ContextualQuery("q", "q", (), False), rag)


def _objects(service=None, *, text="question", chat_type=ChatType.PRIVATE, language_code="en"):
    message = FakeMessage(text)
    update = SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=123, type=chat_type),
        effective_user=SimpleNamespace(id=456, language_code=language_code),
    )
    context = SimpleNamespace(
        args=[], chat_data={}, bot=FakeBot(),
        application=SimpleNamespace(bot_data={"chat_service": service or FakeService()}),
    )
    return update, context, message


def test_application_registers_handlers_without_network() -> None:
    service = FakeService()
    app = build_application(TOKEN, service)  # type: ignore[arg-type]
    handlers = [handler for group in app.handlers.values() for handler in group]
    commands = {next(iter(handler.commands)) for handler in handlers if isinstance(handler, CommandHandler)}
    assert commands == {"start", "help", "language", "reset"}
    assert sum(isinstance(handler, MessageHandler) for handler in handlers) == 1
    assert app.bot_data["chat_service"] is service
    assert app.update_processor.max_concurrent_updates == 1


@pytest.mark.parametrize("code,marker", [("ru-RU", "Messina Info"), ("en-US", "Messina Info"), ("it-IT", "Messina Info")])
def test_start_and_help_are_localized(code: str, marker: str) -> None:
    for handler in (start_command, help_command):
        update, context, message = _objects(language_code=code)
        asyncio.run(handler(update, context))
        assert marker in message.replies[0]
        assert "/reset" in message.replies[0]


def test_language_valid_and_invalid() -> None:
    update, context, message = _objects()
    context.args = ["it"]
    asyncio.run(language_command(update, context))
    assert context.chat_data["language"] == "it"
    context.args = ["de"]
    asyncio.run(language_command(update, context))
    assert "Usage" in message.replies[-1]


def test_reset_uses_to_thread(monkeypatch) -> None:
    import messina_info.telegram_bot as module
    service = FakeService(); update, context, message = _objects(service)
    calls = []
    async def fake_to_thread(function, **kwargs): calls.append((function, kwargs)); return function(**kwargs)
    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)
    asyncio.run(reset_command(update, context))
    assert calls and service.reset_calls[0]["external_chat_id"] == "123"
    assert service.reset_calls[0]["external_user_id"] == "456"
    assert message.replies[-1] == "Conversation history cleared."


def test_private_question_uses_to_thread_and_stops_typing(monkeypatch) -> None:
    import messina_info.telegram_bot as module
    service = FakeService(); update, context, message = _objects(service)
    used_to_thread, lifecycle = [], []
    async def fake_to_thread(function, **kwargs):
        used_to_thread.append(True); await asyncio.sleep(0); return function(**kwargs)
    async def fake_typing(*args):
        lifecycle.append("start")
        try: await asyncio.Event().wait()
        finally: lifecycle.append("stop")
    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(module, "_typing_loop", fake_typing)
    asyncio.run(handle_question(update, context))
    assert used_to_thread == [True] and lifecycle == ["start", "stop"]
    assert service.reply_calls[0]["external_chat_id"] == "123"
    assert service.reply_calls[0]["external_user_id"] == "456"
    assert message.replies == ["Answer"]


def test_group_message_never_calls_service() -> None:
    service = FakeService(); update, context, message = _objects(service, chat_type=ChatType.GROUP)
    asyncio.run(handle_question(update, context))
    assert service.reply_calls == []
    assert message.replies == ["Please use this bot in a private chat."]


def test_cyrillic_fallback_language_and_explicit_language_priority() -> None:
    service = FakeService(); update, context, _ = _objects(service, text="Когда срок?", language_code="de")
    asyncio.run(handle_question(update, context))
    assert service.reply_calls[-1]["language"] == "ru"
    context.chat_data["language"] = "it"
    asyncio.run(handle_question(update, context))
    assert service.reply_calls[-1]["language"] == "it"


def test_answer_format_deduplicates_sources_and_fallback_has_no_sources() -> None:
    first = AnswerSource("a", 1, 1, None, "en", "https://t.me/source/1")
    second = AnswerSource("b", 1, 2, None, "en", "https://t.me/source/1")
    text = format_chat_result(_result(sources=(first, second)), "en")
    assert text.count("https://t.me/source/1") == 1 and "Sources:" in text
    assert format_chat_result(_result("Fallback", reason="provider_error"), "en") == "Fallback"


def test_telegram_thanks_has_no_sources(tmp_path) -> None:
    class NoRAG:
        def answer_contextual(self, *args, **kwargs):
            raise AssertionError("RAG called")

    service = ChatService(tmp_path / "chat.db", NoRAG())  # type: ignore[arg-type]
    update, context, message = _objects(service, text="спасибо", language_code="ru")
    asyncio.run(handle_question(update, context))
    assert len(message.replies) == 1
    assert "Пожалуйста" in message.replies[0]
    assert "Sources" not in message.replies[0]


def test_splitting_never_exceeds_limit() -> None:
    parts = split_message("x" * 9001)
    assert "".join(parts) == "x" * 9001
    assert max(map(len, parts)) <= 4000


def test_unexpected_error_is_safe_and_typing_stops(monkeypatch, caplog) -> None:
    import messina_info.telegram_bot as module
    secret_question = "secret prompt and context"
    service = FakeService(error=RuntimeError("failure")); update, context, message = _objects(service, text=secret_question)
    stopped = []
    async def fake_typing(*args):
        try: await asyncio.Event().wait()
        finally: stopped.append(True)
    async def fake_to_thread(function, **kwargs): await asyncio.sleep(0); return function(**kwargs)
    monkeypatch.setattr(module, "_typing_loop", fake_typing)
    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)
    with caplog.at_level(logging.ERROR): asyncio.run(handle_question(update, context))
    assert stopped == [True]
    assert message.replies == ["A temporary error occurred. Please try again later."]
    assert secret_question not in caplog.text and TOKEN not in caplog.text
