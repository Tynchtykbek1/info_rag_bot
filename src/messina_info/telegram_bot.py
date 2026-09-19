"""Local long-polling Telegram Bot API application."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import cast

from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from .bot_settings import create_chat_service, load_bot_settings
from .chat import ChatResult, ChatService
from .config import load_local_dotenv
from .rag import Language


logger = logging.getLogger(__name__)
_PRIVATE_ONLY = "Please use this bot in a private chat."
_SUPPORTED_LANGUAGES = {"ru", "en", "it"}
_HELP = {
    "ru": "Я отвечаю по сообщениям Messina Info на русском, английском и итальянском. Можно задавать уточняющие вопросы. /language ru|en|it выбирает язык, /reset очищает историю. Ответы содержат ссылки на источники и могут быть неточными.",
    "en": "I answer from Messina Info messages in Russian, English, and Italian. You can ask follow-up questions. Use /language ru|en|it to choose a language and /reset to clear history. Answers include source links and may be inaccurate.",
    "it": "Rispondo usando i messaggi di Messina Info in russo, inglese e italiano. Puoi fare domande successive. Usa /language ru|en|it per scegliere la lingua e /reset per cancellare la cronologia. Le risposte includono link alle fonti e possono essere inesatte.",
}
_INVALID_TEXT = {"ru": "Пожалуйста, отправьте текстовый вопрос.", "en": "Please send a text question.", "it": "Invia una domanda di testo, per favore."}
_TEMPORARY_ERROR = {"ru": "Произошла временная ошибка. Попробуйте ещё раз позже.", "en": "A temporary error occurred. Please try again later.", "it": "Si è verificato un errore temporaneo. Riprova più tardi."}
_RESET = {"ru": "История очищена.", "en": "Conversation history cleared.", "it": "Cronologia della conversazione cancellata."}


def _language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Language:
    selected = context.chat_data.get("language")
    if selected in _SUPPORTED_LANGUAGES:
        return cast(Language, selected)
    user_code = (getattr(update.effective_user, "language_code", None) or "").lower()
    for language in ("ru", "en", "it"):
        if user_code.startswith(language):
            return cast(Language, language)
    text = getattr(update.effective_message, "text", None) or ""
    if any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in text):
        return "ru"
    return "en"


def _service(context: ContextTypes.DEFAULT_TYPE) -> ChatService:
    return cast(ChatService, context.application.bot_data["chat_service"])


async def _private(update: Update) -> bool:
    if update.effective_chat is not None and update.effective_chat.type == ChatType.PRIVATE:
        return True
    if update.effective_message is not None:
        await update.effective_message.reply_text(_PRIVATE_ONLY)
    return False


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _private(update):
        await update.effective_message.reply_text(_HELP[_language(update, context)])


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start_command(update, context)


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _private(update):
        return
    requested = context.args[0].lower() if len(context.args) == 1 else ""
    if requested not in _SUPPORTED_LANGUAGES:
        await update.effective_message.reply_text("Usage: /language ru|en|it")
        return
    context.chat_data["language"] = requested
    confirmations = {"ru": "Выбран русский язык.", "en": "English selected.", "it": "Lingua italiana selezionata."}
    await update.effective_message.reply_text(confirmations[requested])


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _private(update):
        return
    language = _language(update, context)
    try:
        await asyncio.to_thread(
            _service(context).reset,
            platform="telegram",
            external_chat_id=str(update.effective_chat.id),
            external_user_id=str(update.effective_user.id) if update.effective_user else None,
            language=language,
        )
    except Exception as exc:
        logger.error("Chat reset failed: %s", type(exc).__name__)
        await update.effective_message.reply_text(_TEMPORARY_ERROR[language])
        return
    await update.effective_message.reply_text(_RESET[language])


async def _typing_loop(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    while True:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(4)


def format_chat_result(result: ChatResult, language: Language) -> str:
    del language
    urls: list[str] = []
    for source in result.rag_answer.sources:
        if source.source_url and source.source_url not in urls:
            urls.append(source.source_url)
    if not urls:
        return result.rag_answer.answer
    return result.rag_answer.answer + "\n\nSources:\n" + "\n".join(f"• {url}" for url in urls)


def split_message(text: str, max_length: int = 4000) -> list[str]:
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    return [text[pos:pos + max_length] for pos in range(0, len(text), max_length)] if text else []


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _private(update):
        return
    language = _language(update, context)
    message = update.effective_message
    query = message.text if message is not None else None
    if not isinstance(query, str) or not query.strip():
        await message.reply_text(_INVALID_TEXT[language])
        return
    chat_id = update.effective_chat.id
    typing_task = asyncio.create_task(_typing_loop(context, chat_id))
    try:
        result = await asyncio.to_thread(
            _service(context).reply,
            platform="telegram",
            external_chat_id=str(chat_id),
            external_user_id=str(update.effective_user.id) if update.effective_user else None,
            query=query,
            language=language,
        )
    except Exception as exc:
        logger.error("Question handling failed: %s", type(exc).__name__)
        await message.reply_text(_TEMPORARY_ERROR[language])
        return
    finally:
        typing_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await typing_task
    for part in split_message(format_chat_result(result, language)):
        try:
            await message.reply_text(part)
        except Exception as exc:
            logger.error("Telegram send failed: %s", type(exc).__name__)
            break


def build_application(token: str, chat_service: ChatService) -> Application:
    if not isinstance(token, str) or not token.strip():
        raise ValueError("token must not be empty")
    app = ApplicationBuilder().token(token.strip()).concurrent_updates(False).build()
    app.bot_data["chat_service"] = chat_service
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    return app


def main() -> None:
    load_local_dotenv()
    settings = load_bot_settings()
    service = create_chat_service(settings)
    build_application(settings.telegram_token, service).run_polling()
