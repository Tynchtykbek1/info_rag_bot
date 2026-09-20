"""Bounded, structured interpretation of uncertain messages."""

from __future__ import annotations

import json
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .conversations import ConversationMessage
from .llm import GeminiProvider, LLMProviderError
from .message_routing import MessageRoute
from .rag import Language


class Interpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: MessageRoute
    standalone_query: str = Field(strict=True, min_length=1, max_length=1200)
    reason: str = Field(strict=True, min_length=1, max_length=300)


class MessageInterpreter(Protocol):
    def interpret(
        self, message: str, language: Language,
        history: Sequence[ConversationMessage],
    ) -> Interpretation: ...


SYSTEM_INSTRUCTION = """Classify the current user message as SMALL_TALK, DOMAIN_QUERY,
OUT_OF_DOMAIN, or UNCLEAR. Domain: University of Messina, ERSU, and student life
in Messina. OUT_OF_DOMAIN means a clear unrelated request; UNCLEAR means its
meaning cannot be resolved. Correct obvious typos and slang. For DOMAIN_QUERY,
rewrite ambiguous follow-ups as a standalone question using the bounded history.
Use assistant history only to resolve references, never as factual evidence.
Do not answer, invent facts, or create citations. Return only the JSON schema."""


class GeminiMessageInterpreter:
    def __init__(self, provider: GeminiProvider, *, max_messages: int = 6,
                 max_history_chars: int = 1800) -> None:
        self.provider = provider
        self.max_messages = max_messages
        self.max_history_chars = max_history_chars

    def interpret(self, message: str, language: Language,
                  history: Sequence[ConversationMessage]) -> Interpretation:
        bounded: list[dict[str, str]] = []
        remaining = self.max_history_chars
        for item in reversed(history[-self.max_messages:]):
            content = item.content[:remaining]
            if not content:
                break
            bounded.append({"role": item.role, "content": content})
            remaining -= len(content)
        payload = json.dumps({"language": language, "history": bounded[::-1],
                              "current_message": message}, ensure_ascii=False)
        client = self.provider._load_client()
        from google.genai import types

        for attempt in range(self.provider.max_retries + 1):
            try:
                response = client.models.generate_content(  # type: ignore[attr-defined]
                    model=self.provider.model,
                    contents=payload,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_json_schema=Interpretation.model_json_schema(),
                        max_output_tokens=512,
                    ),
                )
                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    return Interpretation.model_validate(parsed)
                return Interpretation.model_validate_json(response.text)
            except Exception as exc:
                if self.provider._transient(exc) and attempt < self.provider.max_retries:
                    self.provider._sleep(0.25 * (2 ** attempt))
                    continue
                raise LLMProviderError("Gemini interpretation failed") from exc
        raise LLMProviderError("Gemini interpretation failed")
