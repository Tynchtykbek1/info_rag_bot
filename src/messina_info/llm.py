"""Structured LLM abstraction and lazy Google Gemini provider."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, ConfigDict

from .config import gemini_model, load_local_dotenv


GEMINI_MODEL_ENV = "MESSINA_GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class StructuredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answerable: bool
    answer: str | None
    cited_source_ids: list[str]
    reason: str


class LLMProvider(Protocol):
    def generate(self, system_instruction: str, user_prompt: str) -> StructuredResponse: ...


class LLMConfigurationError(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


class GeminiProvider:
    """Lazy structured-output Gemini client with bounded transient retries."""

    def __init__(
        self,
        model: str | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        dotenv_path: str | Path | None = None,
    ) -> None:
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("invalid Gemini timeout or retry count")
        if dotenv_path is not None:
            load_local_dotenv(dotenv_path)
        self.model = model or gemini_model(DEFAULT_GEMINI_MODEL)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._sleep = sleep
        self._client: object | None = None

    @property
    def loaded(self) -> bool:
        return self._client is not None

    def _load_client(self) -> object:
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise LLMConfigurationError("GEMINI_API_KEY is not configured")
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise LLMConfigurationError(
                    "install the 'rag' optional dependencies to use Gemini"
                ) from exc
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
            )
        return self._client

    @staticmethod
    def _transient(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        return isinstance(exc, (TimeoutError, ConnectionError)) or code in {
            408, 429, 500, 502, 503, 504
        }

    def generate(self, system_instruction: str, user_prompt: str) -> StructuredResponse:
        client = self._load_client()
        from google.genai import types

        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(  # type: ignore[attr-defined]
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=StructuredResponse,
                        max_output_tokens=1024,
                    ),
                )
                parsed = getattr(response, "parsed", None)
                if isinstance(parsed, StructuredResponse):
                    return parsed
                return StructuredResponse.model_validate_json(response.text)
            except Exception as exc:
                if self._transient(exc) and attempt < self.max_retries:
                    self._sleep(0.25 * (2**attempt))
                    continue
                if isinstance(exc, LLMConfigurationError):
                    raise
                raise LLMProviderError("Gemini request failed") from exc
        raise LLMProviderError("Gemini request failed")
