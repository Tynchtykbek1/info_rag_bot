"""Local environment configuration helpers.

The real ``.env`` file is intentionally never read by repository tooling or
tests. Applications may opt into loading it through this helper; existing
process environment variables always win.
"""

from __future__ import annotations

from os import environ
from pathlib import Path

from dotenv import load_dotenv


def load_local_dotenv(path: str | Path | None = None) -> bool:
    """Load local dotenv values without overriding real environment values."""

    return load_dotenv(dotenv_path=path, override=False)


def gemini_model(default: str) -> str:
    """Return the configured Gemini model without exposing any secret value."""

    return environ.get("MESSINA_GEMINI_MODEL") or default
