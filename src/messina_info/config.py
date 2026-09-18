"""Local environment configuration helpers.

The real ``.env`` file is intentionally never read by repository tooling or
tests. Applications may opt into loading it through this helper; existing
process environment variables always win.
"""

from __future__ import annotations

from os import environ
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


def load_local_dotenv(path: str | Path = DEFAULT_DOTENV_PATH) -> bool:
    """Load local dotenv values without overriding real environment values."""

    return load_dotenv(dotenv_path=path, override=False)


def gemini_model(default: str) -> str:
    """Return the configured Gemini model without exposing any secret value."""

    return environ.get("MESSINA_GEMINI_MODEL") or default
