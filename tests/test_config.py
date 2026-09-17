from pathlib import Path

from messina_info.config import load_local_dotenv


def test_dotenv_loads_values_without_overriding_environment(monkeypatch, tmp_path: Path) -> None:
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "GEMINI_API_KEY=file-only-key\nMESSINA_GEMINI_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("MESSINA_GEMINI_MODEL", "environment-model")

    assert load_local_dotenv(dotenv_file)
    assert __import__("os").environ["GEMINI_API_KEY"] == "file-only-key"
    assert __import__("os").environ["MESSINA_GEMINI_MODEL"] == "environment-model"


def test_missing_dotenv_is_safe(tmp_path: Path) -> None:
    assert not load_local_dotenv(tmp_path / "missing.env")
