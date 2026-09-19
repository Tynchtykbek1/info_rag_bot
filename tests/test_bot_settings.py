from pathlib import Path
from types import SimpleNamespace

import pytest

from messina_info.bot_settings import BotSettings, create_chat_service, load_bot_settings
from messina_info.embeddings import DEFAULT_EMBEDDING_MODEL
from messina_info.llm import DEFAULT_GEMINI_MODEL
from messina_info.reranking import DEFAULT_ONNX_FILE, DEFAULT_RERANKER_MODEL


TOKEN = "123456:secret-token-value"


def test_settings_defaults_and_secret_repr() -> None:
    settings = load_bot_settings({"TELEGRAM_BOT_TOKEN": TOKEN})

    assert settings.database_path == Path("messina.db")
    assert settings.index_path == Path(".retrieval-index")
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.gemini_timeout_seconds == 30.0
    assert settings.reranker_model == DEFAULT_RERANKER_MODEL
    assert settings.reranker_backend == "onnx"
    assert settings.reranker_provider == "CPUExecutionProvider"
    assert settings.reranker_file == DEFAULT_ONNX_FILE
    assert (settings.candidate_k, settings.reranker_batch_size) == (15, 8)
    assert TOKEN not in repr(settings)


def test_settings_overrides_are_parsed() -> None:
    settings = load_bot_settings({
        "TELEGRAM_BOT_TOKEN": TOKEN,
        "MESSINA_DATABASE_PATH": "data/chat.db",
        "MESSINA_INDEX_PATH": "index",
        "MESSINA_EMBEDDING_MODEL": "embedding",
        "MESSINA_GEMINI_MODEL": "gemini",
        "MESSINA_GEMINI_TIMEOUT_SECONDS": "12.5",
        "MESSINA_RERANKER_MODEL": "reranker",
        "MESSINA_RERANKER_BACKEND": "torch",
        "MESSINA_RERANKER_PROVIDER": "Provider",
        "MESSINA_RERANKER_FILE": "",
        "MESSINA_CANDIDATE_K": "20",
        "MESSINA_RERANKER_BATCH_SIZE": "4",
    })
    assert settings.database_path == Path("data/chat.db")
    assert settings.index_path == Path("index")
    assert settings.gemini_timeout_seconds == 12.5
    assert settings.reranker_file is None
    assert (settings.candidate_k, settings.reranker_batch_size) == (20, 4)


def test_blank_optional_strings_use_defaults_from_env_template() -> None:
    settings = load_bot_settings({
        "TELEGRAM_BOT_TOKEN": TOKEN,
        "MESSINA_GEMINI_MODEL": "",
        "MESSINA_DATABASE_PATH": " ",
    })
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.database_path == Path("messina.db")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MESSINA_CANDIDATE_K", "0"),
        ("MESSINA_CANDIDATE_K", "1.5"),
        ("MESSINA_RERANKER_BATCH_SIZE", "true"),
        ("MESSINA_GEMINI_TIMEOUT_SECONDS", "nan"),
        ("MESSINA_GEMINI_TIMEOUT_SECONDS", "-1"),
        ("MESSINA_RERANKER_BACKEND", "bad"),
    ],
)
def test_invalid_settings_are_rejected_without_token(name: str, value: str) -> None:
    with pytest.raises(ValueError) as exc:
        load_bot_settings({"TELEGRAM_BOT_TOKEN": TOKEN, name: value})
    assert TOKEN not in str(exc.value)


def test_missing_token_is_clear() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        load_bot_settings({})


def test_service_factory_creates_each_dependency_once(monkeypatch, tmp_path: Path) -> None:
    calls = {name: [] for name in ("embedding", "index", "reranker", "llm", "rag", "chat")}

    def construct(name):
        def factory(*args, **kwargs):
            value = SimpleNamespace(name=name)
            calls[name].append((args, kwargs, value))
            return value
        return factory

    import messina_info.bot_settings as module
    monkeypatch.setattr(module, "SentenceTransformerEmbeddingProvider", construct("embedding"))
    monkeypatch.setattr(module, "load_retrieval_index", construct("index"))
    monkeypatch.setattr(module, "ONNXCrossEncoderReranker", construct("reranker"))
    monkeypatch.setattr(module, "GeminiProvider", construct("llm"))
    monkeypatch.setattr(module, "RAGService", construct("rag"))
    monkeypatch.setattr(module, "ChatService", construct("chat"))
    settings = load_bot_settings({
        "TELEGRAM_BOT_TOKEN": TOKEN,
        "MESSINA_DATABASE_PATH": str(tmp_path / "chat.db"),
    })

    result = create_chat_service(settings)

    assert result is calls["chat"][0][2]
    assert all(len(items) == 1 for items in calls.values())
    assert calls["rag"][0][1]["candidate_k"] == 15
    assert calls["rag"][0][1]["reranker_batch_size"] == 8


def test_service_factory_reports_index_failure_without_token(monkeypatch) -> None:
    import messina_info.bot_settings as module
    monkeypatch.setattr(module, "SentenceTransformerEmbeddingProvider", lambda model: object())
    monkeypatch.setattr(module, "load_retrieval_index", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    settings = load_bot_settings({"TELEGRAM_BOT_TOKEN": TOKEN})
    with pytest.raises(RuntimeError, match="Cannot load retrieval index") as exc:
        create_chat_service(settings)
    assert TOKEN not in str(exc.value)
