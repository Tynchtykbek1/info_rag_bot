import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from messina_info.interpreter import Interpretation
from messina_info.message_routing import MessageRoute
from messina_info.routing_evaluation import (
    Action, Case, CaseResult, action_for_route, evaluate_case, evaluate_cases,
    load_cases, main, prepare_live_interpreter, summarize,
)


DATASET = Path(__file__).resolve().parents[1] / "eval" / "conversation_routing_cases.jsonl"


def case(case_id="x", message="hello", expected=Action.DIRECT_REPLY,
         language="en", history=()):
    return Case(case_id, "dev", language, history, None, message, expected, ())


def write_cases(tmp_path, rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def row(**updates):
    result = {"id": "x", "split": "dev", "language": "en", "history": [],
              "last_outcome": None, "message": "hello",
              "expected_action": "DIRECT_REPLY", "tags": []}
    result.update(updates)
    return result


def test_dataset_parses_and_has_balanced_coverage():
    cases = load_cases(DATASET)
    assert len(cases) >= 90
    assert {language: sum(c.language == language for c in cases) for language in ("ru", "en", "it")} == {
        "ru": 30, "en": 30, "it": 30,
    }
    assert {action: sum(c.expected_action == action for c in cases) for action in Action} == {
        action: 18 for action in Action
    }
    assert {c.split for c in cases} == {"dev", "holdout"}
    tags = {tag for item in cases for tag in item.tags}
    assert {"typo", "prompt_injection", "unsupported_factual_claim", "provider_error",
            "fallback_explanation", "topic_switch", "slang", "profanity_only"} <= tags
    assert next(item.message for item in cases if item.id == "ru_search_knowledge_002") == (
        "Кагда дедлайн подачи документов в UniME?"
    )


@pytest.mark.parametrize("message,tags,language,match", [
    ("????? ?????? UniME?", ["typo"], "ru", "placeholder"),
    ("Only Latin text", ["explicit_domain"], "ru", "Cyrillic"),
    ("hello ???", ["greeting"], "en", "placeholder"),
    ("damaged \ufffd text", ["unclear"], "en", "damaged Unicode"),
    ("broken \ud800 text", ["unclear"], "en", "damaged Unicode"),
    ("bad \x00 text", ["unclear"], "en", "damaged Unicode"),
    ("Ð¿Ñ€Ð¸Ð²ÐµÑ‚", ["multilingual"], "ru", "damaged Unicode"),
])
def test_dataset_quality_rejects_corruption(tmp_path, message, tags, language, match):
    with pytest.raises(ValueError, match=match):
        load_cases(write_cases(tmp_path, [row(message=message, tags=tags, language=language)]))


def test_intentional_unclear_question_marks_are_allowed(tmp_path):
    result = load_cases(write_cases(tmp_path, [row(language="ru", message="???",
                                                   tags=["unclear"]) ]))
    assert result[0].message == "???"


def test_every_dataset_case_passes_quality_rules():
    assert len(load_cases(DATASET)) == 90


@pytest.mark.parametrize("changes,match", [
    ({"language": "de"}, "language"),
    ({"expected_action": "MYSTERY"}, "expected action"),
    ({"history": [{"role": "system", "content": "x"}]}, "history"),
    ({"split": "secret"}, "split"),
    ({"message": "  "}, "message"),
])
def test_invalid_case_is_rejected(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        load_cases(write_cases(tmp_path, [row(**changes)]))


def test_duplicate_ids_and_invalid_json_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(write_cases(tmp_path, [row(), row()]))
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_cases(bad)


@pytest.mark.parametrize("route,action", [
    (MessageRoute.DOMAIN_QUERY, Action.SEARCH_KNOWLEDGE),
    (MessageRoute.SMALL_TALK, Action.DIRECT_REPLY),
    (MessageRoute.UNCLEAR, Action.ASK_CLARIFICATION),
    (MessageRoute.OUT_OF_DOMAIN, Action.OUT_OF_SCOPE),
])
def test_route_mapping(route, action):
    assert action_for_route(route) == action
    assert Action.EXPLAIN_PREVIOUS not in set(action_for_route(item) for item in MessageRoute)


def test_local_fast_path_and_deferred_never_call_provider():
    class Forbidden:
        def interpret(self, *args):
            raise AssertionError("provider called")
    direct = evaluate_case(case(), mode="local", interpreter=Forbidden())
    deferred = evaluate_case(case("y", "what about that?", Action.ASK_CLARIFICATION),
                             mode="local", interpreter=Forbidden())
    assert (direct.predicted_action, direct.fast_path) == ("DIRECT_REPLY", True)
    assert (deferred.predicted_action, deferred.fast_path) == ("DEFERRED", False)
    assert deferred.interpreter_calls == deferred.provider_calls == 0


def test_metrics_exclude_deferred_and_errors_from_accuracy():
    cases = [case("a"), case("b", expected=Action.OUT_OF_SCOPE),
             case("c", expected=Action.SEARCH_KNOWLEDGE),
             case("d", expected=Action.ASK_CLARIFICATION)]
    results = [
        CaseResult("a", "en", "DIRECT_REPLY", "DIRECT_REPLY", True, 0, 0, 10),
        CaseResult("b", "en", "OUT_OF_SCOPE", "DIRECT_REPLY", True, 0, 0, 20),
        CaseResult("c", "it", "SEARCH_KNOWLEDGE", "DEFERRED", False, 0, 0, 0),
        CaseResult("d", "ru", "ASK_CLARIFICATION", "PROVIDER_ERROR", False, 1, 1, 100,
                   "RuntimeError", True),
    ]
    report = summarize(cases, results)
    assert report["completed_cases"] == 2 and report["deferred_cases"] == 1
    assert report["provider_errors"] == 1 and report["invalid_schema_count"] == 1
    assert report["overall_accuracy"] == 0.5
    assert report["accuracy_by_language"]["en"] == 0.5
    assert report["accuracy_by_language"]["ru"] is None
    assert report["accuracy_by_expected_action"]["DIRECT_REPLY"] == 1.0
    assert report["accuracy_by_expected_action"]["OUT_OF_SCOPE"] == 0.0
    assert report["confusion_matrix"]["SEARCH_KNOWLEDGE"]["DEFERRED"] == 1
    assert report["confusion_matrix"]["OUT_OF_SCOPE"]["DIRECT_REPLY"] == 1
    assert report["potential_wrong_bypass_ids"] == ["b"]
    assert report["median_latency_ms"] == 20
    assert report["p95_latency_ms"] == 100


class FakeInterpreter:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def interpret(self, message, language, history):
        self.calls.append((message, language, history))
        if self.fail:
            raise RuntimeError("synthetic provider failure")
        return Interpretation(intent=MessageRoute.OUT_OF_DOMAIN,
                              standalone_query=message, reason="synthetic")


def test_live_fake_provider_resume_and_batch(tmp_path):
    cases = [case("a", "hello"), case("b", "how do I bake bread?", Action.OUT_OF_SCOPE),
             case("c", "where is Paris?", Action.OUT_OF_SCOPE)]
    records = tmp_path / "records.jsonl"
    interpreter = FakeInterpreter()
    first = evaluate_cases(cases, mode="live", interpreter=interpreter,
                           offset=1, limit=1, records_path=records)
    assert len(first) == 1 and len(interpreter.calls) == 1
    assert first[0].provider_calls == 1
    again = evaluate_cases(cases, mode="live", interpreter=interpreter,
                           offset=1, limit=2, records_path=records)
    assert [r.id for r in again] == ["b", "c"]
    assert len(interpreter.calls) == 2
    assert len(records.read_text(encoding="utf-8").splitlines()) == 2


def test_deferred_only_filters_before_offset_and_calls_once(tmp_path):
    cases = [case("fast-1", "hello"),
             case("deferred-1", "how do I bake bread?", Action.OUT_OF_SCOPE),
             case("fast-2", "thanks"),
             case("deferred-2", "where is Paris?", Action.OUT_OF_SCOPE),
             case("deferred-3", "why?", Action.ASK_CLARIFICATION)]
    fake = FakeInterpreter()
    records = tmp_path / "records.jsonl"
    results = evaluate_cases(cases, mode="live", interpreter=fake, deferred_only=True,
                             offset=1, limit=2, records_path=records)
    assert [result.id for result in results] == ["deferred-2", "deferred-3"]
    assert all(result.interpreter_calls == result.provider_calls == 1 for result in results)
    assert len(fake.calls) == 2
    again = evaluate_cases(cases, mode="live", interpreter=fake, deferred_only=True,
                           offset=1, limit=2, records_path=records)
    assert again == results and len(fake.calls) == 2


def test_provider_error_is_retryable_and_not_scored_as_wrong(tmp_path):
    cases = [case("a", "how do I bake bread?", Action.OUT_OF_SCOPE)]
    records = tmp_path / "records.jsonl"
    error = evaluate_cases(cases, mode="live", interpreter=FakeInterpreter(fail=True),
                           records_path=records)
    assert error[0].predicted_action == "PROVIDER_ERROR"
    assert summarize(cases, error)["overall_accuracy"] is None
    recovered = evaluate_cases(cases, mode="live", interpreter=FakeInterpreter(),
                               records_path=records)
    assert recovered[0].predicted_action == "OUT_OF_SCOPE"
    assert len(records.read_text(encoding="utf-8").splitlines()) == 2


def test_invalid_structured_response_counted_without_exposing_output(tmp_path):
    class Bad:
        def interpret(self, *args):
            return {"intent": "UNKNOWN", "standalone_query": "x", "reason": "x"}
    results = evaluate_cases([case("x", "unrelated question?", Action.OUT_OF_SCOPE)],
                             mode="live", interpreter=Bad(), records_path=tmp_path / "records.jsonl")
    assert results[0].predicted_action == "PROVIDER_ERROR"
    assert results[0].invalid_schema


def test_runner_does_not_open_production_sqlite(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DB opened")))
    results = evaluate_cases([case()], mode="local")
    assert results[0].predicted_action == "DIRECT_REPLY"


def test_cli_local_writes_machine_readable_report_without_provider(tmp_path, capsys, monkeypatch):
    import messina_info.routing_evaluation as module
    monkeypatch.setattr(module, "GeminiProvider", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Gemini loaded")))
    dataset = write_cases(tmp_path, [row(), row(id="y", message="what about that?",
                                                 expected_action="ASK_CLARIFICATION")])
    report_path = tmp_path / "report.json"
    main(["--dataset", str(dataset), "--mode", "local", "--report", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    console = json.loads(capsys.readouterr().out)
    assert report["total_cases"] == 2 and report["deferred_cases"] == 1
    assert report["fast_path_accuracy"] == 1.0
    assert console["interpreter_candidates"] == 1


def test_local_cli_never_loads_dotenv(tmp_path, monkeypatch):
    import messina_info.routing_evaluation as module
    monkeypatch.setattr(module, "load_local_dotenv", lambda *args: (_ for _ in ()).throw(AssertionError("dotenv read")))
    dataset = write_cases(tmp_path, [row()])
    main(["--dataset", str(dataset), "--mode", "local", "--dotenv", str(tmp_path / "secret.env")])


def _isolated_environment(monkeypatch):
    import messina_info.config as config
    fresh = {}
    monkeypatch.setattr(os, "environ", fresh)
    monkeypatch.setattr(config, "environ", fresh)
    return fresh


def test_live_loads_selected_dotenv_and_resolves_model(tmp_path, monkeypatch):
    environment = _isolated_environment(monkeypatch)
    dotenv = tmp_path / "selected.env"
    dotenv.write_text("GEMINI_API_KEY=file-secret\nMESSINA_GEMINI_MODEL=gemini-flash-lite-latest\n",
                      encoding="utf-8")
    interpreter, model = prepare_live_interpreter(dotenv)
    assert model == "gemini-flash-lite-latest"
    assert interpreter.provider.model == model
    assert interpreter.provider.max_retries == 0
    assert environment["GEMINI_API_KEY"] == "file-secret"


def test_process_environment_wins_over_dotenv(tmp_path, monkeypatch):
    environment = _isolated_environment(monkeypatch)
    environment.update(GEMINI_API_KEY="process-secret", MESSINA_GEMINI_MODEL="process-model")
    dotenv = tmp_path / "selected.env"
    dotenv.write_text("GEMINI_API_KEY=file-secret\nMESSINA_GEMINI_MODEL=file-model\n",
                      encoding="utf-8")
    _, model = prepare_live_interpreter(dotenv)
    assert model == "process-model"
    assert environment["GEMINI_API_KEY"] == "process-secret"


def test_missing_key_stops_before_evaluation_or_records(tmp_path, monkeypatch, capsys):
    _isolated_environment(monkeypatch)
    dataset = write_cases(tmp_path, [row(message="how do I bake bread?",
                                         expected_action="OUT_OF_SCOPE")])
    records = tmp_path / "records.jsonl"
    with pytest.raises(SystemExit):
        main(["--dataset", str(dataset), "--mode", "live", "--dotenv",
              str(tmp_path / "missing.env"), "--records", str(records)])
    assert not records.exists()
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_live_cli_report_contains_model_without_secret(tmp_path, monkeypatch, capsys):
    import messina_info.routing_evaluation as module
    _isolated_environment(monkeypatch)
    dotenv = tmp_path / "selected.env"
    secret = "test-secret-never-print"
    dotenv.write_text(f"GEMINI_API_KEY={secret}\nMESSINA_GEMINI_MODEL=gemini-flash-lite-latest\n",
                      encoding="utf-8")
    dataset = write_cases(tmp_path, [row(message="how do I bake bread?",
                                         expected_action="OUT_OF_SCOPE")])
    class FakeGeminiInterpreter(FakeInterpreter):
        def __init__(self, provider):
            super().__init__()
    monkeypatch.setattr(module, "GeminiMessageInterpreter", FakeGeminiInterpreter)
    records, report = tmp_path / "records.jsonl", tmp_path / "report.json"
    main(["--dataset", str(dataset), "--mode", "live", "--deferred-only",
          "--dotenv", str(dotenv), "--records", str(records), "--report", str(report)])
    output = capsys.readouterr()
    assert secret not in output.out + output.err + report.read_text(encoding="utf-8")
    assert secret not in records.read_text(encoding="utf-8")
    assert json.loads(report.read_text(encoding="utf-8"))["resolved_model"] == "gemini-flash-lite-latest"
    assert json.loads(report.read_text(encoding="utf-8"))["provider_calls"] == 1
