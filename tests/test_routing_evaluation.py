import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from messina_info.interpreter import Interpretation
from messina_info.message_routing import MessageRoute
from messina_info.routing_evaluation import (
    Action, Case, CaseResult, action_for_route, case_fingerprint, evaluate_case,
    evaluate_cases, load_cases, main, prepare_live_interpreter, summarize,
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
    assert len(cases) == 90
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


def test_calibrated_tags_follow_annotation_guide():
    cases = load_cases(DATASET)
    assert all(case.expected_action == Action.DIRECT_REPLY
               for case in cases if "profanity_only" in case.tags)
    assert all(case.expected_action == Action.ASK_CLARIFICATION
               for case in cases if "underspecified_domain" in case.tags)
    assert all(case.expected_action == Action.SEARCH_KNOWLEDGE
               for case in cases if "thanks_domain" in case.tags)
    assert sum("profanity_only" in case.tags for case in cases) == 3
    assert sum("underspecified_domain" in case.tags for case in cases) == 3
    assert sum("thanks_domain" in case.tags for case in cases) == 3


def test_case_fingerprint_is_stable_and_covers_mutable_annotations():
    original = case("fingerprint", "message", Action.SEARCH_KNOWLEDGE,
                    history=(("user", "previous"),))
    assert case_fingerprint(original) == case_fingerprint(original)
    variants = [
        replace(original, message="changed"),
        replace(original, history=(("user", "different"),)),
        replace(original, last_outcome="answered"),
        replace(original, tags=("new-tag",)),
        replace(original, expected_action=Action.OUT_OF_SCOPE),
    ]
    assert all(case_fingerprint(item) != case_fingerprint(original) for item in variants)
    assert len(case_fingerprint(original)) == 64


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


def test_report_counts_dev_and_holdout_separately():
    cases = [case("dev-ok"),
             replace(case("dev-bad", expected=Action.OUT_OF_SCOPE), split="dev"),
             replace(case("holdout-ok", language="it"), split="holdout")]
    results = [
        evaluate_case(cases[0], mode="local"),
        CaseResult("dev-bad", "en", "OUT_OF_SCOPE", "DIRECT_REPLY", True, 0, 0, 1),
        evaluate_case(cases[2], mode="local"),
    ]
    report = summarize(cases, results, run_id="run", resolved_model="model")
    assert report["run_id"] == "run" and report["resolved_model"] == "model"
    assert report["case_fingerprints"] == {
        item.id: case_fingerprint(item) for item in cases
    }
    assert report["by_split"]["dev"] == {
        "total_cases": 2, "selected_cases": 2, "completed_cases": 2, "accuracy": 0.5,
    }
    assert report["by_split"]["holdout"] == {
        "total_cases": 1, "selected_cases": 1, "completed_cases": 1, "accuracy": 1.0,
    }


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
    first = evaluate_cases(cases, mode="live", interpreter=interpreter, run_id="run-1", resolved_model="model-1",
                           offset=1, limit=1, records_path=records)
    assert len(first) == 1 and len(interpreter.calls) == 1
    assert first[0].provider_calls == 1
    assert first[0].run_id == "run-1" and first[0].resolved_model == "model-1"
    assert first[0].case_fingerprint == case_fingerprint(cases[1])
    again = evaluate_cases(cases, mode="live", interpreter=interpreter, run_id="run-1", resolved_model="model-1",
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
    results = evaluate_cases(cases, mode="live", interpreter=fake, run_id="run-1", resolved_model="model-1", deferred_only=True,
                             offset=1, limit=2, records_path=records)
    assert [result.id for result in results] == ["deferred-2", "deferred-3"]
    assert all(result.interpreter_calls == result.provider_calls == 1 for result in results)
    assert len(fake.calls) == 2
    again = evaluate_cases(cases, mode="live", interpreter=fake, run_id="run-1", resolved_model="model-1", deferred_only=True,
                           offset=1, limit=2, records_path=records)
    assert again == results and len(fake.calls) == 2


def test_provider_error_is_retryable_and_not_scored_as_wrong(tmp_path):
    cases = [case("a", "how do I bake bread?", Action.OUT_OF_SCOPE)]
    records = tmp_path / "records.jsonl"
    error = evaluate_cases(cases, mode="live", interpreter=FakeInterpreter(fail=True), run_id="run-1", resolved_model="model-1",
                           records_path=records)
    assert error[0].predicted_action == "PROVIDER_ERROR"
    assert summarize(cases, error)["overall_accuracy"] is None
    recovered = evaluate_cases(cases, mode="live", interpreter=FakeInterpreter(), run_id="run-1", resolved_model="model-1",
                               records_path=records)
    assert recovered[0].predicted_action == "OUT_OF_SCOPE"
    assert len(records.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("change,match", [
    ("fingerprint", "fingerprint"), ("run", "run_id"), ("model", "model"),
])
def test_resume_rejects_changed_identity(tmp_path, change, match):
    original = case("resume", "how do I bake bread?", Action.OUT_OF_SCOPE)
    records = tmp_path / "records.jsonl"
    evaluate_cases([original], mode="live", interpreter=FakeInterpreter(),
                   run_id="run-1", resolved_model="model-1", records_path=records)
    resumed_case = replace(original, message="changed question") if change == "fingerprint" else original
    run_id = "run-2" if change == "run" else "run-1"
    model = "model-2" if change == "model" else "model-1"
    with pytest.raises(ValueError, match=match):
        evaluate_cases([resumed_case], mode="live", interpreter=FakeInterpreter(),
                       run_id=run_id, resolved_model=model, records_path=records)


def test_resume_rejects_legacy_record_without_identity(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "id": "legacy", "language": "en", "expected_action": "OUT_OF_SCOPE",
        "predicted_action": "OUT_OF_SCOPE", "fast_path": False,
        "interpreter_calls": 1, "provider_calls": 1, "latency_ms": 1,
        "error": None, "invalid_schema": False,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        evaluate_cases([case("legacy", "how do I bake bread?", Action.OUT_OF_SCOPE)],
                       mode="live", interpreter=FakeInterpreter(), run_id="run-1",
                       resolved_model="model-1", records_path=records)


def test_invalid_structured_response_counted_without_exposing_output(tmp_path):
    class Bad:
        def interpret(self, *args):
            return {"intent": "UNKNOWN", "standalone_query": "x", "reason": "x"}
    results = evaluate_cases([case("x", "unrelated question?", Action.OUT_OF_SCOPE)],
                             mode="live", interpreter=Bad(), run_id="run-1", resolved_model="model-1", records_path=tmp_path / "records.jsonl")
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
        main(["--dataset", str(dataset), "--mode", "live", "--run-id", "run-1", "--dotenv",
              str(tmp_path / "missing.env"), "--records", str(records)])
    assert not records.exists()
    assert "GEMINI_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("run_args", [[], ["--run-id", " \t "]])
def test_live_cli_requires_run_id_before_loading_dotenv(tmp_path, monkeypatch, run_args):
    import messina_info.routing_evaluation as module
    dataset = write_cases(tmp_path, [row(message="how do I bake bread?",
                                         expected_action="OUT_OF_SCOPE")])
    monkeypatch.setattr(module, "load_local_dotenv", lambda *args: (_ for _ in ()).throw(
        AssertionError("dotenv loaded before run identity validation")))
    with pytest.raises(SystemExit):
        main(["--dataset", str(dataset), "--mode", "live",
              "--records", str(tmp_path / "records.jsonl"), *run_args])


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
    main(["--dataset", str(dataset), "--mode", "live", "--run-id", "run-1", "--deferred-only",
          "--dotenv", str(dotenv), "--records", str(records), "--report", str(report)])
    output = capsys.readouterr()
    assert secret not in output.out + output.err + report.read_text(encoding="utf-8")
    assert secret not in records.read_text(encoding="utf-8")
    assert json.loads(report.read_text(encoding="utf-8"))["resolved_model"] == "gemini-flash-lite-latest"
    assert json.loads(report.read_text(encoding="utf-8"))["provider_calls"] == 1


@pytest.mark.parametrize("run_id,model,match", [
    ("run-2", "model-1", "run_id mismatch"),
    ("run-1", "model-2", "model mismatch"),
])
def test_disjoint_batches_reject_different_run_identity(tmp_path, monkeypatch, run_id, model, match):
    import messina_info.routing_evaluation as module
    cases = [case("a", "how do I bake bread?"), case("b", "where is Paris?")]
    records = tmp_path / "records.jsonl"
    first = FakeInterpreter()
    evaluate_cases(cases, mode="live", interpreter=first, run_id="run-1",
                   resolved_model="model-1", offset=0, limit=1, records_path=records)
    assert len(first.calls) == 1
    before = records.read_bytes()
    fake = FakeInterpreter()
    monkeypatch.setattr(module, "prepare_live_interpreter", lambda: pytest.fail("provider prepared"))
    for interpreter in (fake, None):
        with pytest.raises(ValueError, match=match):
            evaluate_cases(cases, mode="live", interpreter=interpreter, run_id=run_id,
                           resolved_model=model, offset=1, limit=1, records_path=records)
        assert records.read_bytes() == before
        assert fake.calls == []


@pytest.mark.parametrize("problem,match", [
    ("unknown", "unknown case ID"),
    ("fingerprint", "fingerprint"),
    ("duplicate_run", "run_id mismatch"),
    ("duplicate_model", "model mismatch"),
    ("blank_run", "run_id mismatch"),
])
def test_all_record_lines_validated_before_new_cases(tmp_path, problem, match):
    cases = [case("a", "how do I bake bread?"), case("b", "where is Paris?")]
    records = tmp_path / "records.jsonl"
    evaluate_cases(cases, mode="live", interpreter=FakeInterpreter(), run_id="run-1",
                   resolved_model="model-1", limit=1, records_path=records)
    original = records.read_text(encoding="utf-8")
    changed = json.loads(original)
    if problem == "unknown":
        changed["id"] = "secret-unknown-id"
    elif problem == "fingerprint":
        changed["case_fingerprint"] = "secret-fingerprint"
    elif problem == "duplicate_model":
        changed["resolved_model"] = "secret-model"
    else:
        changed["run_id"] = " \t " if problem == "blank_run" else "secret-run"
    records.write_text(json.dumps(changed) + "\n" + (original if problem.startswith("duplicate") else ""),
                       encoding="utf-8")
    before = records.read_bytes()
    fake = FakeInterpreter()
    with pytest.raises(ValueError, match=match) as error:
        evaluate_cases(cases, mode="live", interpreter=fake, run_id="run-1",
                       resolved_model="model-1", offset=1, records_path=records)
    assert "secret" not in str(error.value)
    assert fake.calls == []
    assert records.read_bytes() == before


def test_final_aggregate_reuses_deferred_and_adds_fast_paths(tmp_path):
    cases = load_cases(DATASET)
    records = tmp_path / "records.jsonl"
    fake = FakeInterpreter()
    deferred = evaluate_cases(cases, mode="live", interpreter=fake, run_id="run-1",
                              resolved_model="model-1", deferred_only=True, records_path=records)
    calls = len(fake.calls)
    results = evaluate_cases(cases, mode="live", interpreter=fake, run_id="run-1",
                             resolved_model="model-1", records_path=records)
    assert len(fake.calls) == calls
    assert all(result in results for result in deferred)
    assert all(result.provider_calls == 0 for result in results if result.fast_path)
    report = summarize(cases, results)
    assert report["selected_cases"] == report["completed_cases"] == 90
    assert len(records.read_text(encoding="utf-8").splitlines()) == 90


def test_underspecified_domain_messages():
    cases = load_cases(DATASET)
    assert {c.language: c.message for c in cases if "underspecified_domain" in c.tags} == {
        "ru": "У меня вопрос про ERSU",
        "en": "I have a question about ERSU",
        "it": "Ho una domanda sull'ERSU",
    }
    assert not any("broad_domain" in c.tags for c in cases)
