import json
from dataclasses import replace

import pytest

from messina_info.conversation_controller import ConversationAction, ConversationController
from messina_info.routing_evaluation import Action, Case, evaluate_cases, main, summarize
from tests.test_conversation_controller import FakeProvider, decision


def sample(case_id="annotation-only-id", message="Tell me more about that service"):
    return Case(case_id, "dev", "en", (("user", "A prior unrelated question"),
                ("assistant", "An unverified claim")), "insufficient_evidence", message,
                Action.SEARCH_KNOWLEDGE, ("annotation-only-tag",))


def forbid(*args, **kwargs):
    pytest.fail("legacy gate, dotenv or real provider used")


@pytest.mark.parametrize("action", list(ConversationAction))
def test_controller_engine_actions_payload_and_no_legacy_gate(tmp_path, monkeypatch, action):
    import messina_info.routing_evaluation as module
    monkeypatch.setattr(module, "route_message", forbid)
    monkeypatch.setattr(module, "needs_interpretation", forbid)
    provider = FakeProvider(decision(action.value))
    item = sample(message="  whazzup with this service?!  ")
    records = tmp_path / "controller.jsonl"
    results = evaluate_cases([item], engine="controller", mode="live",
                             controller=ConversationController(provider), run_id="controller-test",
                             resolved_model="fake-model", records_path=records, deferred_only=True)
    assert results[0].predicted_action == action.value
    assert results[0].provider_calls == results[0].interpreter_calls == len(provider.calls) == 1
    payload = json.loads(provider.calls[0]["contents"])
    assert payload == {"language": "en", "history": [
        {"role": role, "content": content} for role, content in item.history],
        "last_outcome": item.last_outcome, "current_message": item.message}
    request = provider.calls[0]["contents"] + provider.calls[0]["config"].system_instruction
    assert all(secret not in request for secret in (item.id, item.tags[0], "expected_action", "case_id", "tags"))
    assert json.loads(records.read_text())["engine"] == "controller"
    assert summarize([item], results)["engine"] == "controller"


def test_controller_local_cli_no_env_or_provider(tmp_path, monkeypatch, capsys):
    import messina_info.routing_evaluation as module
    monkeypatch.setattr(module, "load_local_dotenv", forbid)
    monkeypatch.setattr(module, "GeminiProvider", forbid)
    monkeypatch.setattr(module, "route_message", forbid)
    monkeypatch.setattr(module, "needs_interpretation", forbid)
    dataset = tmp_path / "cases.jsonl"
    rows = [dict(id=str(i), split="dev", language="en", history=[], last_outcome=None,
                 message=message, expected_action="DIRECT_REPLY", tags=[])
            for i, message in enumerate(["hello", "hello, is there a fee?"])]
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    report = tmp_path / "report.json"
    main(["--dataset", str(dataset), "--mode", "local", "--engine", "controller",
          "--report", str(report)])
    result = json.loads(report.read_text())
    assert result["engine"] == "controller"
    assert result["completed_cases"] == result["deferred_cases"] == 1
    assert result["provider_calls"] == 0
    main(["--dataset", str(dataset), "--mode", "local", "--engine", "controller",
          "--deferred-only", "--report", str(report)])
    assert json.loads(report.read_text())["selected_cases"] == 1


@pytest.mark.parametrize("error,invalid,retryable", [(None, True, False), (TimeoutError("secret"), False, True)])
def test_controller_errors_and_resume(tmp_path, error, invalid, retryable):
    provider = FakeProvider(decision(reply=" "), error=error)
    controller = ConversationController(provider)
    records = tmp_path / "records.jsonl"
    kwargs = dict(engine="controller", mode="live", controller=controller,
                  run_id="test", resolved_model="fake-model", records_path=records)
    first = evaluate_cases([sample()], **kwargs)
    assert len(provider.calls) == 1
    assert first[0].predicted_action == "PROVIDER_ERROR"
    assert first[0].invalid_schema is invalid
    assert first[0].retryable_error is retryable
    before = records.read_bytes()
    evaluate_cases([sample()], **kwargs)
    assert len(provider.calls) == (2 if retryable else 1)
    if not retryable:
        assert records.read_bytes() == before
    assert "secret" not in records.read_text()


@pytest.mark.parametrize("source,target,omit_engine", [
    ("legacy", "controller", False), ("controller", "legacy", False),
    ("legacy", "controller", True), ("legacy", "legacy", True),
])
def test_engine_resume_compatibility(tmp_path, monkeypatch, source, target, omit_engine):
    import messina_info.routing_evaluation as module
    cases = [sample("first", "hello"), sample("second", "thanks")]
    records = tmp_path / "records.jsonl"
    evaluate_cases(cases, engine=source, mode="live", interpreter=object(),
                   controller=ConversationController(FakeProvider()), run_id="same-run",
                   resolved_model="same-model", records_path=records, limit=1)
    if omit_engine:
        data = json.loads(records.read_text())
        del data["engine"]
        records.write_text(json.dumps(data) + "\n")
    before = records.read_bytes()
    monkeypatch.setattr(module, "prepare_live_interpreter", forbid)
    monkeypatch.setattr(module, "prepare_live_controller", forbid)
    if source != target:
        with pytest.raises(ValueError, match="engine mismatch"):
            evaluate_cases(cases, engine=target, mode="live", run_id="same-run",
                           resolved_model="same-model", records_path=records, offset=1)
        assert records.read_bytes() == before
    else:
        results = evaluate_cases(cases, engine=target, mode="live", interpreter=object(),
                                 run_id="same-run", resolved_model="same-model", records_path=records)
        assert len(results) == 2 and all(r.engine == "legacy" for r in results)


def test_controller_final_aggregate_no_additional_calls(tmp_path):
    cases = [sample("fast", "hello"), sample("deferred")]
    provider = FakeProvider(decision("EXPLAIN_PREVIOUS"))
    kwargs = dict(engine="controller", mode="live", controller=ConversationController(provider),
                  run_id="test", resolved_model="fake-model", records_path=tmp_path / "records.jsonl")
    evaluate_cases(cases, deferred_only=True, **kwargs)
    results = evaluate_cases(cases, **kwargs)
    assert len(provider.calls) == 1
    assert [r.predicted_action for r in results] == ["DIRECT_REPLY", "EXPLAIN_PREVIOUS"]
    assert summarize(cases, results)["completed_cases"] == 2
    with pytest.raises(ValueError, match="engine mismatch"):
        summarize(cases, [results[0], replace(results[1], engine="legacy")])


def test_controller_configuration_failure_no_provider_call(tmp_path):
    from messina_info.llm import LLMConfigurationError

    class Unconfigured(FakeProvider):
        def _load_client(self):
            raise LLMConfigurationError("secret configuration")

    provider = Unconfigured()
    results = evaluate_cases([sample()], engine="controller", mode="live",
                             controller=ConversationController(provider), run_id="test",
                             resolved_model="fake-model", records_path=tmp_path / "records.jsonl")
    assert results[0].provider_calls == 0
    assert not results[0].retryable_error
    assert provider.calls == []


def test_controller_live_cli_uses_selected_engine_with_fake_provider(tmp_path, monkeypatch, capsys):
    import messina_info.routing_evaluation as module
    provider = FakeProvider(decision("EXPLAIN_PREVIOUS"))
    monkeypatch.setattr(module, "prepare_live_interpreter", forbid)
    monkeypatch.setattr(module, "prepare_live_controller",
                        lambda path: (ConversationController(provider), "fake-model"))
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(dict(id="cli", split="dev", language="en", history=[],
                                      last_outcome="provider_error", message="What happened before?",
                                      expected_action="EXPLAIN_PREVIOUS", tags=[])))
    report = tmp_path / "report.json"
    main(["--dataset", str(dataset), "--mode", "live", "--engine", "controller",
          "--run-id", "controller-cli", "--records", str(tmp_path / "records.jsonl"),
          "--report", str(report)])
    result = json.loads(report.read_text())
    assert result["engine"] == "controller" and result["resolved_model"] == "fake-model"
    assert result["overall_accuracy"] == 1
    assert len(provider.calls) == 1
