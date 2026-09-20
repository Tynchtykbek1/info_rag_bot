"""Offline and explicitly requested live baseline for conversation routing."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from .conversations import ConversationMessage
from .config import DEFAULT_DOTENV_PATH, gemini_model, load_local_dotenv
from .interpreter import GeminiMessageInterpreter, Interpretation, MessageInterpreter
from .llm import DEFAULT_GEMINI_MODEL, GeminiProvider, LLMConfigurationError
from .message_routing import MessageRoute, needs_interpretation, route_message


class Action(str, Enum):
    SEARCH_KNOWLEDGE = "SEARCH_KNOWLEDGE"
    DIRECT_REPLY = "DIRECT_REPLY"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    EXPLAIN_PREVIOUS = "EXPLAIN_PREVIOUS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


ROUTE_ACTION = {
    MessageRoute.DOMAIN_QUERY: Action.SEARCH_KNOWLEDGE,
    MessageRoute.SMALL_TALK: Action.DIRECT_REPLY,
    MessageRoute.UNCLEAR: Action.ASK_CLARIFICATION,
    MessageRoute.OUT_OF_DOMAIN: Action.OUT_OF_SCOPE,
}
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_PLACEHOLDERS = re.compile(r"\?{3,}")
_DAMAGED = re.compile(r"[\ufffd\ud800-\udfff\x00-\x1f]|[ÐÑ][\u0080-\u00bf]")


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    language: str
    history: tuple[tuple[str, str], ...]
    last_outcome: str | None
    message: str
    expected_action: Action
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CaseResult:
    id: str
    language: str
    expected_action: str
    predicted_action: str
    fast_path: bool
    interpreter_calls: int
    provider_calls: int
    latency_ms: float
    error: str | None = None
    invalid_schema: bool = False


def action_for_route(route: MessageRoute) -> Action:
    return ROUTE_ACTION[route]


def _case(data: Any, line_number: int) -> Case:
    if not isinstance(data, dict):
        raise ValueError(f"line {line_number}: case must be an object")
    required = {"id", "split", "language", "history", "last_outcome",
                "message", "expected_action", "tags"}
    if set(data) != required:
        raise ValueError(f"line {line_number}: invalid case fields")
    for field in ("id", "message"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ValueError(f"line {line_number}: invalid {field}")
    if not isinstance(data["split"], str) or data["split"] not in {"dev", "holdout"}:
        raise ValueError(f"line {line_number}: unknown split")
    if not isinstance(data["language"], str) or data["language"] not in {"ru", "en", "it"}:
        raise ValueError(f"line {line_number}: unknown language")
    try:
        expected = Action(data["expected_action"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"line {line_number}: unknown expected action") from exc
    if data["last_outcome"] is not None and (
        not isinstance(data["last_outcome"], str) or data["last_outcome"] not in {
        "answered", "insufficient_evidence", "unsupported_factual_claim", "provider_error"
    }):
        raise ValueError(f"line {line_number}: unknown last outcome")
    history = data["history"]
    if not isinstance(history, list):
        raise ValueError(f"line {line_number}: history must be a list")
    items: list[tuple[str, str]] = []
    for item in history:
        if (not isinstance(item, dict) or set(item) != {"role", "content"}
                or item["role"] not in {"user", "assistant"}
                or not isinstance(item["content"], str) or not item["content"].strip()):
            raise ValueError(f"line {line_number}: invalid history message")
        items.append((item["role"], item["content"]))
    if not isinstance(data["tags"], list) or any(
        not isinstance(tag, str) or not tag.strip() for tag in data["tags"]
    ):
        raise ValueError(f"line {line_number}: invalid tags")
    texts = [data["message"], *(content for _, content in items)]
    if any(_DAMAGED.search(text) for text in texts):
        raise ValueError(f"line {line_number}: damaged Unicode text")
    if _PLACEHOLDERS.search(data["message"]) and "unclear" not in data["tags"]:
        raise ValueError(f"line {line_number}: placeholder question marks require unclear tag")
    if (data["language"] == "ru" and not {"unclear", "multilingual"}.intersection(data["tags"])
            and not _CYRILLIC.search(data["message"])):
        raise ValueError(f"line {line_number}: Russian case lacks Cyrillic")
    return Case(data["id"], data["split"], data["language"], tuple(items),
                data["last_outcome"], data["message"], expected, tuple(data["tags"]))


def load_cases(path: str | Path) -> tuple[Case, ...]:
    cases: list[Case] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            case = _case(data, line_number)
            if case.id in seen:
                raise ValueError(f"line {line_number}: duplicate case ID {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("dataset is empty")
    return tuple(cases)


def _history(case: Case) -> tuple[ConversationMessage, ...]:
    return tuple(ConversationMessage(i, "evaluation", role, content, 0)
                 for i, (role, content) in enumerate(case.history, 1))


def _invalid_schema(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ValidationError):
            return True
        current = current.__cause__
    return False


def evaluate_case(case: Case, *, mode: str,
                  interpreter: MessageInterpreter | None = None) -> CaseResult:
    started = time.perf_counter()
    routed = route_message(case.message, case.language)  # type: ignore[arg-type]
    fast = not needs_interpretation(routed)
    provider_calls = 0
    if fast:
        prediction = action_for_route(routed.route).value
        calls = 0
        error = None
        invalid = False
    elif mode == "local":
        prediction, calls, error, invalid = "DEFERRED", 0, None, False
    elif mode == "live":
        if interpreter is None:
            raise ValueError("live mode requires an interpreter")
        calls = 1
        provider_calls = 1
        try:
            parsed = interpreter.interpret(routed.normalized, case.language, _history(case))  # type: ignore[arg-type]
            validated = Interpretation.model_validate(parsed)
            prediction = action_for_route(validated.intent).value
            error, invalid = None, False
        except Exception as exc:
            prediction, error, invalid = "PROVIDER_ERROR", type(exc).__name__, _invalid_schema(exc)
            if isinstance(exc, LLMConfigurationError):
                provider_calls = 0
    else:
        raise ValueError("mode must be local or live")
    return CaseResult(case.id, case.language, case.expected_action.value, prediction,
                      fast, calls, provider_calls, (time.perf_counter() - started) * 1000,
                      error, invalid)


def _read_records(path: Path) -> dict[str, CaseResult]:
    if not path.exists():
        return {}
    records: dict[str, CaseResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = CaseResult(**json.loads(line))
            records[record.id] = record
    return records


def prepare_live_interpreter(
    dotenv_path: str | Path = DEFAULT_DOTENV_PATH,
) -> tuple[MessageInterpreter, str]:
    """Load only Gemini configuration; fail before any case if the key is absent."""
    try:
        load_local_dotenv(dotenv_path)
    except Exception as exc:
        raise ValueError("cannot load live dotenv") from exc
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is required for live evaluation")
    model = gemini_model(DEFAULT_GEMINI_MODEL)
    return GeminiMessageInterpreter(GeminiProvider(model=model, max_retries=0)), model


def evaluate_cases(cases: Sequence[Case], *, mode: str,
                   interpreter: MessageInterpreter | None = None,
                   offset: int = 0, limit: int | None = None,
                   deferred_only: bool = False,
                   delay_seconds: float = 0.0,
                   records_path: str | Path | None = None) -> tuple[CaseResult, ...]:
    if mode not in {"local", "live"} or offset < 0 or limit is not None and limit < 0 or delay_seconds < 0:
        raise ValueError("invalid evaluation options")
    if mode == "live" and records_path is None:
        raise ValueError("live mode requires records_path for resume")
    if mode == "local" and records_path is not None:
        raise ValueError("local mode uses --report, not --records")
    eligible = [case for case in cases if needs_interpretation(
        route_message(case.message, case.language)  # type: ignore[arg-type]
    )] if deferred_only else list(cases)
    selected = eligible[offset:offset + limit if limit is not None else None]
    path = Path(records_path) if records_path is not None else None
    previous = _read_records(path) if path is not None else {}
    if mode == "live" and interpreter is None:
        # One SDK request per deferred case: retries belong to a later, explicit run.
        interpreter, _ = prepare_live_interpreter()
    results: list[CaseResult] = []
    for case in selected:
        prior = previous.get(case.id)
        if prior is not None and (
            prior.language != case.language or prior.expected_action != case.expected_action.value
        ):
            raise ValueError(f"resumed case metadata changed: {case.id}")
        if prior is not None and prior.predicted_action != "PROVIDER_ERROR":
            results.append(prior)
            continue
        if mode == "live" and delay_seconds and results:
            time.sleep(delay_seconds)
        result = evaluate_case(case, mode=mode, interpreter=interpreter)
        results.append(result)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return tuple(results)


def summarize(cases: Sequence[Case], results: Sequence[CaseResult]) -> dict[str, Any]:
    completed = [r for r in results if r.predicted_action not in {"DEFERRED", "PROVIDER_ERROR"}]
    attempted = [r for r in results if r.predicted_action != "DEFERRED"]
    correct = sum(r.predicted_action == r.expected_action for r in completed)

    def accuracy(items: Sequence[CaseResult]) -> float | None:
        return sum(r.predicted_action == r.expected_action for r in items) / len(items) if items else None

    latency = sorted(r.latency_ms for r in attempted)
    matrix = {action.value: {predicted: 0 for predicted in
              [*(a.value for a in Action), "DEFERRED", "PROVIDER_ERROR"]} for action in Action}
    for result in results:
        matrix[result.expected_action][result.predicted_action] += 1
    return {
        "total_cases": len(cases), "selected_cases": len(results),
        "completed_cases": len(completed),
        "deferred_cases": sum(r.predicted_action == "DEFERRED" for r in results),
        "provider_errors": sum(r.predicted_action == "PROVIDER_ERROR" for r in results),
        "invalid_schema_count": sum(r.invalid_schema for r in results),
        "overall_accuracy": correct / len(completed) if completed else None,
        "fast_path_accuracy": accuracy([r for r in completed if r.fast_path]),
        "accuracy_by_language": {language: accuracy([r for r in completed if r.language == language])
                                 for language in ("ru", "en", "it")},
        "accuracy_by_expected_action": {action.value: accuracy([
            r for r in completed if r.expected_action == action.value]) for action in Action},
        "confusion_matrix": matrix,
        "fast_path_coverage": sum(r.fast_path for r in results) / len(results) if results else None,
        "interpreter_call_rate": sum(r.interpreter_calls for r in results) / len(results) if results else None,
        "interpreter_deferred_rate": sum(not r.fast_path for r in results) / len(results) if results else None,
        "interpreter_candidates": sum(not r.fast_path for r in results),
        "provider_calls": sum(r.provider_calls for r in results),
        "median_latency_ms": statistics.median(latency) if latency else None,
        "p95_latency_ms": latency[max(0, int(0.95 * len(latency) + 0.999999) - 1)] if latency else None,
        "potential_wrong_bypass_ids": [r.id for r in results if r.fast_path and
                                        r.predicted_action != r.expected_action],
        "decision_distribution_by_language": {
            language: {decision: sum(r.language == language and r.predicted_action == decision
                                     for r in results) for decision in
                       [*(a.value for a in Action), "DEFERRED", "PROVIDER_ERROR"]}
            for language in ("ru", "en", "it")
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--mode", choices=("local", "live"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--deferred-only", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument("--records", type=Path, help="Resumable JSONL output (required for live)")
    parser.add_argument("--report", type=Path, help="Machine-readable summary JSON")
    args = parser.parse_args(argv)
    cases = load_cases(args.dataset)
    interpreter = None
    resolved_model = None
    if args.mode == "live":
        try:
            interpreter, resolved_model = prepare_live_interpreter(args.dotenv)
        except ValueError as exc:
            parser.error(str(exc))
    results = evaluate_cases(cases, mode=args.mode, offset=args.offset, limit=args.limit,
                             deferred_only=args.deferred_only,
                             delay_seconds=args.delay_seconds, records_path=args.records,
                             interpreter=interpreter)
    report = summarize(cases, results)
    report["resolved_model"] = resolved_model
    report["deferred_only"] = args.deferred_only
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "total_cases", "selected_cases", "completed_cases", "deferred_cases",
        "provider_errors", "overall_accuracy", "fast_path_accuracy",
        "fast_path_coverage", "interpreter_candidates", "interpreter_call_rate",
        "provider_calls", "median_latency_ms",
        "p95_latency_ms", "invalid_schema_count", "resolved_model",
        "deferred_only")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
