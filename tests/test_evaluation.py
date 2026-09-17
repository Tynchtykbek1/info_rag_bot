import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import numpy as np

from messina_info.evaluation import (
    RetrievalCase,
    load_retrieval_cases,
    validate_retrieval_cases,
    evaluate_retrieval,
)
from messina_info.retrieval import IndexManifest, RetrievalDocument, RetrievalIndex


def _case(**overrides: object) -> RetrievalCase:
    values = {
        "case_id": "synthetic_en_01",
        "query": "How can I complete this synthetic procedure?",
        "language": "en",
        "category": "procedure",
        "answerable": True,
        "relevant_message_ids": (10,),
        "preferred_message_id": 10,
        "time_sensitive": False,
        "notes": "Synthetic test fixture",
    }
    values.update(overrides)
    return RetrievalCase(**values)  # type: ignore[arg-type]


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(**overrides: object) -> dict[str, object]:
    case = _case(**overrides)
    return {
        "case_id": case.case_id,
        "query": case.query,
        "language": case.language,
        "category": case.category,
        "answerable": case.answerable,
        "relevant_message_ids": list(case.relevant_message_ids),
        "preferred_message_id": case.preferred_message_id,
        "time_sensitive": case.time_sensitive,
        "notes": case.notes,
    }


def test_load_retrieval_cases_builds_frozen_cases_and_tuples(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, [_record(query="  Una domanda realistica?  ", language="it")])

    cases = load_retrieval_cases(path)

    assert cases == [_case(query="  Una domanda realistica?  ", language="it")]
    assert cases[0].relevant_message_ids == (10,)
    with pytest.raises(FrozenInstanceError):
        cases[0].query = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "contents, message",
    [
        ('{"case_id":\n', "malformed JSON"),
        ("[]\n", "expected a JSON object"),
        ("\n", "blank lines are not allowed"),
    ],
)
def test_load_retrieval_cases_rejects_malformed_jsonl(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_retrieval_cases(path)


def test_load_retrieval_cases_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad-schema.jsonl"
    record = _record()
    record["relevant_message_ids"] = [True]
    _write_jsonl(path, [record])

    with pytest.raises(ValueError, match="array of integers"):
        load_retrieval_cases(path)


def test_validation_rejects_duplicate_ids_and_normalized_queries() -> None:
    cases = [
        _case(),
        _case(query="  HOW can I complete this synthetic procedure?  "),
    ]

    errors = validate_retrieval_cases(cases, {10})

    assert "synthetic_en_01: duplicate case_id" in errors
    assert (
        "synthetic_en_01: duplicate normalized query within language" in errors
    )


def test_duplicate_query_is_allowed_in_a_different_language() -> None:
    cases = [_case(), _case(case_id="synthetic_it_01", language="it")]

    assert validate_retrieval_cases(cases, {10}) == []


def test_validation_rejects_unsupported_language_and_missing_query() -> None:
    case = _case(language="de", query=" \t\n")

    errors = validate_retrieval_cases([case], {10})

    assert "synthetic_en_01: unsupported language 'de'" in errors
    assert "synthetic_en_01: missing query" in errors


def test_validation_rejects_answerability_id_mismatches() -> None:
    answerable = _case(relevant_message_ids=(), preferred_message_id=None)
    unanswerable = _case(
        case_id="synthetic_unanswerable_en_01",
        query="Is this deliberately unavailable?",
        answerable=False,
        relevant_message_ids=(10,),
        preferred_message_id=10,
    )

    errors = validate_retrieval_cases([answerable, unanswerable], {10})

    assert "synthetic_en_01: answerable case has no relevant message IDs" in errors
    assert "synthetic_en_01: answerable case has no preferred message ID" in errors
    assert (
        "synthetic_unanswerable_en_01: unanswerable case has relevant message IDs"
        in errors
    )
    assert (
        "synthetic_unanswerable_en_01: unanswerable case has a preferred message ID"
        in errors
    )


def test_validation_rejects_preferred_id_outside_relevant_ids() -> None:
    errors = validate_retrieval_cases([_case(preferred_message_id=11)], {10, 11})

    assert errors == ["synthetic_en_01: preferred message ID is not relevant"]


def test_validation_reports_every_message_id_absent_from_database() -> None:
    case = replace(
        _case(),
        relevant_message_ids=(10, 20, 30),
        preferred_message_id=10,
    )

    errors = validate_retrieval_cases([case], {10})

    assert errors == [
        "synthetic_en_01: message ID 20 is absent from database",
        "synthetic_en_01: message ID 30 is absent from database",
    ]


class _EvaluationProvider:
    model_name = "fake-e5"
    dimension = 2

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "deadline" in text else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def _evaluation_index() -> RetrievalIndex:
    documents = [
        RetrievalDocument(
            section_key=f"100:{message_id}:0:en",
            channel_id=100,
            message_id=message_id,
            date=None,
            language="en",
            section_index=0,
            source_url=f"https://example.test/{message_id}",
            text=f"Synthetic document {message_id}",
            fingerprint=str(message_id),
        )
        for message_id in (10, 20, 30)
    ]
    return RetrievalIndex(
        np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
        documents,
        IndexManifest(1, "fake-e5", 2, 3, "synthetic", "2024-01-01T00:00:00Z"),
        _EvaluationProvider(),
    )


def test_evaluation_computes_hit_at_k_mrr_and_languages() -> None:
    cases = [
        _case(query="deadline", relevant_message_ids=(10,), preferred_message_id=10),
        _case(
            case_id="synthetic_it_01",
            query="different topic",
            language="it",
            relevant_message_ids=(20,),
            preferred_message_id=20,
        ),
    ]

    report = evaluate_retrieval(_evaluation_index(), cases)

    assert report["overall"] == {
        "checked_cases": 2,
        "hit_at_1": 0.5,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "mrr_at_5": 0.75,
    }
    assert report["by_language"]["en"]["hit_at_1"] == 1.0
    assert report["by_language"]["it"]["mrr_at_5"] == 0.5
    assert report["by_language"]["ru"]["checked_cases"] == 0


def test_unanswerable_cases_are_separate_and_optional_threshold_is_reported() -> None:
    case = _case(
        answerable=False,
        relevant_message_ids=(),
        preferred_message_id=None,
        query="different topic",
    )

    without_threshold = evaluate_retrieval(_evaluation_index(), [case])
    with_threshold = evaluate_retrieval(_evaluation_index(), [case], min_score=1.1)

    assert without_threshold["overall"]["checked_cases"] == 0
    assert without_threshold["unanswerable"]["rejection_accuracy"] is None
    assert with_threshold["unanswerable"]["rejection_accuracy"] == 1.0
    assert with_threshold["unanswerable"]["cases"][0]["rejected"] is True
