"""Loading and validation helpers for retrieval evaluation cases."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .embeddings import EmbeddingProvider
from .retrieval import RetrievalIndex
from .reranking import Reranker, rerank_search


EvaluationLanguage = Literal["ru", "en", "it"]


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    language: EvaluationLanguage
    category: str
    answerable: bool
    relevant_message_ids: tuple[int, ...]
    preferred_message_id: int | None
    time_sensitive: bool
    notes: str


_FIELDS = {
    "case_id",
    "query",
    "language",
    "category",
    "answerable",
    "relevant_message_ids",
    "preferred_message_id",
    "time_sensitive",
    "notes",
}


def _schema_error(line_number: int, detail: str) -> ValueError:
    return ValueError(f"invalid retrieval case on line {line_number}: {detail}")


def _case_from_object(value: object, line_number: int) -> RetrievalCase:
    if not isinstance(value, dict):
        raise _schema_error(line_number, "expected a JSON object")

    missing = _FIELDS - value.keys()
    extra = value.keys() - _FIELDS
    if missing:
        raise _schema_error(line_number, f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise _schema_error(line_number, f"unexpected fields: {', '.join(sorted(extra))}")

    string_fields = ("case_id", "query", "language", "category", "notes")
    for field in string_fields:
        if not isinstance(value[field], str):
            raise _schema_error(line_number, f"{field} must be a string")
    for field in ("answerable", "time_sensitive"):
        if not isinstance(value[field], bool):
            raise _schema_error(line_number, f"{field} must be a boolean")

    relevant_ids = value["relevant_message_ids"]
    if not isinstance(relevant_ids, list) or any(
        not isinstance(message_id, int) or isinstance(message_id, bool)
        for message_id in relevant_ids
    ):
        raise _schema_error(line_number, "relevant_message_ids must be an array of integers")
    preferred_id = value["preferred_message_id"]
    if preferred_id is not None and (
        not isinstance(preferred_id, int) or isinstance(preferred_id, bool)
    ):
        raise _schema_error(line_number, "preferred_message_id must be an integer or null")

    return RetrievalCase(
        case_id=value["case_id"],
        query=value["query"],
        language=cast(EvaluationLanguage, value["language"]),
        category=value["category"],
        answerable=value["answerable"],
        relevant_message_ids=tuple(relevant_ids),
        preferred_message_id=preferred_id,
        time_sensitive=value["time_sensitive"],
        notes=value["notes"],
    )


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    """Load and validate JSONL retrieval cases."""

    cases: list[RetrievalCase] = []
    with Path(path).open(encoding="utf-8") as case_file:
        for line_number, line in enumerate(case_file, start=1):
            if not line.strip():
                raise _schema_error(line_number, "blank lines are not allowed")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _schema_error(line_number, f"malformed JSON ({exc.msg})") from exc
            cases.append(_case_from_object(value, line_number))
    return cases


def _normalized_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def validate_retrieval_cases(
    cases: Sequence[RetrievalCase],
    existing_message_ids: Collection[int],
) -> list[str]:
    """Return validation errors without modifying data."""

    errors: list[str] = []
    seen_case_ids: set[str] = set()
    seen_queries: set[tuple[str, str]] = set()
    known_ids = set(existing_message_ids)

    for index, case in enumerate(cases, start=1):
        label = case.case_id.strip() or f"case #{index}"
        if not case.case_id.strip():
            errors.append(f"{label}: missing case_id")
        elif case.case_id in seen_case_ids:
            errors.append(f"{label}: duplicate case_id")
        seen_case_ids.add(case.case_id)

        if case.language not in {"ru", "en", "it"}:
            errors.append(f"{label}: unsupported language {case.language!r}")

        normalized_query = _normalized_query(case.query)
        if not normalized_query:
            errors.append(f"{label}: missing query")
        else:
            query_key = (case.language, normalized_query)
            if query_key in seen_queries:
                errors.append(f"{label}: duplicate normalized query within language")
            seen_queries.add(query_key)

        if case.answerable and not case.relevant_message_ids:
            errors.append(f"{label}: answerable case has no relevant message IDs")
        if case.answerable and case.preferred_message_id is None:
            errors.append(f"{label}: answerable case has no preferred message ID")
        if not case.answerable and case.relevant_message_ids:
            errors.append(f"{label}: unanswerable case has relevant message IDs")
        if not case.answerable and case.preferred_message_id is not None:
            errors.append(f"{label}: unanswerable case has a preferred message ID")
        if (
            case.preferred_message_id is not None
            and case.preferred_message_id not in case.relevant_message_ids
        ):
            errors.append(f"{label}: preferred message ID is not relevant")

        for message_id in case.relevant_message_ids:
            if message_id not in known_ids:
                errors.append(f"{label}: message ID {message_id} is absent from database")

    return errors


def _empty_metrics() -> dict[str, int | float]:
    return {
        "checked_cases": 0,
        "hit_at_1": 0.0,
        "hit_at_3": 0.0,
        "hit_at_5": 0.0,
        "mrr_at_5": 0.0,
    }


def _finalize_metrics(totals: dict[str, int | float]) -> dict[str, int | float]:
    count = int(totals["checked_cases"])
    if count == 0:
        return totals
    return {
        "checked_cases": count,
        "hit_at_1": float(totals["hit_at_1"]) / count,
        "hit_at_3": float(totals["hit_at_3"]) / count,
        "hit_at_5": float(totals["hit_at_5"]) / count,
        "mrr_at_5": float(totals["mrr_at_5"]) / count,
    }


def evaluate_retrieval(
    index: RetrievalIndex,
    cases: Sequence[RetrievalCase],
    *,
    embedding_provider: EmbeddingProvider | None = None,
    min_score: float | None = None,
    reranker: Reranker | None = None,
    candidate_k: int = 15,
    reranker_batch_size: int = 8,
    deduplicate: bool = True,
) -> dict[str, object]:
    """Evaluate retrieval ranks and report unanswerable scores separately."""

    totals = _empty_metrics()
    language_totals = {language: _empty_metrics() for language in ("ru", "en", "it")}
    unanswerable: list[dict[str, object]] = []

    for case in cases:
        if reranker is None:
            results = index.search(
                case.query,
                k=5,
                embedding_provider=embedding_provider,
                recency_weight=0.0,
            )
        else:
            results = rerank_search(
                index,
                case.query,
                reranker,
                k=5,
                candidate_k=candidate_k,
                batch_size=reranker_batch_size,
                deduplicate=deduplicate,
                embedding_provider=embedding_provider,
            )
        if not case.answerable:
            max_score = results[0].semantic_score if results else None
            item: dict[str, object] = {
                "case_id": case.case_id,
                "language": case.language,
                "max_semantic_score": max_score,
            }
            if min_score is not None:
                item["rejected"] = max_score is None or max_score < min_score
            unanswerable.append(item)
            continue

        relevant = set(case.relevant_message_ids)
        rank = next(
            (
                position
                for position, result in enumerate(results, start=1)
                if result.document.message_id in relevant
            ),
            None,
        )
        for metrics in (totals, language_totals[case.language]):
            metrics["checked_cases"] = int(metrics["checked_cases"]) + 1
            if rank is not None and rank <= 5:
                metrics["mrr_at_5"] = float(metrics["mrr_at_5"]) + 1.0 / rank
                metrics["hit_at_5"] = float(metrics["hit_at_5"]) + 1.0
                if rank <= 3:
                    metrics["hit_at_3"] = float(metrics["hit_at_3"]) + 1.0
                if rank == 1:
                    metrics["hit_at_1"] = float(metrics["hit_at_1"]) + 1.0

    scores = [
        float(item["max_semantic_score"])
        for item in unanswerable
        if item["max_semantic_score"] is not None
    ]
    summary: dict[str, object] = {
        "count": len(unanswerable),
        "cases": unanswerable,
        "max_semantic_score_min": min(scores) if scores else None,
        "max_semantic_score_max": max(scores) if scores else None,
        "max_semantic_score_mean": sum(scores) / len(scores) if scores else None,
        "min_score": min_score,
        "rejection_accuracy": None,
    }
    if min_score is not None:
        rejected = sum(bool(item["rejected"]) for item in unanswerable)
        summary["rejection_accuracy"] = (
            rejected / len(unanswerable) if unanswerable else None
        )
    return {
        "overall": _finalize_metrics(totals),
        "by_language": {
            language: _finalize_metrics(metrics)
            for language, metrics in language_totals.items()
        },
        "unanswerable": summary,
    }
