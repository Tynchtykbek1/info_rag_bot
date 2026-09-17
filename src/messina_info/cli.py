"""Command-line interface for building and evaluating retrieval indexes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_ENV,
    SentenceTransformerEmbeddingProvider,
)
from .evaluation import evaluate_retrieval, load_retrieval_cases
from .retrieval import build_retrieval_index, load_retrieval_index


def _provider(model: str) -> SentenceTransformerEmbeddingProvider:
    return SentenceTransformerEmbeddingProvider(model)


def _build(args: argparse.Namespace) -> int:
    index = build_retrieval_index(
        args.database, args.index, _provider(args.model), model_name=args.model
    )
    print(json.dumps(asdict(index.manifest), ensure_ascii=False, indent=2))
    return 0


def _search(args: argparse.Namespace) -> int:
    provider = _provider(args.model)
    index = load_retrieval_index(args.index, provider, expected_model=args.model)
    results = index.search(
        args.query,
        k=args.k,
        recency_weight=args.recency_weight,
        half_life_days=args.half_life_days,
    )
    output = [
        {
            "semantic_score": result.semantic_score,
            "recency_bonus": result.recency_bonus,
            "final_score": result.final_score,
            **asdict(result.document),
        }
        for result in results
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    provider = _provider(args.model)
    index = load_retrieval_index(args.index, provider, expected_model=args.model)
    cases = load_retrieval_cases(args.dataset)
    report = evaluate_retrieval(index, cases, min_score=args.min_score)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="messina-info")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_model = os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)

    build = subparsers.add_parser("build", help="build an exact vector index")
    build.add_argument("--database", type=Path, required=True)
    build.add_argument("--index", type=Path, required=True)
    build.add_argument("--model", default=default_model)
    build.set_defaults(handler=_build)

    search = subparsers.add_parser("search", help="search an existing index")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--model", default=default_model)
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--recency-weight", type=float, default=0.0)
    search.add_argument("--half-life-days", type=float, default=180.0)
    search.set_defaults(handler=_search)

    evaluate = subparsers.add_parser("evaluate", help="evaluate retrieval cases")
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model", default=default_model)
    evaluate.add_argument("--min-score", type=float)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
