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
from .reranking import (
    DEFAULT_ONNX_FILE,
    DEFAULT_RERANKER_MODEL,
    ONNXCrossEncoderReranker,
    rerank_search,
)


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
    if args.mode == "fast":
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
    else:
        reranker = _reranker(args)
        reranked = rerank_search(
            index,
            args.query,
            reranker,
            k=args.k,
            candidate_k=args.candidate_k,
            batch_size=args.reranker_batch_size,
        )
        output = [
            {
                "dense_rank": result.dense_rank,
                "semantic_score": result.semantic_score,
                "reranker_score": result.reranker_score,
                "reranked_rank": result.reranked_rank,
                **asdict(result.document),
            }
            for result in reranked
        ]
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    provider = _provider(args.model)
    index = load_retrieval_index(args.index, provider, expected_model=args.model)
    cases = load_retrieval_cases(args.dataset)
    report = evaluate_retrieval(
        index,
        cases,
        min_score=args.min_score,
        reranker=_reranker(args) if args.mode == "quality" else None,
        candidate_k=args.candidate_k,
        reranker_batch_size=args.reranker_batch_size,
    )
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
    _add_reranker_arguments(search)
    search.set_defaults(handler=_search)

    evaluate = subparsers.add_parser("evaluate", help="evaluate retrieval cases")
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--model", default=default_model)
    evaluate.add_argument("--min-score", type=float)
    _add_reranker_arguments(evaluate)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def _add_reranker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("fast", "quality"), default="fast")
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-backend", choices=("onnx", "torch"), default="onnx")
    parser.add_argument("--reranker-provider", default="CPUExecutionProvider")
    parser.add_argument("--reranker-file", default=DEFAULT_ONNX_FILE)
    parser.add_argument("--reranker-batch-size", type=int, default=8)


def _reranker(args: argparse.Namespace) -> ONNXCrossEncoderReranker:
    return ONNXCrossEncoderReranker(
        args.reranker_model,
        backend=args.reranker_backend,
        provider=args.reranker_provider,
        file_name=args.reranker_file,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
