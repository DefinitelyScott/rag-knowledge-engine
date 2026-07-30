"""Command line interface for the RAG knowledge engine."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .answerer import ExtractiveAnswerer, OpenAIAnswerer
from .engine import EngineConfig, RAGEngine

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus")


def _build_engine(args) -> RAGEngine:
    config = EngineConfig(
        target_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        rrf_k=args.rrf_k,
        mmr_lambda=args.mmr,
    )
    return RAGEngine.from_corpus(args.corpus, config=config)


def _cmd_search(args) -> int:
    engine = _build_engine(args)
    results = engine.search(args.query, k=args.k, method=args.method)
    if not results:
        print("No matching passages.")
        return 0
    for rank, result in enumerate(results, start=1):
        print("{}. [{}] score={:.5f}".format(rank, result.doc_id, result.score))
        print("   {}".format(result.text[:300].replace("\n", " ")))
    print(
        "\nredundancy (mean pairwise cosine of the {} results): {:.3f}".format(
            len(results), engine.retriever.redundancy(results)
        )
    )
    return 0


def _cmd_ask(args) -> int:
    engine = _build_engine(args)
    answerer = ExtractiveAnswerer() if not args.llm else OpenAIAnswerer(model=args.model)
    answer = engine.answer(args.query, k=args.k, method=args.method, answerer=answerer)
    print(answer.formatted())
    return 0


def _cmd_stats(args) -> int:
    engine = _build_engine(args)
    for key, value in engine.stats().items():
        print("{:<20} {}".format(key, round(value, 2) if isinstance(value, float) else value))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragkb", description="Hybrid-retrieval RAG engine (no dependencies)."
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="corpus directory")
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--mmr",
        type=float,
        default=None,
        metavar="LAMBDA",
        help=(
            "re-rank results with Maximal Marginal Relevance; LAMBDA in [0, 1] "
            "trades relevance (1.0) against diversity (0.0)"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="show the retrieved passages")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--method", default="hybrid", choices=("hybrid", "bm25", "vector"))
    search.set_defaults(func=_cmd_search)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("query")
    ask.add_argument("-k", type=int, default=4)
    ask.add_argument("--method", default="hybrid", choices=("hybrid", "bm25", "vector"))
    ask.add_argument("--llm", action="store_true", help="use the OpenAI answerer")
    ask.add_argument("--model", default="gpt-4o-mini")
    ask.set_defaults(func=_cmd_ask)

    stats = sub.add_parser("stats", help="print index statistics")
    stats.set_defaults(func=_cmd_stats)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
