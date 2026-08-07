"""Command line interface for the RAG knowledge engine."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .answerer import ExtractiveAnswerer, OpenAIAnswerer
from .engine import EngineConfig, RAGEngine, read_corpus
from .expansion import ExpansionConfig
from .text import tokenize

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus")


def _expansion_config(args) -> Optional[ExpansionConfig]:
    if args.expand is None:
        return None
    return ExpansionConfig(
        feedback_docs=args.expand_docs,
        feedback_terms=args.expand,
        original_weight=args.expand_weight,
    )


def _build_engine(args) -> RAGEngine:
    if getattr(args, "index", None):
        return _load_engine(args)
    config = EngineConfig(
        target_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        rrf_k=args.rrf_k if args.rrf_k is not None else 60,
        mmr_lambda=args.mmr,
        expansion=_expansion_config(args),
    )
    return RAGEngine.from_corpus(args.corpus, config=config)


def _load_engine(args) -> RAGEngine:
    """Load a saved index instead of rebuilding, verifying freshness if we can.

    The saved config keeps governing behaviour; only knobs the user passed
    explicitly on this invocation override it. Chunking flags cannot apply to
    an already-chunked index and are ignored. When the corpus directory is
    still present its fingerprint is checked so a stale index fails loudly;
    when it is absent the index is trusted as-is.
    """
    verify = read_corpus(args.corpus) if os.path.isdir(args.corpus) else None
    engine = RAGEngine.load(args.index, verify_corpus=verify)
    if args.rrf_k is not None:
        engine.retriever.rrf_k = engine.config.rrf_k = args.rrf_k
    if args.mmr is not None:
        engine.retriever.mmr_lambda = engine.config.mmr_lambda = args.mmr
    expansion = _expansion_config(args)
    if expansion is not None:
        engine.retriever.expansion = engine.config.expansion = expansion
    return engine


def _print_expansion(engine: RAGEngine, query: str, method: str) -> None:
    """Show which terms the feedback pass borrowed, heaviest first."""
    model = engine.retriever.expansion_model(query, method=method)
    if not model:
        return
    original = set(tokenize(query))
    added = [
        (term, weight) for term, weight in model.items() if term not in original
    ]
    if not added:
        return
    added.sort(key=lambda pair: (-pair[1], pair[0]))
    print(
        "expanded with: {}".format(
            ", ".join("{} ({:.3f})".format(term, weight) for term, weight in added)
        )
    )


def _cmd_search(args) -> int:
    engine = _build_engine(args)
    _print_expansion(engine, args.query, args.method)
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


def _cmd_index(args) -> int:
    import time

    started = time.perf_counter()
    engine = _build_engine(args)
    built = time.perf_counter()
    engine.save(args.out)
    print(
        "indexed {} documents into {} chunks in {:.3f}s -> {}".format(
            len(engine.documents),
            len(engine.chunks),
            built - started,
            args.out,
        )
    )
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
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=None,
        help="RRF constant (default 60; with --index, the saved value)",
    )
    parser.add_argument(
        "--index",
        default=None,
        metavar="PATH",
        help=(
            "load a saved index built by `ragkb index` instead of re-reading "
            "and re-indexing the corpus; chunking flags are ignored"
        ),
    )
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
    parser.add_argument(
        "--expand",
        type=int,
        default=None,
        metavar="TERMS",
        help=(
            "expand the query with TERMS terms borrowed from the top results of "
            "a first retrieval pass (pseudo-relevance feedback)"
        ),
    )
    parser.add_argument(
        "--expand-docs",
        type=int,
        default=5,
        metavar="N",
        help="how many first-pass results to treat as relevant (default 5)",
    )
    parser.add_argument(
        "--expand-weight",
        type=float,
        default=0.8,
        metavar="ALPHA",
        help=(
            "weight kept on the original query, in [0, 1]; 1.0 disables the "
            "borrowed terms entirely (default 0.8)"
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

    index = sub.add_parser(
        "index", help="build the index once and save it for fast startup"
    )
    index.add_argument(
        "--out", default="index.json", help="where to write the index (JSON)"
    )
    index.set_defaults(func=_cmd_index)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
