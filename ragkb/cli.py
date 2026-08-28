"""Command line interface for the RAG knowledge engine."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

from .abstain import AbstentionConfig
from .answerer import ExtractiveAnswerer, OpenAIAnswerer
from .attribution import attribute, best_chunk_for_doc
from .crossencoder import RerankConfig
from .engine import EngineConfig, RAGEngine, read_corpus
from .expansion import ExpansionConfig
from .retriever import resolve_fusion_weights
from .serve import DEFAULT_HOST, DEFAULT_PORT
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


def _rerank_config(args) -> Optional[RerankConfig]:
    if args.rerank is None:
        return None
    return RerankConfig(alpha=args.rerank)


def _fusion_weights(args) -> Optional[Dict[str, float]]:
    """Parse ``--fusion-weights bm25=1,vector=1,ngram=2`` into a mapping.

    Errors are raised as ``SystemExit`` so a typo in a weight spec reads like
    any other argument error rather than like a crash. Names are *not* checked
    here: which lists exist depends on whether an n-gram index was fitted, so
    :func:`ragkb.retriever.resolve_fusion_weights` owns that check and this
    function only owns the syntax.
    """
    if args.fusion_weights is None:
        return None
    weights: Dict[str, float] = {}
    for part in args.fusion_weights.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                "--fusion-weights expects name=value pairs, got {!r}".format(part)
            )
        name, _, raw = part.partition("=")
        try:
            weights[name.strip()] = float(raw)
        except ValueError:
            raise SystemExit(
                "--fusion-weights value for {!r} is not a number: {!r}".format(
                    name.strip(), raw
                )
            )
    return weights or None


def _abstain_config(args) -> Optional[AbstentionConfig]:
    if args.abstain is None:
        return None
    return AbstentionConfig(threshold=args.abstain)


def _build_engine(args) -> RAGEngine:
    if getattr(args, "index", None):
        return _load_engine(args)
    config = EngineConfig(
        target_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        rrf_k=args.rrf_k if args.rrf_k is not None else 60,
        mmr_lambda=args.mmr,
        expansion=_expansion_config(args),
        rerank=_rerank_config(args),
        ngram=args.ngram,
        fusion_weights=_fusion_weights(args),
        abstain=_abstain_config(args),
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
    rerank = _rerank_config(args)
    if rerank is not None:
        engine.retriever.rerank = engine.config.rerank = rerank
    weights = _fusion_weights(args)
    if weights is not None:
        # Query-time, so unlike --ngram this one can be applied to a saved
        # index: it re-weights ranked lists that already exist rather than
        # asking for one that was never fitted.
        engine.config.fusion_weights = weights
        engine.retriever.fusion_weights = resolve_fusion_weights(
            weights, [name for name, _ in engine.retriever._lists()]
        )
    if args.ngram is not None and engine.retriever.ngram is None:
        # The n-gram list is fitted state, not a query-time knob, so a saved
        # index that was built without it cannot grow one by flag. Say so
        # rather than answering from a two-list fusion the user did not ask
        # for - the same reasoning as the staleness check above.
        raise SystemExit(
            "index at {} was built without an n-gram list; rebuild it with "
            "`ragkb --ngram {} index`".format(args.index, args.ngram)
        )
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
        provenance = ""
        if result.rerank_rank is not None:
            parts = ["bm25 #{}".format(result.bm25_rank), "vector #{}".format(result.vector_rank)]
            if result.ngram_rank is not None:
                parts.append("ngram #{}".format(result.ngram_rank))
            provenance = "  ({} -> rerank #{})".format(
                ", ".join(parts), result.rerank_rank
            )
        print(
            "{}. [{}] score={:.5f}{}".format(
                rank, result.doc_id, result.score, provenance
            )
        )
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
    if answer.answerability is not None:
        # Printed for answered questions too, not only refusals: a gate you
        # only hear from when it fires is one you cannot calibrate.
        print("\nanswerability: {}".format(answer.answerability.explain()))
    return 0


def _cmd_explain(args) -> int:
    """Trace one chunk's score back to the terms and ranks that produced it."""
    engine = _build_engine(args)
    retriever = engine.retriever
    if args.doc is not None:
        try:
            chunk_index = best_chunk_for_doc(
                retriever, args.query, args.doc, method=args.method
            )
        except KeyError:
            known = sorted({chunk.doc_id for chunk in retriever.chunks})
            print(
                "no document named {!r}. known documents: {}".format(
                    args.doc, ", ".join(known)
                )
            )
            return 1
    elif args.chunk is not None:
        if not 0 <= args.chunk < len(retriever.chunks):
            print(
                "chunk index out of range (0..{}).".format(len(retriever.chunks) - 1)
            )
            return 1
        chunk_index = args.chunk
    else:
        results = engine.search(args.query, k=args.rank, method=args.method)
        if len(results) < args.rank:
            print("fewer than {} results for that query.".format(args.rank))
            return 1
        chunk_index = results[args.rank - 1].chunk_index

    report = attribute(retriever, args.query, chunk_index, method=args.method)
    print(report.format(top_terms=args.terms))
    print("\npassage:\n  {}".format(report_text(retriever, chunk_index)))
    return 0


def report_text(retriever, chunk_index: int, limit: int = 300) -> str:
    return retriever.chunks[chunk_index].text[:limit].replace("\n", " ")


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


def _cmd_serve(args) -> int:
    """Fit (or load) the index once, then serve it over HTTP until interrupted."""
    from .serve import serve

    engine = _build_engine(args)
    serve(engine, host=args.host, port=args.port, quiet=not args.log_requests)
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
    parser.add_argument(
        "--rerank",
        type=float,
        default=None,
        metavar="ALPHA",
        help=(
            "re-rank the candidate pool with the lexical cross-encoder; ALPHA "
            "in [0, 1] is the weight kept on the first-stage score, so 1.0 "
            "disables the second stage and 0.0 ignores fusion entirely"
        ),
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=None,
        metavar="N",
        help=(
            "add a character N-gram index as a third list in the fusion, so "
            "the retrievers stop sharing one word-level view of the text; 4 "
            "is the usual value. N is required rather than optional because "
            "an optional-argument flag would swallow the subcommand after it. "
            "Index-time: a saved index has to be rebuilt to gain one"
        ),
    )
    parser.add_argument(
        "--fusion-weights",
        default=None,
        metavar="SPEC",
        help=(
            "per-list weights in the RRF sum, as comma-separated name=value "
            "pairs, e.g. `bm25=1,vector=1,ngram=2`. Only the ratios matter - "
            "weights are rescaled to mean 1.0 - so a uniform spec is exactly "
            "plain RRF. A list left out of SPEC keeps weight 1.0; weight it "
            "0 to drop it from the fusion. Query-time: it applies to a saved "
            "index too"
        ),
    )
    parser.add_argument(
        "--abstain",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "decline to answer when the retrieved evidence scores below "
            "THRESHOLD in [0, 1]; {:g} is the operating point reported by "
            "evals/abstention_eval.py. THRESHOLD is required rather than "
            "optional for the same reason as --ngram: an optional-argument "
            "flag would swallow the subcommand after it. Off by default, so "
            "the engine answers everything unless asked not to".format(
                AbstentionConfig().threshold
            )
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    methods = ("hybrid", "bm25", "vector", "ngram")

    search = sub.add_parser("search", help="show the retrieved passages")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--method", default="hybrid", choices=methods)
    search.set_defaults(func=_cmd_search)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("query")
    ask.add_argument("-k", type=int, default=4)
    ask.add_argument("--method", default="hybrid", choices=methods)
    ask.add_argument("--llm", action="store_true", help="use the OpenAI answerer")
    ask.add_argument("--model", default="gpt-4o-mini")
    ask.set_defaults(func=_cmd_ask)

    explain = sub.add_parser(
        "explain",
        help="show which terms and which ranks produced one chunk's score",
    )
    explain.add_argument("query")
    explain.add_argument(
        "--rank",
        type=int,
        default=1,
        metavar="R",
        help="explain the Rth retrieved chunk (default 1, the top hit)",
    )
    explain.add_argument(
        "--chunk",
        type=int,
        default=None,
        metavar="INDEX",
        help="explain this chunk index instead, retrieved or not",
    )
    explain.add_argument(
        "--doc",
        default=None,
        metavar="DOC_ID",
        help=(
            "explain this document's best-scoring chunk, retrieved or not - "
            "the 'why did the right document lose' case"
        ),
    )
    explain.add_argument(
        "--terms",
        type=int,
        default=6,
        metavar="N",
        help="how many term rows to print per list (default %(default)s)",
    )
    explain.add_argument("--method", default="hybrid", choices=methods)
    explain.set_defaults(func=_cmd_explain)

    stats = sub.add_parser("stats", help="print index statistics")
    stats.set_defaults(func=_cmd_stats)

    index = sub.add_parser(
        "index", help="build the index once and save it for fast startup"
    )
    index.add_argument(
        "--out", default="index.json", help="where to write the index (JSON)"
    )
    index.set_defaults(func=_cmd_index)

    serve = sub.add_parser(
        "serve",
        help="serve a small read-only browser demo (standard library only)",
    )
    serve.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=(
            "interface to bind (default {}). The demo has no authentication, "
            "so binding it anywhere else exposes the corpus".format(DEFAULT_HOST)
        ),
    )
    serve.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="port (default %(default)s)"
    )
    serve.add_argument(
        "--log-requests",
        action="store_true",
        help="print a line per request instead of staying quiet",
    )
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
