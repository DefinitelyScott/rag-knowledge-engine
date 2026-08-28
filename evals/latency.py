"""Measure what each retrieval stage costs, and what it buys.

`evals/evaluate.py` answers "did it rank better?" for every optional stage in
the pipeline. This script answers the other half of the same question. Four
stages are optional - the character n-gram third list, pseudo-relevance
feedback, cross-encoder re-ranking and MMR - and each of them was added with an
accuracy justification and no cost one. A stage is only genuinely optional once
both numbers exist, because "+0.02 nDCG" reads very differently at 1.1x the
query time than at 4x it.

Four tables:

0. **Inverted index work.** The one structural table here: not milliseconds but
   counts. Ranking is term-at-a-time over postings, and this table separates
   the two things that change from a full scan - how much of the collection a
   query reaches at all, and how many lookups scoring it costs. The first
   number is the disappointing one and is printed first on purpose.

1. **Index build.** Where the one-off cost goes: chunking, then fitting each
   index. Fitting is what the persistence layer exists to skip, so this table
   is also the size of the prize `python -m ragkb.cli index` is claiming.

2. **Query latency.** Each single-list retriever alone, then the fusion, then
   the fusion plus one optional stage at a time, then everything at once. Every
   row is measured against the *same* queries and the same engine, so the
   ``xbase`` column is the marginal cost of that stage and nothing else.

3. **Cost against quality.** The two halves joined on one line per stage:
   milliseconds, multiplier, and the accuracy columns from the eval harness.
   This is the only table in the project where a stage has to justify its price
   rather than just its effect.

Absolute milliseconds describe this machine and this 14-document corpus; the
``xbase`` multipliers are the part that travels. Nothing here is a
scaling study - one corpus size cannot tell you the shape of a curve - so the
numbers say what these stages cost at this size, not how they will grow.

Usage:
    python evals/latency.py [--corpus corpus] [--k 5] [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.bench import Timing, format_timings, measure, measure_one  # noqa: E402
from ragkb.bm25 import BM25Index  # noqa: E402
from ragkb.crossencoder import RerankConfig  # noqa: E402
from ragkb.engine import EngineConfig, RAGEngine, read_corpus  # noqa: E402
from ragkb.expansion import ExpansionConfig  # noqa: E402
from ragkb.metrics import aggregate  # noqa: E402
from ragkb.ngram import CharNgramIndex  # noqa: E402
from ragkb.text import chunk_document, tokenize  # noqa: E402
from ragkb.vector import TfidfIndex  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_questions(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError("gold set is empty: {}".format(path))
    return records


def build_timings(
    documents: Dict[str, str], config: EngineConfig, ngram_n: int, repeats: int
) -> List[Timing]:
    """Time chunking and each index fit separately, then the whole build.

    The parts are measured on their own rather than by subtraction: three
    timings that happen to sum to a fourth are far easier to trust than three
    differences between four end-to-end measurements.
    """

    def chunk_all():
        chunks = []
        for doc_id in sorted(documents):
            chunks.extend(
                chunk_document(
                    doc_id,
                    documents[doc_id],
                    target_tokens=config.target_tokens,
                    overlap_tokens=config.overlap_tokens,
                )
            )
        return chunks

    chunks = chunk_all()
    tokenised = [chunk.tokens for chunk in chunks]

    timings = [
        measure_one("chunk", chunk_all, repeats=repeats),
        measure_one("fit bm25", lambda: BM25Index().fit(tokenised), repeats=repeats),
        measure_one("fit tfidf", lambda: TfidfIndex().fit(tokenised), repeats=repeats),
        measure_one(
            "fit ngram({})".format(ngram_n),
            lambda: CharNgramIndex(n=ngram_n).fit(tokenised),
            repeats=repeats,
        ),
        measure_one(
            "build (2 lists)",
            lambda: RAGEngine(documents, config=config),
            repeats=repeats,
        ),
    ]
    return timings


def query_timings(
    engine: RAGEngine,
    ngram_engine: RAGEngine,
    questions: Sequence[str],
    k: int,
    repeats: int,
    expansion: ExpansionConfig,
    rerank: RerankConfig,
    mmr_lambda: float,
) -> List[Timing]:
    """Time one retrieval configuration per row, over every gold question.

    The single-list rows time the index's own ``search`` rather than
    ``engine.search(method="bm25")``. That distinction is not pedantry: the
    engine populates provenance ranks for *every* list on every query, so its
    ``method`` flag chooses which ranking to return and skips no work at all.
    Timing it would have produced a "bm25 only" row costing the same as the
    fusion, which is true of the engine and false of BM25.
    """
    tokenised = [tokenize(question) for question in questions]

    def index_calls(index):
        return [(lambda t=tokens: index.search(t, k=k)) for tokens in tokenised]

    def calls(target: RAGEngine, **kwargs):
        return [
            (lambda q=question: target.search(q, k=k, **kwargs))
            for question in questions
        ]

    rows = [
        measure("bm25 index", index_calls(engine.retriever.bm25), repeats=repeats),
        measure("tfidf index", index_calls(engine.retriever.vector), repeats=repeats),
        measure(
            "ngram index",
            index_calls(ngram_engine.retriever.ngram),
            repeats=repeats,
        ),
        measure("hybrid (2 lists)", calls(engine), repeats=repeats),
        measure("hybrid (3 lists)", calls(ngram_engine), repeats=repeats),
        measure(
            "  + mmr {:g}".format(mmr_lambda),
            calls(engine, mmr_lambda=mmr_lambda),
            repeats=repeats,
        ),
        measure(
            "  + expansion", calls(engine, expansion=expansion), repeats=repeats
        ),
        measure("  + rerank", calls(engine, rerank=rerank), repeats=repeats),
        measure(
            "  + all three",
            calls(engine, mmr_lambda=mmr_lambda, expansion=expansion, rerank=rerank),
            repeats=repeats,
        ),
    ]
    return rows


def _query_terms(index, question: str):
    """The terms ``index`` will look up for ``question``.

    The n-gram index answers word queries by converting them to grams first, so
    "the query terms" is index-specific and asking the index is the only way to
    count what it will really do.
    """
    tokens = tokenize(question)
    if hasattr(index, "vectorize_query"):
        return index.vectorize_query(tokens)
    return Counter(tokens)


def format_index_work(indexes: Sequence[tuple], questions: Sequence[str]) -> str:
    """Counts, not milliseconds: what inverting the ranking loop removed.

    ``reach`` is the share of the collection a query touches through at least
    one term - the ceiling on what skipping non-matching documents can ever
    save. ``scan`` counts the lookups a document-at-a-time loop performs over
    those documents (reached documents x query terms, hit or miss); ``taat``
    counts the accumulator updates the postings walk performs (the sum of the
    query terms' document frequencies). Both are exact counts of the same
    ranking, so ``taat/scan`` is a machine-independent statement of the saving.
    """
    header = "{:<16}{:>8}{:>10}{:>12}{:>12}{:>12}".format(
        "index", "docs", "reach", "scan", "taat", "taat/scan"
    )
    lines = [header, "-" * len(header)]
    for label, index in indexes:
        reached = scan = taat = 0
        for question in questions:
            terms = _query_terms(index, question)
            hits = len(index.candidates(terms))
            reached += hits
            scan += hits * len(terms)
            taat += sum(len(index.postings.get(term, ())) for term in terms)
        reach = reached / (len(questions) * index.doc_count) if index.doc_count else 0.0
        lines.append(
            "{:<16}{:>8d}{:>9.1f}%{:>12d}{:>12d}{:>12.2f}".format(
                label,
                index.doc_count,
                100.0 * reach,
                scan,
                taat,
                taat / scan if scan else 0.0,
            )
        )
    return "\n".join(lines)


def quality(engine: RAGEngine, gold: Sequence[Dict], k: int, **kwargs) -> Dict[str, float]:
    """The accuracy columns for one configuration, on the same gold set."""
    rankings = [
        engine.ranked_doc_ids(record["question"], k=k, **kwargs) for record in gold
    ]
    report = aggregate(rankings, [record["relevant"] for record in gold], ks=(1, k))
    return {
        "hit@1": report["hit@1"],
        "recall@{}".format(k): report["recall@{}".format(k)],
        "ndcg@{}".format(k): report["ndcg@{}".format(k)],
        "mrr": report["mrr"],
    }


def format_tradeoff(
    rows: Sequence[tuple], baseline: Timing, k: int
) -> str:
    """One line per stage: what it costs, and what it moved."""
    columns = ["hit@1", "recall@{}".format(k), "ndcg@{}".format(k), "mrr"]
    width = max(len(label) for label, _, _ in rows) + 2
    header = "{:<{width}}".format("stage", width=width) + "".join(
        ["{:>12}".format("median_ms"), "{:>8}".format("xbase")]
    ) + "".join("{:>12}".format(name) for name in columns)
    lines = [header, "-" * len(header)]
    for label, timing, scores in rows:
        lines.append(
            "{:<{width}}".format(label, width=width)
            + "{:>12.2f}".format(timing.median_ms)
            + "{:>8.2f}".format(timing.ratio_to(baseline))
            + "".join("{:>12.3f}".format(scores[name]) for name in columns)
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--ngram-n", type=int, default=4)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="how many timed passes over the query set (warmup pass excluded)",
    )
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--expand-terms", type=int, default=5)
    parser.add_argument("--expand-weight", type=float, default=0.8)
    parser.add_argument("--rerank-alpha", type=float, default=0.6)
    args = parser.parse_args(argv)

    gold = load_questions(args.gold)
    questions = [record["question"] for record in gold]
    documents = read_corpus(args.corpus)
    config = EngineConfig(
        target_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens
    )
    engine = RAGEngine(documents, config=config)
    ngram_engine = RAGEngine(
        documents,
        config=EngineConfig(
            target_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            ngram=args.ngram_n,
        ),
    )
    expansion = ExpansionConfig(
        feedback_terms=args.expand_terms, original_weight=args.expand_weight
    )
    rerank = RerankConfig(alpha=args.rerank_alpha)

    stats = engine.stats()
    print(
        "corpus: {} documents, {} chunks, {} vocabulary terms".format(
            int(stats["documents"]), int(stats["chunks"]), int(stats["vocabulary"])
        )
    )
    print(
        "timing: {} questions x {} repeats, median of pooled per-call samples".format(
            len(questions), args.repeats
        )
    )

    print("\ninverted index work per query set (counts, not milliseconds)")
    print(
        format_index_work(
            [
                ("bm25", engine.retriever.bm25),
                ("tfidf", engine.retriever.vector),
                ("ngram({})".format(args.ngram_n), ngram_engine.retriever.ngram),
            ],
            questions,
        )
    )
    print(
        "reach near 100% is the finding, not a bug: a question with one common "
        "word in it\nmatches nearly everything through that word, so skipping "
        "non-matching documents\nsaves almost nothing. taat/scan is where the "
        "inverted index actually pays."
    )

    print("\nindex build (one-off, whole corpus)")
    print(format_timings(build_timings(documents, config, args.ngram_n, args.repeats)))

    rows = query_timings(
        engine,
        ngram_engine,
        questions,
        args.k,
        args.repeats,
        expansion,
        rerank,
        args.mmr_lambda,
    )
    baseline = next(row for row in rows if row.label == "hybrid (2 lists)")
    print("\nquery latency (per query, k={})".format(args.k))
    print(format_timings(rows, baseline=baseline))
    print(
        "the index rows time the index directly; engine.search(method=...) "
        "always queries\nevery list, because provenance ranks are recorded for "
        "all of them on every query."
    )

    by_label = {row.label: row for row in rows}
    tradeoff = [
        ("hybrid (2 lists)", by_label["hybrid (2 lists)"], quality(engine, gold, args.k)),
        (
            "hybrid (3 lists)",
            by_label["hybrid (3 lists)"],
            quality(ngram_engine, gold, args.k),
        ),
        (
            "  + mmr {:g}".format(args.mmr_lambda),
            by_label["  + mmr {:g}".format(args.mmr_lambda)],
            quality(engine, gold, args.k, mmr_lambda=args.mmr_lambda),
        ),
        (
            "  + expansion",
            by_label["  + expansion"],
            quality(engine, gold, args.k, expansion=expansion),
        ),
        (
            "  + rerank",
            by_label["  + rerank"],
            quality(engine, gold, args.k, rerank=rerank),
        ),
        (
            "  + all three",
            by_label["  + all three"],
            quality(
                engine,
                gold,
                args.k,
                mmr_lambda=args.mmr_lambda,
                expansion=expansion,
                rerank=rerank,
            ),
        ),
    ]
    print("\ncost against quality (k={})".format(args.k))
    print(format_tradeoff(tradeoff, baseline, args.k))
    print(
        "\nRead the xbase column, not the milliseconds: absolute timings are "
        "this machine's,\nthe ratios are the pipeline's."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
