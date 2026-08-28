"""Evaluate retrieval quality against evals/gold.jsonl.

Runs the gold question set through each retrieval method and prints an ablation
table so that the hybrid retriever has to justify itself against its own
components. Document-level relevance is used: a result counts as a hit when the
retrieved chunk came from a document listed as relevant for that question.
Alongside hit/recall/MRR the table reports nDCG@k, which is the only column
that rewards ranking the *second* relevant document high - MRR stops looking
after the first hit and recall@k ignores rank order entirely.

The gold set is split by difficulty and each slice is reported separately. The
easy slice is lexically overlapping - the wording of the question shares terms
with the target document - and it saturates, which hides differences between
methods. The hard slice deliberately avoids the target document's vocabulary
and includes questions whose answer is spread over two documents, so recall@k
stops being a copy of hit@k and the fusion has something to prove.

A second table sweeps the query-expansion trade-off. Pseudo-relevance feedback
borrows terms from the first-pass results, which is the only lever here that can
match a word the question never used - and it pays for that reach at rank one,
so the table reports early precision alongside recall rather than one summary
number.

A third table sweeps the second-stage re-ranker. Fusion recalls far better than
it precisely ranks - hit@5 sits well above hit@1 - and closing that gap is
exactly what a cross-encoder stage is for, so that table leads with the early
ranking columns (hit@1, MRR, nDCG@k) where a re-ranker has to show its work.

A fourth table adds the character n-gram retriever as a third list in the
fusion. Everything above it compares two rankings drawn from *one*
representation - BM25 and TF-IDF both index whole words - so that table is the
only place where the fusion is handed lists that can disagree for a structural
reason rather than a scoring one, and it reports the n-gram list alone
alongside the fusion so the third list has to justify its cost the same way
hybrid does.

A fifth table sweeps the MMR diversity trade-off. Accuracy metrics alone cannot
see redundancy - a top-5 made of five paraphrases of one passage scores exactly
as well as five complementary ones - so that table also reports the mean
pairwise cosine within each result set and how many distinct documents it spans.

Usage:
    python evals/evaluate.py [--corpus corpus] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.crossencoder import RerankConfig  # noqa: E402
from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.expansion import ExpansionConfig  # noqa: E402
from ragkb.metrics import aggregate, mean  # noqa: E402
from ragkb.retriever import METHODS  # noqa: E402
from ragkb.text import tokenize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_gold(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                record.setdefault("difficulty", "easy")
                records.append(record)
    if not records:
        raise ValueError("gold set is empty: {}".format(path))
    return records


def slices(gold: Sequence[Dict]) -> List[tuple]:
    """(label, records) for the whole set and for each difficulty present."""
    out = [("all", list(gold))]
    for level in ("easy", "hard"):
        subset = [record for record in gold if record.get("difficulty") == level]
        if subset:
            out.append((level, subset))
    return out


def describe(gold: Sequence[Dict]) -> str:
    multi = sum(1 for record in gold if len(record["relevant"]) > 1)
    return "{} questions, {} with more than one relevant document".format(
        len(gold), multi
    )


def evaluate_method(
    engine: RAGEngine, gold: Sequence[Dict], method: str, k: int
) -> Dict[str, float]:
    rankings = [
        engine.ranked_doc_ids(record["question"], k=k, method=method) for record in gold
    ]
    relevancies = [record["relevant"] for record in gold]
    report = aggregate(rankings, relevancies, ks=(1, 3, k))
    # Trim to the columns that tell distinct stories at this corpus size:
    # hit@k for coverage as k grows, recall@k for the multi-document questions,
    # nDCG@k for how well the whole relevant set is ordered, MRR for the first
    # hit. precision@1 == hit@1 always, and the dropped mid-k recall/precision
    # values move in lockstep with what is kept.
    keys = [
        "hit@1",
        "hit@3",
        "hit@{}".format(k),
        "recall@{}".format(k),
        "ndcg@{}".format(k),
        "mrr",
    ]
    return {key: report[key] for key in keys}


def parse_expansion_specs(text: str, feedback_docs: int) -> List[tuple]:
    """Parse ``"terms:alpha,terms:alpha"`` into labelled expansion configs."""
    specs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        terms, _, alpha = item.partition(":")
        config = ExpansionConfig(
            feedback_docs=feedback_docs,
            feedback_terms=int(terms),
            original_weight=float(alpha) if alpha else 0.8,
        )
        specs.append(
            (
                "{}t a{:g}".format(config.feedback_terms, config.original_weight),
                config,
            )
        )
    return specs


def expansion_report(
    engine: RAGEngine, gold: Sequence[Dict], k: int, config
) -> Dict[str, float]:
    """Accuracy for one expansion setting, over one slice of the gold set."""
    rankings = [
        engine.ranked_doc_ids(record["question"], k=k, expansion=config)
        for record in gold
    ]
    report = aggregate(rankings, [record["relevant"] for record in gold], ks=(1, 3, k))
    return {
        "hit@1": report["hit@1"],
        "hit@3": report["hit@3"],
        "hit@{}".format(k): report["hit@{}".format(k)],
        "recall@{}".format(k): report["recall@{}".format(k)],
        "mrr": report["mrr"],
    }


def rerank_report(
    engine: RAGEngine, gold: Sequence[Dict], k: int, config
) -> Dict[str, float]:
    """Accuracy for one re-ranking setting, over one slice of the gold set."""
    rankings = [
        engine.ranked_doc_ids(record["question"], k=k, rerank=config)
        for record in gold
    ]
    report = aggregate(rankings, [record["relevant"] for record in gold], ks=(1, 3, k))
    return {
        "hit@1": report["hit@1"],
        "hit@3": report["hit@3"],
        "hit@{}".format(k): report["hit@{}".format(k)],
        "ndcg@{}".format(k): report["ndcg@{}".format(k)],
        "mrr": report["mrr"],
    }


def ngram_table(
    base: RAGEngine, ngram_engine: RAGEngine, gold: Sequence[Dict], k: int
) -> Dict[str, Dict[str, float]]:
    """Two-list fusion, the n-gram list alone, and the three-list fusion.

    ``bm25`` and ``vector`` are omitted: they are identical in both engines
    (adding a third list changes the fusion, not its inputs) and already have
    a table of their own above. What is not knowable from that table is
    whether the third list is worth its cost, so the rows here are exactly the
    three that answer it.
    """
    return {
        "hybrid2": evaluate_method(base, gold, "hybrid", k),
        "ngram": evaluate_method(ngram_engine, gold, "ngram", k),
        "hybrid3": evaluate_method(ngram_engine, gold, "hybrid", k),
    }


def agreement(engine: RAGEngine, gold: Sequence[Dict], k: int) -> Dict[str, float]:
    """How much the fused lists overlap in their own top-``k``.

    The accuracy tables cannot explain *why* a fusion helps or does not. This
    can: RRF has nothing to arbitrate when its inputs return the same chunks
    in the same order, so mean pairwise overlap is the measurement that says
    whether a third list added a genuinely different opinion or a paraphrase
    of an existing one.
    """
    lists = engine.retriever._lists()
    overlaps: List[float] = []
    for record in gold:
        tokens = tokenize(record["question"])
        tops = [
            {index for index, _ in retriever.search(tokens, k=k)}
            for _, retriever in lists
        ]
        pairs = [
            len(left & right) / float(k)
            for position, left in enumerate(tops)
            for right in tops[position + 1 :]
        ]
        if pairs:
            overlaps.append(mean(pairs))
    return {"mean_top{}_overlap".format(k): mean(overlaps)}


def diversity_report(
    engine: RAGEngine, gold: Sequence[Dict], k: int, mmr_lambda
) -> Dict[str, float]:
    """Accuracy plus redundancy for one MMR setting, over the whole gold set."""
    rankings, redundancies, distinct = [], [], []
    for record in gold:
        results = engine.search(record["question"], k=k, mmr_lambda=mmr_lambda)
        redundancies.append(engine.retriever.redundancy(results))
        distinct.append(len({result.doc_id for result in results}))
        rankings.append(
            engine.ranked_doc_ids(
                record["question"], k=k, mmr_lambda=mmr_lambda
            )
        )
    relevancies = [record["relevant"] for record in gold]
    report = aggregate(rankings, relevancies, ks=(1, k))
    return {
        "hit@1": report["hit@1"],
        "hit@{}".format(k): report["hit@{}".format(k)],
        "mrr": report["mrr"],
        "redundancy": mean(redundancies),
        "distinct_docs": mean([float(value) for value in distinct]),
    }


def parse_weight_specs(text: str) -> List[tuple]:
    """Parse ``"ngram=2;ngram=4,vector=0.5"`` into labelled weight mappings.

    Semicolons separate whole specs and commas separate the pairs inside one,
    because a spec is itself comma-separated. The CLI's ``--fusion-weights``
    takes exactly one spec and so needs no outer separator.
    """
    specs: List[tuple] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        weights: Dict[str, float] = {}
        for pair in item.split(","):
            pair = pair.strip()
            if not pair:
                continue
            name, _, raw = pair.partition("=")
            weights[name.strip()] = float(raw)
        if weights:
            specs.append((item, weights))
    return specs


def weight_report(
    engine: RAGEngine, gold: Sequence[Dict], k: int, weights
) -> Dict[str, float]:
    """Accuracy for one set of fusion weights, over one slice of the gold set."""
    rankings = [
        engine.ranked_doc_ids(record["question"], k=k, fusion_weights=weights)
        for record in gold
    ]
    report = aggregate(rankings, [record["relevant"] for record in gold], ks=(1, 3, k))
    return {
        "hit@1": report["hit@1"],
        "hit@3": report["hit@3"],
        "hit@{}".format(k): report["hit@{}".format(k)],
        "recall@{}".format(k): report["recall@{}".format(k)],
        "ndcg@{}".format(k): report["ndcg@{}".format(k)],
        "mrr": report["mrr"],
    }


def format_sweep_table(rows: Sequence[tuple], label: str = "lambda") -> str:
    columns = list(rows[0][1].keys())
    # Sized to the longest row label rather than fixed at 8: a weighted-RRF
    # spec like "ngram=2,bm25=0.5" is wider than that, and a label that
    # overruns its column shifts every number on its row out of alignment,
    # which makes a comparison table unreadable exactly where it is densest.
    width = max([len(label)] + [len(str(name)) for name, _ in rows]) + 2
    header = "{:<{w}}".format(label, w=width) + "".join(
        "{:>15}".format(name) for name in columns
    )
    lines = [header, "-" * len(header)]
    for name, report in rows:
        lines.append(
            "{:<{w}}".format(name, w=width)
            + "".join("{:>15.3f}".format(report[column]) for column in columns)
        )
    return "\n".join(lines)


def format_table(reports: Dict[str, Dict[str, float]]) -> str:
    columns = list(next(iter(reports.values())).keys())
    header = "{:<8}".format("method") + "".join(
        "{:>13}".format(name) for name in columns
    )
    lines = [header, "-" * len(header)]
    for method, report in reports.items():
        row = "{:<8}".format(method) + "".join(
            "{:>13.3f}".format(report[name]) for name in columns
        )
        lines.append(row)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--mmr-lambdas",
        default="1.0,0.9,0.7,0.5,0.3",
        help="comma-separated MMR lambda values to sweep (empty to skip)",
    )
    parser.add_argument(
        "--expansion-specs",
        default="5:1.0,3:0.9,5:0.8,10:0.8,5:0.7",
        help=(
            "comma-separated terms:original_weight expansion settings to sweep "
            "(empty to skip)"
        ),
    )
    parser.add_argument(
        "--rerank-alphas",
        default="1.0,0.8,0.6,0.4,0.2,0.0",
        help=(
            "comma-separated cross-encoder blend weights to sweep; 1.0 keeps "
            "the first-stage ordering (empty to skip)"
        ),
    )
    parser.add_argument(
        "--expansion-docs",
        type=int,
        default=5,
        help="how many first-pass results feed the expansion",
    )
    parser.add_argument(
        "--ngram-n",
        type=int,
        default=4,
        help=(
            "n for the character n-gram list added as a third retriever in "
            "its own table; 0 skips that table"
        ),
    )
    parser.add_argument(
        "--fusion-weights",
        default="ngram=1.5;ngram=2;ngram=3;ngram=4;ngram=6;ngram=2,bm25=0.5",
        help=(
            "semicolon-separated weighted-RRF specs to sweep, each a "
            "comma-separated set of name=weight pairs; empty skips the table. "
            "Requires --ngram-n"
        ),
    )
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    engine = RAGEngine.from_corpus(
        args.corpus,
        config=EngineConfig(
            target_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            rrf_k=args.rrf_k,
        ),
    )

    stats = engine.stats()
    print(
        "corpus: {} documents, {} chunks, {} vocabulary terms, "
        "mean chunk {:.1f} tokens".format(
            int(stats["documents"]),
            int(stats["chunks"]),
            int(stats["vocabulary"]),
            stats["mean_chunk_tokens"],
        )
    )
    print("gold set: {}   k={}".format(describe(gold), args.k))

    for label, subset in slices(gold):
        reports = {
            method: evaluate_method(engine, subset, method, args.k)
            for method in METHODS
        }
        print("\n{} ({})".format(label, describe(subset)))
        print(format_table(reports))

    specs = parse_expansion_specs(args.expansion_specs, args.expansion_docs)
    if specs:
        for label, subset in slices(gold):
            rows = [("off", expansion_report(engine, subset, args.k, None))]
            for spec_label, config in specs:
                rows.append(
                    (spec_label, expansion_report(engine, subset, args.k, config))
                )
            print(
                "\nquery expansion, {} feedback docs (hybrid, k={}) - {} slice".format(
                    args.expansion_docs, args.k, label
                )
            )
            print(format_sweep_table(rows, label="terms"))

    alphas = [value for value in args.rerank_alphas.split(",") if value.strip()]
    if alphas:
        for label, subset in slices(gold):
            rows = [("off", rerank_report(engine, subset, args.k, None))]
            for value in alphas:
                rows.append(
                    (
                        value.strip(),
                        rerank_report(
                            engine, subset, args.k, RerankConfig(alpha=float(value))
                        ),
                    )
                )
            print(
                "\ncross-encoder re-ranking (hybrid, k={}) - {} slice".format(
                    args.k, label
                )
            )
            print(format_sweep_table(rows, label="alpha"))

    if args.ngram_n:
        ngram_engine = RAGEngine.from_corpus(
            args.corpus,
            config=EngineConfig(
                target_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
                rrf_k=args.rrf_k,
                ngram=args.ngram_n,
            ),
        )
        ngram_stats = ngram_engine.stats()
        print(
            "\ncharacter {}-grams: {} gram features against {} word features "
            "({:.1f}x)".format(
                args.ngram_n,
                int(ngram_stats["ngram_vocabulary"]),
                int(ngram_stats["vocabulary"]),
                ngram_stats["ngram_vocabulary"] / max(ngram_stats["vocabulary"], 1),
            )
        )
        base_overlap = agreement(engine, gold, args.k)
        three_overlap = agreement(ngram_engine, gold, args.k)
        print(
            "mean pairwise top-{} overlap between the fused lists: "
            "{:.3f} with two lists, {:.3f} with three".format(
                args.k,
                base_overlap["mean_top{}_overlap".format(args.k)],
                three_overlap["mean_top{}_overlap".format(args.k)],
            )
        )
        for label, subset in slices(gold):
            print(
                "\nthird retriever, character {}-grams (k={}) - {} slice".format(
                    args.ngram_n, args.k, label
                )
            )
            print(format_table(ngram_table(engine, ngram_engine, subset, args.k)))

        weight_specs = parse_weight_specs(args.fusion_weights)
        if weight_specs:
            # Only meaningful once the n-gram list exists: with two lists that
            # perform within noise of each other there is nothing for a weight
            # to arbitrate. The three-list fusion is where uniform RRF visibly
            # hurts - it ranks below the n-gram list alone - so this sweep runs
            # against ``ngram_engine`` and asks whether weighting recovers that
            # gap. The ``uniform`` row is the same fusion the tables above
            # report, and reproduces them exactly.
            for label, subset in slices(gold):
                rows = [("uniform", weight_report(ngram_engine, subset, args.k, None))]
                for spec_label, weights in weight_specs:
                    rows.append(
                        (spec_label, weight_report(ngram_engine, subset, args.k, weights))
                    )
                rows.append(
                    (
                        "ngram only",
                        weight_report(
                            ngram_engine, subset, args.k, {"bm25": 0.0, "vector": 0.0}
                        ),
                    )
                )
                print(
                    "\nweighted RRF (three-list fusion, k={}) - {} slice".format(
                        args.k, label
                    )
                )
                print(format_sweep_table(rows, label="weights"))
            print(
                "\nthe `ngram only` row is the control: any weighting that does not "
                "beat it is evidence the fusion is not earning its third list.\n"
                "run `python evals/significance.py` before believing any gap above - "
                "on {} questions these deltas are a handful of queries wide.".format(
                    len(gold)
                )
            )

    lambdas = [value for value in args.mmr_lambdas.split(",") if value.strip()]
    if lambdas:
        rows = [("off", diversity_report(engine, gold, args.k, None))]
        for value in lambdas:
            rows.append(
                (value.strip(), diversity_report(engine, gold, args.k, float(value)))
            )
        print(
            "\nMMR diversity re-ranking (hybrid, k={})".format(args.k)
        )
        print(format_sweep_table(rows))

    failures = [
        record["id"]
        for record in gold
        if not set(engine.ranked_doc_ids(record["question"], k=args.k))
        & set(record["relevant"])
    ]
    partial = [
        record["id"]
        for record in gold
        if len(record["relevant"]) > 1
        and not set(record["relevant"])
        <= set(engine.ranked_doc_ids(record["question"], k=args.k))
        and record["id"] not in failures
    ]
    print(
        "\nhybrid misses at k={}: {}".format(
            args.k, ", ".join(failures) if failures else "none"
        )
    )
    print(
        "hybrid partial recall at k={} (found some but not all): {}".format(
            args.k, ", ".join(partial) if partial else "none"
        )
    )
    if failures or partial:
        # A list of ids is where the diagnosis starts, not where it ends.
        print("run `python evals/failures.py` for why each of those missed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
