"""Evaluate retrieval quality against evals/gold.jsonl.

Runs the gold question set through each retrieval method and prints an ablation
table so that the hybrid retriever has to justify itself against its own
components. Document-level relevance is used: a result counts as a hit when the
retrieved chunk came from a document listed as relevant for that question.

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

A third table sweeps the MMR diversity trade-off. Accuracy metrics alone cannot
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

from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.expansion import ExpansionConfig  # noqa: E402
from ragkb.metrics import aggregate, mean  # noqa: E402
from ragkb.retriever import METHODS  # noqa: E402

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
    return aggregate(rankings, relevancies, ks=(1, 3, k))


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


def format_sweep_table(rows: Sequence[tuple], label: str = "lambda") -> str:
    columns = list(rows[0][1].keys())
    header = "{:<8}".format(label) + "".join(
        "{:>15}".format(name) for name in columns
    )
    lines = [header, "-" * len(header)]
    for label, report in rows:
        lines.append(
            "{:<8}".format(label)
            + "".join("{:>15.3f}".format(report[name]) for name in columns)
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
        "--expansion-docs",
        type=int,
        default=5,
        help="how many first-pass results feed the expansion",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
