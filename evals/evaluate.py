"""Evaluate retrieval quality against evals/gold.jsonl.

Runs the gold question set through each retrieval method and prints an ablation
table so that the hybrid retriever has to justify itself against its own
components. Document-level relevance is used: a result counts as a hit when the
retrieved chunk came from a document listed as relevant for that question.

A second table sweeps the MMR diversity trade-off. Accuracy metrics alone cannot
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
                records.append(json.loads(line))
    if not records:
        raise ValueError("gold set is empty: {}".format(path))
    return records


def evaluate_method(
    engine: RAGEngine, gold: Sequence[Dict], method: str, k: int
) -> Dict[str, float]:
    rankings = [
        engine.ranked_doc_ids(record["question"], k=k, method=method) for record in gold
    ]
    relevancies = [record["relevant"] for record in gold]
    return aggregate(rankings, relevancies, ks=(1, 3, k))


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


def format_diversity_table(rows: Sequence[tuple]) -> str:
    columns = list(rows[0][1].keys())
    header = "{:<8}".format("lambda") + "".join(
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
    print("questions: {}   k={}\n".format(len(gold), args.k))

    reports = {
        method: evaluate_method(engine, gold, method, args.k) for method in METHODS
    }
    print(format_table(reports))

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
        print(format_diversity_table(rows))

    failures = [
        record["id"]
        for record in gold
        if not set(engine.ranked_doc_ids(record["question"], k=args.k))
        & set(record["relevant"])
    ]
    print(
        "\nhybrid misses at k={}: {}".format(
            args.k, ", ".join(failures) if failures else "none"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
