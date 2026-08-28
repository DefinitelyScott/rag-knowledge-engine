"""Put error bars on the ablation table in ``evals/evaluate.py``.

``evaluate.py`` prints three methods side by side and invites the reader to pick
a winner from the third decimal place. On the current gold set that invitation is
a trap: 64 questions means one question is worth 0.016 of any per-query metric,
so hybrid's 0.703 hit@1 against BM25's 0.734 is *two questions*, and the
hard slice - 26 questions - resolves no finer than 0.038.

This script re-runs the same comparison and reports, for each pair of methods
and each metric, the paired bootstrap interval for the difference, an
approximate-randomization p-value, and the discordant-pair counts. The counts
are the part worth reading first: "won 3, lost 4, tied 49" makes the size of the
evidence visible in a way no mean can.

The expected answer here is mostly *negative* - at this corpus size the three
methods are not distinguishable on most metrics - and that is the point. The
value of the script is that the README can now say so with a measurement instead
of hedging in prose, and any future change that claims an improvement has a
standard to clear.

Usage:
    python evals/significance.py
    python evals/significance.py --metrics hit@1,ndcg@5 --slices all
    python evals/significance.py --permutations 50000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.metrics import (  # noqa: E402
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragkb.retriever import METHODS  # noqa: E402
from ragkb.significance import (  # noqa: E402
    compare,
    resolution,
    smallest_achievable_p,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Metrics that decompose into an independent per-query score. Everything the
#: paired tests need is that decomposition - a set-level metric with no
#: per-query value could not be resampled this way.
CUTOFF_METRICS = {
    "hit": hit_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "ndcg": ndcg_at_k,
}

DEFAULT_METRICS = "hit@1,hit@5,recall@5,ndcg@5,mrr"


def metric_value(name: str, ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """Score one query with the metric named ``name`` (e.g. ``"hit@1"``, ``"mrr"``)."""
    if name == "mrr":
        return reciprocal_rank(ranked, relevant)
    kind, _, cutoff = name.partition("@")
    if kind not in CUTOFF_METRICS or not cutoff:
        raise ValueError("unknown metric: {}".format(name))
    return CUTOFF_METRICS[kind](ranked, relevant, int(cutoff))


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


def select_slice(gold: Sequence[Dict], label: str) -> List[Dict]:
    if label == "all":
        return list(gold)
    return [record for record in gold if record.get("difficulty") == label]


def rankings_by_method(
    engine: RAGEngine, gold: Sequence[Dict], k: int
) -> Dict[str, List[List[str]]]:
    """Retrieve once per method per question and reuse for every metric.

    Retrieval is the expensive part and it does not depend on which metric is
    being scored, so hoisting it out keeps a five-metric run the same cost as a
    one-metric run.
    """
    return {
        method: [
            engine.ranked_doc_ids(record["question"], k=k, method=method)
            for record in gold
        ]
        for method in METHODS
    }


def pairs(methods: Sequence[str]) -> List[Tuple[str, str]]:
    return [
        (left, right)
        for index, left in enumerate(methods)
        for right in methods[index + 1 :]
    ]


def format_table(rows: Sequence[Tuple[str, str, object]]) -> str:
    """One row per (metric, comparison).

    The ``floor`` column is the lowest p-value the comparison could have
    reached given how many questions the two methods actually disagreed on.
    When ``floor`` equals 1.000 the p-value beside it carries no information at
    all - the test had nothing to work with - and printing the two together is
    what stops the reader treating those rows as evidence of equivalence.
    """
    header = "{:<10}{:<20}{:>8}{:>8}{:>9}{:>20}{:>8}{:>8}{:>12}".format(
        "metric", "comparison", "A", "B", "delta", "95% CI", "p", "floor", "W/L/T"
    )
    lines = [header, "-" * len(header)]
    for metric, label, result in rows:
        lines.append(
            "{:<10}{:<20}{:>8.3f}{:>8.3f}{:>9.3f}{:>20}{:>8.3f}{:>8.3f}{:>12}".format(
                metric,
                label,
                result.mean_a,
                result.mean_b,
                result.delta,
                "[{:+.3f}, {:+.3f}]".format(result.ci_low, result.ci_high),
                result.p_value,
                result.p_floor,
                "{}/{}/{}".format(result.wins, result.losses, result.ties),
            )
        )
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
        "--metrics",
        default=DEFAULT_METRICS,
        help="comma-separated per-query metrics to test",
    )
    parser.add_argument(
        "--slices",
        default="all,easy,hard",
        help="comma-separated gold-set difficulty slices to report",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for both resampling procedures, so a printed interval is reproducible",
    )
    args = parser.parse_args(argv)

    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    if not metrics:
        raise SystemExit("no metrics requested")

    gold = load_gold(args.gold)
    engine = RAGEngine.from_corpus(
        args.corpus,
        config=EngineConfig(
            target_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            rrf_k=args.rrf_k,
        ),
    )

    print(
        "{} gold questions, k={}, {} bootstrap resamples, {} permutations, seed {}".format(
            len(gold), args.k, args.bootstrap_resamples, args.permutations, args.seed
        )
    )

    any_significant = False
    most_discordant = 0
    for label in [name.strip() for name in args.slices.split(",") if name.strip()]:
        subset = select_slice(gold, label)
        if not subset:
            continue
        ranked = rankings_by_method(engine, subset, args.k)
        relevancies = [record["relevant"] for record in subset]
        rows = []
        for metric in metrics:
            scores = {
                method: [
                    metric_value(metric, ranking, relevant)
                    for ranking, relevant in zip(ranked[method], relevancies)
                ]
                for method in METHODS
            }
            for left, right in pairs(METHODS):
                result = compare(
                    scores[left],
                    scores[right],
                    bootstrap_resamples=args.bootstrap_resamples,
                    permutations=args.permutations,
                    confidence=args.confidence,
                    seed=args.seed,
                )
                any_significant = any_significant or result.significant
                most_discordant = max(most_discordant, result.discordant)
                rows.append((metric, "{} vs {}".format(left, right), result))
        print(
            "\n{} slice - {} questions, one question is worth {:.3f}".format(
                label, len(subset), resolution(len(subset))
            )
        )
        print(format_table(rows))

    if any_significant:
        print(
            "\nAt least one comparison above has a {:.0%} interval that excludes "
            "zero.".format(args.confidence)
        )
    else:
        print(
            "\nNo comparison above has a {:.0%} interval that excludes zero, and "
            "the largest number\nof questions any two methods disagreed on is {} "
            "- so the best p-value reachable\nanywhere in this run was {:.3f}."
            .format(
                args.confidence,
                most_discordant,
                smallest_achievable_p(most_discordant),
            )
        )
        print(
            "Read that as a statement about the gold set, not about the methods. "
            "The three\nretrievers agree on almost every question, so a bigger "
            "gold set - specifically one\nwith more questions where they can "
            "disagree - is the prerequisite for any retrieval\nclaim this project "
            "makes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
