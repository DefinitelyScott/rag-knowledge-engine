"""How much of each sweep table's top row is the sweep choosing its own winner?

`evals/evaluate.py` and `evals/tune.py` print sweep tables; `evals/significance.py`
puts error bars on the gaps *inside* them. None of the three addresses the fact
that the top row of a sweep was picked by looking at the same questions it is
scored on. The maximum of several noisy estimates is biased upward even when
every configuration is identical, so "the best `rrf_k` scored 0.83" is not a
prediction about the next 78 questions.

This script re-runs three query-time sweeps under nested cross-validation
(:mod:`ragkb.holdout`): for each fold, pick the winner on the other folds, then
score that choice on the held-out questions. Three sweeps, because they differ in
the one respect that matters here - `rrf_k` is known to be nearly inert, the
fusion weights visibly move the ranking, and MMR trades one metric for another -
and a bias-measuring harness that only ever sees one kind of sweep proves
nothing.

Each row reports:

* ``in-sample`` - the sweep table's own top-row score.
* ``held-out`` - what "sweep, then take the winner" earns on unseen questions,
  averaged over repeated splits.
* ``optimism`` - the difference, i.e. the part of the headline that is selection.
* ``vs default`` - held-out score minus the shipped constant's score. Negative
  means the sweep, honestly scored, loses to not sweeping.
* ``stable`` - the share of splits on which every fold picked the same
  configuration. Low values mean the sweep's ordering is tracking the sample.

Usage:
    python evals/holdout.py
    python evals/holdout.py --metrics ndcg@5 --folds 10 --repeats 50
    python evals/holdout.py --slice hard --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.holdout import repeated_selection, summarize  # noqa: E402
from ragkb.metrics import (  # noqa: E402
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragkb.significance import resolution  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CUTOFF_METRICS = {
    "hit": hit_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "ndcg": ndcg_at_k,
}

DEFAULT_METRICS = "hit@1,ndcg@5,mrr"

#: The n for the character n-gram index used by the fusion-weight sweep. The
#: weights only have something to arbitrate once a third list exists.
NGRAM_N = 4


def metric_value(name: str, ranked: Sequence[str], relevant: Sequence[str]) -> float:
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


#: One sweep: a display name, the configuration labels in offer order (the
#: shipped default first, so :func:`ragkb.holdout.cross_validated_selection`
#: breaks ties towards not moving), and a callable that ranks one question under
#: one configuration.
Sweep = Tuple[str, str, List[str], Callable[[str, str, int], List[str]]]


def rrf_k_sweep(engine: RAGEngine, values: Sequence[int]) -> Sweep:
    """`rrf_k` only rescales two existing ranked lists, so one engine serves all."""
    ordered = [EngineConfig.rrf_k] + [v for v in values if v != EngineConfig.rrf_k]
    labels = [str(v) for v in ordered]

    def rank(label: str, question: str, k: int) -> List[str]:
        original = engine.retriever.rrf_k
        try:
            engine.retriever.rrf_k = int(label)
            return engine.ranked_doc_ids(question, k=k, method="hybrid")
        finally:
            engine.retriever.rrf_k = original

    return ("rrf_k", str(EngineConfig.rrf_k), labels, rank)


def weight_sweep(engine: RAGEngine, specs: Sequence[Tuple[str, Dict[str, float]]]) -> Sweep:
    """Per-list RRF weights over the three-list fusion."""
    table: Dict[str, Optional[Dict[str, float]]] = {"uniform": None}
    labels = ["uniform"]
    for label, weights in specs:
        table[label] = weights
        labels.append(label)
    table["ngram only"] = {"bm25": 0.0, "vector": 0.0}
    labels.append("ngram only")

    def rank(label: str, question: str, k: int) -> List[str]:
        return engine.ranked_doc_ids(
            question, k=k, method="hybrid", fusion_weights=table[label]
        )

    return ("fusion weights", "uniform", labels, rank)


def mmr_sweep(engine: RAGEngine, lambdas: Sequence[float]) -> Sweep:
    """MMR's relevance/diversity trade-off, with ``off`` as the shipped default."""
    labels = ["off"] + ["{:g}".format(value) for value in lambdas]

    def rank(label: str, question: str, k: int) -> List[str]:
        lam = None if label == "off" else float(label)
        return engine.ranked_doc_ids(question, k=k, method="hybrid", mmr_lambda=lam)

    return ("mmr lambda", "off", labels, rank)


def score_sweep(
    sweep: Sweep, gold: Sequence[Dict], k: int, metrics: Sequence[str]
) -> Dict[str, Dict[str, List[float]]]:
    """``metric -> config -> per-question scores``.

    Retrieval runs once per (config, question) and is scored for every metric,
    which is what keeps a three-metric run the same retrieval cost as a
    one-metric run.
    """
    _, _, labels, rank = sweep
    out: Dict[str, Dict[str, List[float]]] = {metric: {} for metric in metrics}
    for label in labels:
        rankings = [rank(label, record["question"], k) for record in gold]
        for metric in metrics:
            out[metric][label] = [
                metric_value(metric, ranking, record["relevant"])
                for ranking, record in zip(rankings, gold)
            ]
    return out


def format_table(rows: Sequence[Tuple[str, str, str, Dict[str, float]]]) -> str:
    header = "{:<16}{:<10}{:<14}{:>11}{:>11}{:>11}{:>13}{:>9}".format(
        "sweep",
        "metric",
        "picked",
        "in-sample",
        "held-out",
        "optimism",
        "vs default",
        "stable",
    )
    lines = [header, "-" * len(header)]
    for sweep_name, metric, picked, summary in rows:
        gain = summary.get("tuning_gain")
        lines.append(
            "{:<16}{:<10}{:<14}{:>11.3f}{:>11.3f}{:>+11.3f}{:>13}{:>9.0%}".format(
                sweep_name,
                metric,
                picked,
                summary["in_sample_best"],
                summary["cv_selected"],
                summary["optimism"],
                "n/a" if gain is None else "{:+.3f}".format(gain),
                summary["stable_share"],
            )
        )
    return "\n".join(lines)


def parse_weight_specs(raw: str) -> List[Tuple[str, Dict[str, float]]]:
    """``"ngram=2;ngram=4,bm25=0.5"`` -> labelled weight mappings."""
    specs: List[Tuple[str, Dict[str, float]]] = []
    for group in raw.split(";"):
        group = group.strip()
        if not group:
            continue
        weights: Dict[str, float] = {}
        for pair in group.split(","):
            name, _, value = pair.partition("=")
            if not value:
                raise ValueError("bad weight spec: {!r}".format(pair))
            weights[name.strip()] = float(value)
        specs.append((group, weights))
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--slice", default="all", choices=("all", "easy", "hard"))
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="how many different k-fold splits to average over",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rrf-ks", default="1,5,10,30,60,120,300")
    parser.add_argument(
        "--fusion-weights", default="ngram=1.5;ngram=2;ngram=3;ngram=4;ngram=6"
    )
    parser.add_argument("--mmr-lambdas", default="0.9,0.7,0.5,0.3")
    args = parser.parse_args(argv)

    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    if not metrics:
        raise SystemExit("no metrics requested")

    gold = select_slice(load_gold(args.gold), args.slice)
    if not gold:
        raise SystemExit("no questions in slice {!r}".format(args.slice))

    base = RAGEngine.from_corpus(args.corpus)
    three = RAGEngine.from_corpus(args.corpus, config=EngineConfig(ngram=NGRAM_N))

    sweeps = [
        rrf_k_sweep(base, [int(v) for v in args.rrf_ks.split(",") if v.strip()]),
        weight_sweep(three, parse_weight_specs(args.fusion_weights)),
        mmr_sweep(
            base, [float(v) for v in args.mmr_lambdas.split(",") if v.strip()]
        ),
    ]

    print(
        "{} questions ({} slice), k={}, {}-fold x {} splits, seed {}".format(
            len(gold), args.slice, args.k, args.folds, args.repeats, args.seed
        )
    )
    print(
        "one question is worth {:.3f} of any per-query metric; a held-out fold is "
        "{} questions".format(
            resolution(len(gold)), len(gold) // args.folds
        )
    )

    rows = []
    negative_gains = 0
    total_gains = 0
    for sweep in sweeps:
        sweep_name, default_label, labels, _ = sweep
        scored = score_sweep(sweep, gold, args.k, metrics)
        for metric in metrics:
            scores = scored[metric]
            # Offer order puts the default first so exact ties resolve to "do
            # not move" rather than to whichever label sorted first.
            ordered = {label: scores[label] for label in labels}
            results = repeated_selection(
                ordered,
                folds=args.folds,
                repeats=args.repeats,
                seed=args.seed,
                default=default_label,
            )
            summary = summarize(results)
            gain = summary.get("tuning_gain")
            if gain is not None:
                total_gains += 1
                negative_gains += 1 if gain < 0 else 0
            rows.append(
                (sweep_name, metric, results[0].in_sample_argbest, summary)
            )
        print(
            "\n{}: {} configurations, default {!r}".format(
                sweep_name, len(labels), default_label
            )
        )
    print()
    print(format_table(rows))

    print(
        "\n`picked` is the configuration the full-set sweep reports; `held-out` is "
        "what choosing\nit that way actually earns on questions it did not see. "
        "`optimism` is the gap between\nthose two - the part of a sweep table's "
        "top row that is selection rather than setting."
    )
    if total_gains:
        print(
            "{} of {} sweep/metric pairs have a negative `vs default`: on those, "
            "the honest\nestimate of tuning is worse than shipping the constant "
            "and never sweeping.".format(negative_gains, total_gains)
        )
    print(
        "A low `stable` share is the same finding read a different way - if the "
        "winner changes\nwith the split, the sweep table's ordering is a property "
        "of these questions, not of\nthe configurations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
