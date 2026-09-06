"""Put error bars on the ablation table in ``evals/evaluate.py``.

``evaluate.py`` prints three methods side by side and invites the reader to pick
a winner from the third decimal place. On the current gold set that invitation is
a trap: 78 questions means one question is worth 0.013 of any per-query metric,
so hybrid's 0.594 hit@1 against BM25's 0.562 on the hard slice is *one
question*, and that slice - 32 questions - resolves no finer than 0.031.

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
from typing import Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.metrics import (  # noqa: E402
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragkb.significance import (  # noqa: E402
    compare,
    holm_bonferroni,
    resolution,
    smallest_achievable_p,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Which engine serves which method, mirroring ``evals/robustness.py`` so the
#: two harnesses cannot drift into scoring different things under one label.
#: The two-list and three-list engines share identical BM25 and TF-IDF indexes,
#: so routing the single-list rows through the base engine costs nothing.
#:
#: ``ngram`` and ``hybrid3`` are the reason this table exists. ``evaluate.py``
#: prints an ``ngram only`` control and tells the reader to run this script
#: before believing any gap in that table - but until now this script scored
#: only ``hybrid``, ``bm25`` and ``vector``, so the one comparison the project
#: leans on hardest was the one comparison it never tested.
METHOD_ROUTING = (
    ("bm25", "base", "bm25"),
    ("vector", "base", "vector"),
    ("ngram", "ngram", "ngram"),
    ("hybrid2", "base", "hybrid"),
    ("hybrid3", "ngram", "hybrid"),
)

DEFAULT_METHODS = ",".join(name for name, _, _ in METHOD_ROUTING)

#: Comparisons the project's own prose asserts, checked explicitly at the end of
#: a run so the claims cannot drift out of agreement with the measurement.
CLAIMS = (
    (
        "ngram",
        "hybrid3",
        "the `ngram only` control in evaluate.py: the third list alone beats "
        "the three-way fusion",
    ),
    (
        "hybrid3",
        "hybrid2",
        "adding the third list to the fusion beats the two-list fusion",
    ),
    (
        "hybrid2",
        "bm25",
        "fusing BM25 with TF-IDF beats BM25 alone",
    ),
)

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
    engines: Mapping[str, RAGEngine],
    routing: Sequence[Tuple[str, str, str]],
    gold: Sequence[Dict],
    k: int,
) -> Dict[str, List[List[str]]]:
    """Retrieve once per method per question and reuse for every metric.

    Retrieval is the expensive part and it does not depend on which metric is
    being scored, so hoisting it out keeps a five-metric run the same cost as a
    one-metric run.
    """
    return {
        label: [
            engines[engine_name].ranked_doc_ids(record["question"], k=k, method=method)
            for record in gold
        ]
        for label, engine_name, method in routing
    }


def pairs(methods: Sequence[str]) -> List[Tuple[str, str]]:
    return [
        (left, right)
        for index, left in enumerate(methods)
        for right in methods[index + 1 :]
    ]


def format_table(rows: Sequence[Tuple[str, str, object, float]]) -> str:
    """One row per (metric, comparison).

    The ``floor`` column is the lowest p-value the comparison could have
    reached given how many questions the two methods actually disagreed on.
    When ``floor`` equals 1.000 the p-value beside it carries no information at
    all - the test had nothing to work with - and printing the two together is
    what stops the reader treating those rows as evidence of equivalence.

    ``p_holm`` is the raw ``p`` after the familywise correction over every
    comparison in the run. It is the column to read: ``p`` is what this row
    would mean if it were the only test performed, and it never is.
    """
    header = "{:<10}{:<22}{:>8}{:>8}{:>9}{:>20}{:>8}{:>9}{:>8}{:>12}".format(
        "metric",
        "comparison",
        "A",
        "B",
        "delta",
        "95% CI",
        "p",
        "p_holm",
        "floor",
        "W/L/T",
    )
    lines = [header, "-" * len(header)]
    for metric, label, result, adjusted in rows:
        lines.append(
            "{:<10}{:<22}{:>8.3f}{:>8.3f}{:>9.3f}{:>20}{:>8.3f}{:>9.3f}{:>8.3f}{:>12}".format(
                metric,
                label,
                result.mean_a,
                result.mean_b,
                result.delta,
                "[{:+.3f}, {:+.3f}]".format(result.ci_low, result.ci_high),
                result.p_value,
                adjusted,
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
    parser.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help="comma-separated methods to compare pairwise",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="familywise error rate for the Holm correction",
    )
    parser.add_argument("--ngram-n", type=int, default=4)
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

    requested = [name.strip() for name in args.methods.split(",") if name.strip()]
    routing = [entry for entry in METHOD_ROUTING if entry[0] in requested]
    known = {entry[0] for entry in METHOD_ROUTING}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise SystemExit(
            "unknown methods: {} (known: {})".format(
                ",".join(unknown), ",".join(sorted(known))
            )
        )
    if len(routing) < 2:
        raise SystemExit("need at least two methods to compare")
    method_names = [entry[0] for entry in routing]

    gold = load_gold(args.gold)
    base_config = EngineConfig(
        target_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        rrf_k=args.rrf_k,
    )
    engines = {"base": RAGEngine.from_corpus(args.corpus, config=base_config)}
    if any(entry[1] == "ngram" for entry in routing):
        engines["ngram"] = RAGEngine.from_corpus(
            args.corpus,
            config=EngineConfig(
                target_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
                rrf_k=args.rrf_k,
                ngram=args.ngram_n,
            ),
        )

    print(
        "{} gold questions, k={}, {} bootstrap resamples, {} permutations, seed {}".format(
            len(gold), args.k, args.bootstrap_resamples, args.permutations, args.seed
        )
    )
    print("methods: {} (n-gram list at n={})".format(", ".join(method_names), args.ngram_n))

    # Every comparison in the run is computed before anything is printed,
    # because the Holm multiplier depends on how many tests the family
    # contains - a per-slice table cannot be corrected in isolation.
    family: List[object] = []
    blocks: List[Tuple[str, int, List[Tuple[str, str, int]]]] = []
    claim_slots: Dict[Tuple[str, str, str], int] = {}
    for label in [name.strip() for name in args.slices.split(",") if name.strip()]:
        subset = select_slice(gold, label)
        if not subset:
            continue
        ranked = rankings_by_method(engines, routing, subset, args.k)
        relevancies = [record["relevant"] for record in subset]
        rows = []
        for metric in metrics:
            scores = {
                method: [
                    metric_value(metric, ranking, relevant)
                    for ranking, relevant in zip(ranked[method], relevancies)
                ]
                for method in method_names
            }
            for left, right in pairs(method_names):
                result = compare(
                    scores[left],
                    scores[right],
                    bootstrap_resamples=args.bootstrap_resamples,
                    permutations=args.permutations,
                    confidence=args.confidence,
                    seed=args.seed,
                )
                comparison = "{} vs {}".format(left, right)
                slot = len(family)
                family.append(result)
                rows.append((metric, comparison, slot))
                claim_slots[(label, metric, comparison)] = slot
        blocks.append((label, len(subset), rows))

    corrected = holm_bonferroni([result.p_value for result in family], alpha=args.alpha)

    for label, size, rows in blocks:
        print(
            "\n{} slice - {} questions, one question is worth {:.3f}".format(
                label, size, resolution(size)
            )
        )
        print(
            format_table(
                [
                    (metric, comparison, family[slot], corrected[slot][0])
                    for metric, comparison, slot in rows
                ]
            )
        )

    any_significant = any(result.significant for result in family)
    survivors = [slot for slot, (_, reject) in enumerate(corrected) if reject]
    most_discordant = max((result.discordant for result in family), default=0)

    print(
        "\nfamilywise correction: {} comparisons in this run, Holm at alpha={:.2f}."
        .format(len(family), args.alpha)
    )
    print(
        "An uncorrected table of {} tests would be expected to show {:.1f} spurious\n"
        "wins at p<{:.2f} even if every method were identical, which is why the\n"
        "p_holm column exists and the raw p column should not be read alone.".format(
            len(family), len(family) * args.alpha, args.alpha
        )
    )
    if survivors:
        print(
            "{} comparison(s) survive the correction.".format(len(survivors))
        )
    else:
        print("No comparison survives the correction.")

    if any_significant:
        print(
            "\nAt least one comparison above has a {:.0%} interval that excludes "
            "zero - but note that interval is uncorrected and the Holm column is "
            "the\nbinding constraint.".format(args.confidence)
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

    print_claims(claim_slots, family, corrected)
    return 0


def print_claims(
    slots: Mapping[Tuple[str, str, str], int],
    family: Sequence[object],
    corrected: Sequence[Tuple[float, bool]],
    slice_label: str = "all",
    metric: str = "ndcg@5",
) -> None:
    """Check the comparisons the project's prose asserts, by name.

    A README that states a result and a script that computes one can drift
    apart silently. Naming each claim here and printing its verdict beside it
    means the drift shows up in the output the next time anyone runs this.
    """
    # ``pairs`` emits each comparison in the order the methods were requested,
    # so a claim written "the third list beats the fusion" may have been
    # computed as "fusion vs third list". Look up both orientations and flip
    # the sign of the delta and the win/loss counts when the stored comparison
    # runs the other way - a claim silently dropped for being written backwards
    # would defeat the point of checking claims at all.
    checked = []
    for left, right, description in CLAIMS:
        forward = (slice_label, metric, "{} vs {}".format(left, right))
        reverse = (slice_label, metric, "{} vs {}".format(right, left))
        if forward in slots:
            checked.append((left, right, description, slots[forward], False))
        elif reverse in slots:
            checked.append((left, right, description, slots[reverse], True))
    if not checked:
        return
    print(
        "\nclaims this project makes, tested on {} / {} slice".format(
            metric, slice_label
        )
    )
    print("-" * 76)
    for left, right, description, slot, flipped in checked:
        result = family[slot]
        adjusted, survives = corrected[slot]
        delta = -result.delta if flipped else result.delta
        wins = result.losses if flipped else result.wins
        losses = result.wins if flipped else result.losses
        if survives and delta > 0:
            verdict = "SUPPORTED"
        elif result.discordant < 6:
            verdict = "UNTESTABLE"
        else:
            verdict = "NOT SHOWN"
        print(
            "{:<11} {:<34} delta {:+.3f}  p_holm {:.3f}  {}W/{}L/{}T".format(
                verdict,
                "{} vs {}".format(left, right),
                delta,
                adjusted,
                wins,
                losses,
                result.ties,
            )
        )
        print("            {}".format(description))
    print(
        "\nUNTESTABLE means the two methods disagreed on fewer than six questions, "
        "so no\namount of resampling could have reached p<0.05 - the floor is "
        "2^(1-d). It is a\nstatement about the gold set, not about the methods, "
        "and the fix is a bigger and\nmore discriminative gold set rather than a "
        "different test."
    )


if __name__ == "__main__":
    raise SystemExit(main())
