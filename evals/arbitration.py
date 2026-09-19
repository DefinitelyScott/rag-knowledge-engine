"""Does the fusion produce a ranking its inputs would not have - and does it pay?

The ablation table in the README reports hybrid, BM25 and TF-IDF side by side
and lets the reader infer what fusion bought. At twenty-six documents it prints
something the table cannot explain:

    method          hit@1        hit@3        hit@5     recall@5       ndcg@5          mrr
    hybrid          0.727        0.864        0.924        0.850        0.777        0.802
    vector          0.727        0.871        0.924        0.850        0.777        0.802

Five of six columns are identical, to three decimals, between the two-list
fusion and one of the two lists it fuses. There are two completely different
explanations for that, and the table cannot tell them apart:

1. **Inertness.** RRF returned the vector list. The two inputs agreed, or one
   dominated, and fusion never had anything to arbitrate. Under this reading
   the fusion is decorative and the tie is the honest report of that.
2. **Cancellation.** RRF returned a *different* list on most questions, and the
   questions it improved and the questions it damaged happened to balance.
   Under this reading the fusion is doing a great deal of work and the work is
   worth nothing on net - which is a far stronger and more interesting claim
   than "it did nothing", and is not the same claim at all.

The distinction matters beyond bookkeeping, because the two readings imply
opposite next moves. If the fusion is inert, the fix is to give it lists that
disagree - which is exactly the argument the README's third-retriever section
makes. If the fusion is arbitrating constantly and netting to zero, more
disagreement buys nothing by itself; what is missing is a reason to trust one
list over another on a given query, which is a weighting or routing problem.
The repo has already acted on the first reading. This script tests whether the
first reading was ever the right one.

``agreement()`` in ``evals/evaluate.py`` is the closest existing measurement and
it answers a different question: it reports how much the fused lists overlap
with **each other**. That is a property of the inputs. Whether the *output*
departs from them is a property of the fusion, and nothing here measured it.

WHAT IS MEASURED

The unit is the deduplicated document ranking at ``k`` - ``ranked_doc_ids``,
the same object every metric in ``evals/evaluate.py`` scores. Comparing chunk
rankings would overstate the reordering rate, because two chunk orders that
differ only in which chunk of the same document came first collapse to one
document ranking and score identically.

Against each input list the fused ranking falls into exactly one of:

* **same order** - identical document sequence. The fusion cannot have changed
  the score on this question, for any metric.
* **same set, reordered** - same documents, different sequence. Rank-sensitive
  metrics (nDCG, MRR, hit@1) can move; set-only metrics (hit@k, recall@k)
  cannot.
* **different set** - at least one document differs. Every metric can move.

``promoted`` counts separately: questions where the fused top-``k`` contains a
document that appeared in **no** input's top-``k``. This is the one thing
fusion can do that no single input could, and if it never happens then the
fused list is only ever a re-shuffle of documents its inputs already surfaced.

THE DECOMPOSITION

Because a same-order question has a per-query delta of exactly zero, the
aggregate delta factors cleanly:

    mean_delta_over_all  =  (reordered / n)  x  mean_delta_over_reordered

The left side is the number in the ablation table. The right side splits it
into how often the fusion acted and what it got when it did. The identity is
asserted numerically at the end of the run rather than taken on faith, because
it is the premise every reading below rests on - and it is exactly the kind of
thing that silently stops holding when someone changes what ``ranked_doc_ids``
returns.

The reordered subset is then handed to ``ragkb.significance.compare`` for a
paired test on that subset alone. Restricting the test to the questions where
the systems can differ does not change the p-value - tied pairs contribute
nothing to a paired permutation test either way - but it does change the
reported W/L/T and the confidence interval, which are computed over ``n``. The
CI on the subset is the one that describes the effect where the effect exists,
and it is wider, which is the honest picture.

A note on what this cannot show. Arbitration rate is not quality. A fusion that
reorders every question and nets to zero is *worse* than an inert one on cost
grounds, not better, and a high rate here is not a defence of the technique.
The number is diagnostic, not a score.
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
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragkb.significance import Comparison, compare  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Each fusion and the lists it fuses, named as ``evals/significance.py`` names
#: them. ``engine`` picks which of the two built engines answers, because the
#: n-gram list only exists on one of them.
FUSIONS = (
    ("hybrid2", "base", ("bm25", "vector")),
    ("hybrid3", "ngram", ("bm25", "vector", "ngram")),
)

#: Metrics reported, and whether the metric can move when the *set* of returned
#: documents is unchanged. ``False`` means a same-set reordering is invisible to
#: it, which is why those rows show a reordering count far above their
#: scored-differently count and it is not a bug.
METRICS = (
    ("hit@1", True),
    ("hit@5", False),
    ("recall@5", False),
    ("ndcg@5", True),
    ("mrr", True),
)

SAME_ORDER = "same order"
SAME_SET = "same set, reordered"
DIFFERENT_SET = "different set"


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


def score(metric: str, ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """One metric on one question, matching ``ragkb.metrics.aggregate``."""
    if metric == "mrr":
        return reciprocal_rank(ranked, relevant)
    name, _, depth = metric.partition("@")
    at = int(depth) if depth else k
    if name == "hit":
        return hit_at_k(ranked, relevant, at)
    if name == "recall":
        return recall_at_k(ranked, relevant, at)
    if name == "ndcg":
        return ndcg_at_k(ranked, relevant, at)
    raise ValueError("unknown metric: {}".format(metric))


def relation(fused: Sequence[str], other: Sequence[str]) -> str:
    """Which of the three reordering relations holds between two rankings."""
    if list(fused) == list(other):
        return SAME_ORDER
    if set(fused) == set(other):
        return SAME_SET
    return DIFFERENT_SET


def rankings(
    engine: RAGEngine, gold: Sequence[Dict], method: str, k: int
) -> List[List[str]]:
    return [
        engine.ranked_doc_ids(record["question"], k=k, method=method)
        for record in gold
    ]


def relation_counts(
    fused: Sequence[Sequence[str]], other: Sequence[Sequence[str]]
) -> Dict[str, int]:
    counts = {SAME_ORDER: 0, SAME_SET: 0, DIFFERENT_SET: 0}
    for left, right in zip(fused, other):
        counts[relation(left, right)] += 1
    return counts


def promoted(
    fused: Sequence[Sequence[str]], inputs: Mapping[str, Sequence[Sequence[str]]]
) -> int:
    """Questions whose fused top-k holds a document no input's top-k held.

    The only outcome unreachable by any single input list, so a zero here would
    mean the fusion is strictly a permutation of what its inputs already showed.
    """
    total = 0
    for position, ranking in enumerate(fused):
        union = set()
        for lists in inputs.values():
            union.update(lists[position])
        if set(ranking) - union:
            total += 1
    return total


def format_relations(
    label: str,
    fused: Sequence[Sequence[str]],
    inputs: Mapping[str, Sequence[Sequence[str]]],
) -> str:
    n = len(fused)
    lines = [
        "{} against the {} lists it fuses ({} questions)".format(
            label, len(inputs), n
        ),
        "{:<10}{:>13}{:>22}{:>16}{:>13}".format(
            "input", "same order", "same set, reordered", "different set", "acted on"
        ),
        "-" * 74,
    ]
    for name, lists in inputs.items():
        counts = relation_counts(fused, lists)
        acted = counts[SAME_SET] + counts[DIFFERENT_SET]
        lines.append(
            "{:<10}{:>13}{:>22}{:>16}{:>13}".format(
                name,
                "{} ({:.3f})".format(counts[SAME_ORDER], counts[SAME_ORDER] / n),
                "{} ({:.3f})".format(counts[SAME_SET], counts[SAME_SET] / n),
                "{} ({:.3f})".format(
                    counts[DIFFERENT_SET], counts[DIFFERENT_SET] / n
                ),
                "{} ({:.3f})".format(acted, acted / n),
            )
        )
    count = promoted(fused, inputs)
    lines.append("")
    lines.append(
        "fused top-k holds a document that appeared in no input's top-k: "
        "{} of {} ({:.3f})".format(count, n, count / n)
    )
    return "\n".join(lines)


def decompose(
    fused_scores: Sequence[float],
    other_scores: Sequence[float],
    acted: Sequence[bool],
) -> Tuple[float, float, int, int, int]:
    """``(aggregate delta, delta on the acted subset, acted, wins, losses)``."""
    n = len(fused_scores)
    subset = [i for i in range(n) if acted[i]]
    aggregate_delta = mean(fused_scores) - mean(other_scores)
    if subset:
        subset_delta = mean([fused_scores[i] for i in subset]) - mean(
            [other_scores[i] for i in subset]
        )
    else:
        subset_delta = 0.0
    wins = sum(1 for i in subset if fused_scores[i] > other_scores[i])
    losses = sum(1 for i in subset if fused_scores[i] < other_scores[i])
    return aggregate_delta, subset_delta, len(subset), wins, losses


def check_identity(
    aggregate_delta: float, subset_delta: float, acted: int, n: int
) -> None:
    """``aggregate = (acted / n) x subset``, or the premise has broken.

    A same-order question scores identically under both systems by
    construction, so it contributes exactly zero to the aggregate difference.
    If this ever fails, ``relation`` and the scoring disagree about what "the
    same ranking" means and every table above it is uninterpretable.
    """
    predicted = (acted / float(n)) * subset_delta if n else 0.0
    if abs(predicted - aggregate_delta) > 1e-9:
        raise AssertionError(
            "decomposition broke: aggregate {:+.9f} but (acted/n) x subset is "
            "{:+.9f}. A question the reordering test called unchanged scored "
            "differently, so the two are not measuring the same ranking.".format(
                aggregate_delta, predicted
            )
        )


def format_payoff(
    label: str,
    other: str,
    fused: Sequence[Sequence[str]],
    other_rankings: Sequence[Sequence[str]],
    gold: Sequence[Dict],
    k: int,
    comparisons: Mapping[str, Comparison],
) -> str:
    n = len(gold)
    acted = [relation(f, o) != SAME_ORDER for f, o in zip(fused, other_rankings)]
    lines = [
        "{} vs {}: what the reordering bought".format(label, other),
        "{:<10}{:>11}{:>11}{:>11}{:>9}{:>11}{:>13}{:>20}".format(
            "metric",
            label[:9],
            other[:9],
            "delta",
            "acted",
            "on acted",
            "W/L/T",
            "95% CI (acted)",
        ),
        "-" * 96,
    ]
    for metric, _ in METRICS:
        fused_scores = [
            score(metric, ranking, record["relevant"], k)
            for ranking, record in zip(fused, gold)
        ]
        other_scores = [
            score(metric, ranking, record["relevant"], k)
            for ranking, record in zip(other_rankings, gold)
        ]
        aggregate_delta, subset_delta, count, wins, losses = decompose(
            fused_scores, other_scores, acted
        )
        check_identity(aggregate_delta, subset_delta, count, n)
        result = comparisons[metric]
        lines.append(
            "{:<10}{:>11.3f}{:>11.3f}{:>+11.4f}{:>9}{:>+11.4f}{:>13}{:>20}".format(
                metric,
                mean(fused_scores),
                mean(other_scores),
                aggregate_delta,
                count,
                subset_delta,
                "{}/{}/{}".format(wins, losses, count - wins - losses),
                "[{:+.3f}, {:+.3f}]".format(result.ci_low, result.ci_high),
            )
        )
    lines.append("")
    lines.append(
        "  `acted` counts questions where the fused ranking differs from this "
        "input's."
    )
    lines.append(
        "  `delta` = (acted / {}) x `on acted`, checked numerically, not "
        "asserted.".format(n)
    )
    lines.append(
        "  W/L/T is over the acted subset: T there means the fusion reordered "
        "and the"
    )
    lines.append("  metric did not notice, which is a different fact from a tie.")
    return "\n".join(lines)


def subset_comparisons(
    fused: Sequence[Sequence[str]],
    other_rankings: Sequence[Sequence[str]],
    gold: Sequence[Dict],
    k: int,
    args,
) -> Dict[str, Comparison]:
    """Paired test restricted to the questions the fusion actually reordered."""
    keep = [
        i
        for i in range(len(gold))
        if relation(fused[i], other_rankings[i]) != SAME_ORDER
    ]
    results = {}
    for metric, _ in METRICS:
        left = [score(metric, fused[i], gold[i]["relevant"], k) for i in keep]
        right = [
            score(metric, other_rankings[i], gold[i]["relevant"], k) for i in keep
        ]
        if not left:
            left, right = [0.0], [0.0]
        results[metric] = compare(
            left,
            right,
            bootstrap_resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            confidence=args.confidence,
            seed=args.seed,
        )
    return results


def verdict(
    label: str,
    other: str,
    fused: Sequence[Sequence[str]],
    other_rankings: Sequence[Sequence[str]],
    gold: Sequence[Dict],
    k: int,
    metric: str = "ndcg@5",
) -> str:
    """Name which of the two readings in the module docstring this pair fits."""
    n = len(gold)
    acted = [relation(f, o) != SAME_ORDER for f, o in zip(fused, other_rankings)]
    count = sum(acted)
    fused_scores = [
        score(metric, ranking, record["relevant"], k)
        for ranking, record in zip(fused, gold)
    ]
    other_scores = [
        score(metric, ranking, record["relevant"], k)
        for ranking, record in zip(other_rankings, gold)
    ]
    aggregate_delta, _, _, wins, losses = decompose(fused_scores, other_scores, acted)
    rate = count / float(n) if n else 0.0
    if rate < 0.10:
        reading = "INERT - the fusion returned this list on {:.0%} of questions".format(
            1.0 - rate
        )
    elif abs(aggregate_delta) < 0.005:
        reading = (
            "CANCELLATION - reordered {:.0%} of questions, {}W/{}L on {}, "
            "net {:+.4f}".format(rate, wins, losses, metric, aggregate_delta)
        )
    else:
        reading = "MOVED - reordered {:.0%}, net {:+.4f} on {} ({}W/{}L)".format(
            rate, aggregate_delta, metric, wins, losses
        )
    return "  {} vs {:<8} {}".format(label, other, reading)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("--slice", default="all", choices=("all", "easy", "hard"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--ngram-n", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    gold = select_slice(load_gold(args.gold), args.slice)
    shared = dict(
        target_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        rrf_k=args.rrf_k,
    )
    engines = {
        "base": RAGEngine.from_corpus(args.corpus, config=EngineConfig(**shared)),
        "ngram": RAGEngine.from_corpus(
            args.corpus, config=EngineConfig(ngram=args.ngram_n, **shared)
        ),
    }

    print(
        "corpus: {} documents, {} chunks   gold: {} questions ({} slice)   "
        "k={}".format(
            len(engines["base"].documents),
            len(engines["base"].chunks),
            len(gold),
            args.slice,
            args.k,
        )
    )

    verdicts: List[str] = []
    for label, engine_name, inputs in FUSIONS:
        engine = engines[engine_name]
        fused = rankings(engine, gold, "hybrid", args.k)
        input_rankings = {
            name: rankings(engine, gold, name, args.k) for name in inputs
        }

        print()
        print(format_relations(label, fused, input_rankings))

        for name in inputs:
            comparisons = subset_comparisons(
                fused, input_rankings[name], gold, args.k, args
            )
            print()
            print(
                format_payoff(
                    label,
                    name,
                    fused,
                    input_rankings[name],
                    gold,
                    args.k,
                    comparisons,
                )
            )
            verdicts.append(
                verdict(label, name, fused, input_rankings[name], gold, args.k)
            )

    print()
    print("Reading, on nDCG@5:")
    for line in verdicts:
        print(line)
    print()
    print(
        "INERT means the ablation table's tie is a report that fusion had "
        "nothing to do."
    )
    print(
        "CANCELLATION means the opposite: it acted constantly and the actions "
        "cancelled. The"
    )
    print(
        "ablation table prints the same number in both cases, which is why it "
        "cannot be read"
    )
    print("as evidence for either one.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
