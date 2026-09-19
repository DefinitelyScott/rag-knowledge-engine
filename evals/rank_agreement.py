"""Restate the concurrence claim in the unit RRF actually reads.

The README explains why the `rrf_k` sweep found nothing to tune with two lists
like this:

    the two lists agree on 0.838 of their top-5 by chunk, and a fusion constant
    has no room to arbitrate lists that already concur

That is a set statistic doing an ordinal job. ``1 / (rrf_k + rank)`` consumes
rank and nothing else, so a claim about what a fusion constant can arbitrate has
to be made in a measure that can see order. This script makes it in three, and
reports what changes.

Three questions, in order:

1.  **Do the lists concur?** A pairwise table of every retriever against every
    other, in set overlap, RBO and Kendall tau-b, with the two unarguable
    endpoints - identical top-5, and same top-1 - beside them.

2.  **Does the claim survive the change of unit?** The claim is comparative: it
    says BM25 and TF-IDF concur *more* than a structurally different list would.
    That either holds in every measure or it does not, and the size of the gap
    is the part at risk.

3.  **Does the mechanism it asserts hold?** The claim is not only that the lists
    concur, it is that concurrence is *why* `rrf_k` is inert. That predicts
    something testable and falsifiable: the questions where `rrf_k` does change
    the fused ordering should be the questions where the lists agree least. If
    the two groups have the same rank agreement, the concurrence story is a
    coincidence and the real cause is elsewhere. Tested with the unpaired
    permutation test in :mod:`ragkb.significance`, because the two groups are
    disjoint sets of different questions of very unequal size.

Usage:
    python evals/rank_agreement.py [--corpus corpus] [--k 5] [--ngram-n 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.metrics import mean  # noqa: E402
from ragkb.rankagree import (  # noqa: E402
    DEFAULT_RBO_P,
    compare_rankings,
    rank_biased_overlap,
)
from ragkb.significance import two_sample_permutation  # noqa: E402
from ragkb.text import tokenize  # noqa: E402


def load_gold(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def per_list_rankings(
    engine: RAGEngine, question: str, k: int
) -> Dict[str, List[int]]:
    """The raw top-``k`` of each *input* list, before any fusion.

    Deliberately ``retriever._lists()`` rather than ``engine.search`` - the
    subject here is what the fusion is handed, not what it returns. Going
    through ``search`` would measure the fused output against itself.
    """
    tokens = tokenize(question)
    return {
        name: [index for index, _ in retriever.search(tokens, k=k)]
        for name, retriever in engine.retriever._lists()
    }


def pairwise_table(
    engine: RAGEngine, gold: Sequence[Dict], k: int, p: float
) -> List[Tuple[str, Dict[str, float]]]:
    """Every input list against every other, averaged over the gold set."""
    names = [name for name, _ in engine.retriever._lists()]
    pairs = [
        (names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
    ]
    accumulated: Dict[Tuple[str, str], Dict[str, List[float]]] = {
        pair: {
            "set": [],
            "rbo": [],
            "tau": [],
            "conjoint": [],
            "identical": [],
            "head": [],
        }
        for pair in pairs
    }

    for record in gold:
        tops = per_list_rankings(engine, record["question"], k)
        for pair in pairs:
            left, right = tops[pair[0]], tops[pair[1]]
            result = compare_rankings(left, right, depth=k, p=p)
            bucket = accumulated[pair]
            bucket["set"].append(result.set_overlap)
            bucket["rbo"].append(result.rbo)
            bucket["conjoint"].append(float(result.conjoint))
            bucket["identical"].append(1.0 if result.identical else 0.0)
            bucket["head"].append(1.0 if result.same_head else 0.0)
            # A tau of ``None`` is "undefined on fewer than two shared items",
            # not "zero correlation". Averaging it in as 0.0 would drag every
            # cross-representation pair toward the middle for the one reason
            # that should push it to the edge, so it is dropped and the count
            # of surviving questions is printed instead.
            if result.tau is not None:
                bucket["tau"].append(result.tau)

    rows = []
    for pair in pairs:
        bucket = accumulated[pair]
        rows.append(
            (
                "{} vs {}".format(*pair),
                {
                    "set_overlap": mean(bucket["set"]),
                    "rbo": mean(bucket["rbo"]),
                    "tau_b": mean(bucket["tau"]) if bucket["tau"] else float("nan"),
                    "tau_n": float(len(bucket["tau"])),
                    "conjoint": mean(bucket["conjoint"]),
                    "same_top1": mean(bucket["head"]),
                    "identical": mean(bucket["identical"]),
                },
            )
        )
    return rows


def format_pairwise(rows: Sequence[Tuple[str, Dict[str, float]]]) -> str:
    columns = [
        ("set overlap", "set_overlap", "{:.3f}"),
        ("RBO", "rbo", "{:.3f}"),
        ("tau_b", "tau_b", "{:.3f}"),
        ("tau n", "tau_n", "{:.0f}"),
        ("conjoint", "conjoint", "{:.2f}"),
        ("same top1", "same_top1", "{:.3f}"),
        ("identical", "identical", "{:.3f}"),
    ]
    width = max((len(label) for label, _ in rows), default=10) + 2
    header = "{:<{w}}".format("pair", w=width) + "".join(
        "{:>13}".format(title) for title, _, _ in columns
    )
    lines = [header, "-" * len(header)]
    for label, values in rows:
        line = "{:<{w}}".format(label, w=width)
        for _, key, fmt in columns:
            line += "{:>13}".format(fmt.format(values[key]))
        lines.append(line)
    return "\n".join(lines)


def mean_pairwise_rbo(tops: Dict[str, List[int]], k: int, p: float) -> float:
    """One number per question: how much its input lists agree overall."""
    names = sorted(tops)
    values = [
        rank_biased_overlap(tops[names[i]], tops[names[j]], p=p, depth=k)[0]
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    return mean(values) if values else 0.0


def rrf_sensitivity(
    corpus: str,
    gold: Sequence[Dict],
    k: int,
    p: float,
    ngram: int = 0,
    low: int = 1,
    high: int = 60,
    chunk_tokens: int = 120,
    overlap_tokens: int = 30,
) -> Dict[str, float]:
    """Split the gold set by whether ``rrf_k`` moves that question's ordering.

    Then ask whether the two groups differ in how much their input lists
    agreed in the first place. That is the mechanism the README asserts, stated
    so that it could come out false.
    """

    def build(rrf_k: int) -> RAGEngine:
        return RAGEngine.from_corpus(
            corpus,
            config=EngineConfig(
                target_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                rrf_k=rrf_k,
                ngram=ngram or None,
            ),
        )

    engine_low, engine_high = build(low), build(high)
    changed: List[float] = []
    unchanged: List[float] = []
    for record in gold:
        question = record["question"]
        # Input lists do not depend on rrf_k, so either engine measures the
        # same agreement. The fused orderings are what differ.
        agreement = mean_pairwise_rbo(
            per_list_rankings(engine_high, question, k), k, p
        )
        order_low = [
            result.chunk_index for result in engine_low.search(question, k=k)
        ]
        order_high = [
            result.chunk_index for result in engine_high.search(question, k=k)
        ]
        (changed if order_low != order_high else unchanged).append(agreement)

    return {
        "n_changed": float(len(changed)),
        "n_unchanged": float(len(unchanged)),
        "mean_changed": mean(changed) if changed else float("nan"),
        "mean_unchanged": mean(unchanged) if unchanged else float("nan"),
        "delta": (mean(unchanged) - mean(changed))
        if changed and unchanged
        else float("nan"),
        "p_value": two_sample_permutation(changed, unchanged, resamples=20000, seed=7),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--gold", default=os.path.join("evals", "gold.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--rrf-k-low",
        type=int,
        default=1,
        help="the contrasting rrf_k for the sensitivity split",
    )
    parser.add_argument("--ngram-n", type=int, default=4)
    parser.add_argument("--rbo-p", type=float, default=DEFAULT_RBO_P)
    args = parser.parse_args()

    gold = load_gold(args.gold)

    def build(ngram: int) -> RAGEngine:
        return RAGEngine.from_corpus(
            args.corpus,
            config=EngineConfig(
                target_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
                rrf_k=args.rrf_k,
                ngram=ngram or None,
            ),
        )

    engine = build(args.ngram_n)
    stats = engine.stats()
    print(
        "corpus: {} documents, {} chunks   gold set: {} questions   k={}   "
        "RBO p={} (residual {:.3f})".format(
            int(stats["documents"]),
            int(stats["chunks"]),
            len(gold),
            args.k,
            args.rbo_p,
            args.rbo_p**args.k,
        )
    )

    print("\npairwise agreement between the retrievers RRF is handed")
    print(format_pairwise(pairwise_table(engine, gold, args.k, args.rbo_p)))
    print(
        "\n'identical' is the share of questions where the two lists returned the\n"
        "same {} chunks in the same order. 'tau n' is how many questions had at\n"
        "least two shared items, without which tau-b is undefined - read it before\n"
        "reading tau_b, and read 'conjoint' before trusting it.".format(args.k)
    )

    print("\ndoes rrf_k move the questions where the lists agree least?")
    header = "{:<12}{:>11}{:>13}{:>13}{:>10}{:>11}".format(
        "lists", "changed", "RBO changed", "RBO same", "delta", "p"
    )
    print(header)
    print("-" * len(header))
    for label, ngram in (("two", 0), ("three", args.ngram_n)):
        result = rrf_sensitivity(
            args.corpus,
            gold,
            args.k,
            args.rbo_p,
            ngram=ngram,
            low=args.rrf_k_low,
            high=args.rrf_k,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
        )
        print(
            "{:<12}{:>11}{:>13.3f}{:>13.3f}{:>10.3f}{:>11.4f}".format(
                label,
                "{:.0f}/{:.0f}".format(
                    result["n_changed"], result["n_changed"] + result["n_unchanged"]
                ),
                result["mean_changed"],
                result["mean_unchanged"],
                result["delta"],
                result["p_value"],
            )
        )
    print(
        "\n'changed' counts questions whose fused top-{} ordering differs between\n"
        "rrf_k={} and rrf_k={}. A positive delta means the questions rrf_k moves are\n"
        "the ones whose input lists agreed least, which is the mechanism the README\n"
        "asserts. p is two-sided and unpaired: the groups are different questions.".format(
            args.k, args.rrf_k_low, args.rrf_k
        )
    )


if __name__ == "__main__":
    main()
