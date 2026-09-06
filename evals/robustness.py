"""Measure how each retrieval method degrades when the query is misspelled.

`evals/evaluate.py` scores every method on queries that were typed correctly.
That flatters the two word-level indexes and tests the third list under the one
condition it was never expected to help in. The character n-gram list was added
because BM25 and TF-IDF tokenise into whole words and therefore "miss the same
misspelling and the same morphological variant" (:mod:`ragkb.retriever`) - a
claim about *noisy* input that the clean tables cannot confirm or refute. This
script supplies the missing condition.

Every gold question is corrupted by :mod:`ragkb.perturb` at a controlled
character error rate and re-scored. What matters is not the absolute number at
any rate - a corpus of this size saturates - but the *shape of the decline*: a
method whose accuracy falls off a cliff between 5% and 20% noise is a method
whose users will feel every typo they make.

Three tables:

0. **Realised noise.** The requested rate and the achieved rate differ, because
   tokens shorter than four characters are skipped and a budget of edits rounds
   to a whole number. Reporting the achieved rate first means every row below is
   labelled with damage that was measured rather than damage that was intended.

1. **Accuracy at each rate.** One table per rate, methods as rows, so the
   columns line up with `evaluate.py` and the two can be read side by side. The
   clean row at rate 0.0 is the identity check: it must reproduce the numbers
   from the eval harness exactly, or the corruption is being applied where it
   should not be.

2. **Retention.** The summary that actually answers the question: each method's
   nDCG@k at the highest rate as a fraction of its own clean score, so a weak
   method that degrades gracefully is not mistaken for a strong one. Retention
   is reported with the standard deviation across seeds, because a gap in
   retention that is narrower than the seed spread is not a finding.

Every rate above zero is averaged over several seeds. One draw of the noise is a
single sample of a random variable, and comparing methods on one draw measures
the draw as much as the methods.

Usage:
    python evals/robustness.py [--corpus corpus] [--k 5]
                               [--rates 0.05,0.1,0.2] [--seeds 7] [--ngram-n 4]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import textwrap
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.metrics import aggregate, mean  # noqa: E402
from ragkb.perturb import TypoConfig, character_error_rate, perturb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Which engine serves which method. The two-list engine and the three-list
#: engine share identical BM25 and TF-IDF indexes, so routing the single-list
#: rows through the base engine costs nothing and keeps the comparison honest.
METHOD_ROUTING = (
    ("bm25", "base", "bm25"),
    ("vector", "base", "vector"),
    ("ngram", "ngram", "ngram"),
    ("hybrid2", "base", "hybrid"),
    ("hybrid3", "ngram", "hybrid"),
)

COLUMNS = ("hit@1", "hit@{k}", "ndcg@{k}", "mrr")


def load_questions(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def corrupt(gold: Sequence[Dict], config: TypoConfig) -> List[str]:
    return [perturb(record["question"], config) for record in gold]


def score(
    engine: RAGEngine,
    method: str,
    questions: Sequence[str],
    relevancies: Sequence[Sequence[str]],
    k: int,
) -> Dict[str, float]:
    rankings = [
        engine.ranked_doc_ids(question, k=k, method=method) for question in questions
    ]
    report = aggregate(rankings, relevancies, ks=(1, k))
    return {
        column.format(k=k): report[column.format(k=k)] for column in COLUMNS
    }


def format_table(reports: Dict[str, Dict[str, float]], label: str = "method") -> str:
    columns = list(next(iter(reports.values())).keys())
    header = "{:<10}".format(label) + "".join(
        "{:>13}".format(name) for name in columns
    )
    lines = [header, "-" * len(header)]
    for name, report in reports.items():
        lines.append(
            "{:<10}".format(name)
            + "".join("{:>13.3f}".format(report[column]) for column in columns)
        )
    return "\n".join(lines)


def sweep(
    engines: Dict[str, RAGEngine],
    gold: Sequence[Dict],
    k: int,
    rates: Sequence[float],
    seeds: Sequence[int],
) -> Tuple[Dict[float, Dict[str, Dict[str, float]]], Dict[float, Dict[str, List[float]]], Dict[float, float]]:
    """Score every method at every rate, averaged over ``seeds``.

    Returns the mean report per rate and method, the per-seed nDCG values (kept
    so the summary can report a spread rather than a bare mean), and the
    realised character error rate per requested rate.
    """
    relevancies = [record["relevant"] for record in gold]
    ndcg_key = "ndcg@{}".format(k)

    means: Dict[float, Dict[str, Dict[str, float]]] = {}
    per_seed: Dict[float, Dict[str, List[float]]] = {}
    realised: Dict[float, float] = {}

    for rate in rates:
        # At rate 0.0 the corruption is the identity, so every seed produces the
        # same queries and running more than one of them measures nothing.
        active_seeds = [seeds[0]] if rate == 0.0 else list(seeds)
        collected: Dict[str, List[Dict[str, float]]] = {
            name: [] for name, _, _ in METHOD_ROUTING
        }
        errors: List[float] = []

        for seed in active_seeds:
            config = TypoConfig(rate=rate, seed=seed)
            questions = corrupt(gold, config)
            errors.extend(
                character_error_rate(record["question"], noisy)
                for record, noisy in zip(gold, questions)
            )
            for name, engine_key, method in METHOD_ROUTING:
                collected[name].append(
                    score(engines[engine_key], method, questions, relevancies, k)
                )

        realised[rate] = mean(errors)
        means[rate] = {
            name: {
                column: mean([report[column] for report in reports])
                for column in reports[0]
            }
            for name, reports in collected.items()
        }
        per_seed[rate] = {
            name: [report[ndcg_key] for report in reports]
            for name, reports in collected.items()
        }
    return means, per_seed, realised


def retention_table(
    means: Dict[float, Dict[str, Dict[str, float]]],
    per_seed: Dict[float, Dict[str, List[float]]],
    clean_rate: float,
    noisy_rate: float,
    k: int,
) -> str:
    ndcg_key = "ndcg@{}".format(k)
    header = "{:<10}{:>13}{:>13}{:>13}{:>13}".format(
        "method", "clean", "noisy", "retained", "sd(seeds)"
    )
    lines = [header, "-" * len(header)]
    for name, _, _ in METHOD_ROUTING:
        clean = means[clean_rate][name][ndcg_key]
        noisy = means[noisy_rate][name][ndcg_key]
        values = per_seed[noisy_rate][name]
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        retained = noisy / clean if clean > 0.0 else 0.0
        lines.append(
            "{:<10}{:>13.3f}{:>13.3f}{:>12.1f}%{:>13.3f}".format(
                name, clean, noisy, 100.0 * retained, spread
            )
        )
    return "\n".join(lines)


def verdict(
    means: Dict[float, Dict[str, Dict[str, float]]],
    clean_rate: float,
    noisy_rate: float,
    k: int,
) -> str:
    """State what the numbers just said about the third list, from the numbers.

    Written as a computation rather than as prose so it cannot drift out of
    agreement with the table above it when the corpus grows. Two questions are
    settled here, and they have different answers: whether adding the n-gram
    list to the fusion buys robustness, and whether the fusion is the best use
    of that list once you have it.
    """
    ndcg = "ndcg@{}".format(k)

    def retained(name: str) -> float:
        clean = means[clean_rate][name][ndcg]
        return means[noisy_rate][name][ndcg] / clean if clean > 0.0 else 0.0

    two, three, alone = retained("hybrid2"), retained("hybrid3"), retained("ngram")
    noisy_three = means[noisy_rate]["hybrid3"][ndcg]
    noisy_alone = means[noisy_rate]["ngram"][ndcg]

    lines = []
    if three > two:
        lines.append(
            "the third list does buy robustness: adding it lifts retention from "
            "{:.1f}% to {:.1f}%.".format(100.0 * two, 100.0 * three)
        )
    else:
        lines.append(
            "the third list buys no robustness here: retention moves from "
            "{:.1f}% to {:.1f}%.".format(100.0 * two, 100.0 * three)
        )
    if noisy_alone > noisy_three:
        lines.append(
            "but the fusion is not the way to spend it. The n-gram list alone "
            "scores {:.3f} nDCG@{} at this noise level against {:.3f} for the "
            "three-way fusion, and retains {:.1f}% of its clean score against "
            "{:.1f}%. Fusing a noise-tolerant list with two brittle ones lets "
            "the brittle ones vote, and under noise their votes are wrong "
            "together - the same `ngram only` control that evaluate.py flags on "
            "clean text, failing here far more loudly.".format(
                noisy_alone, k, noisy_three, 100.0 * alone, 100.0 * three
            )
        )
    else:
        lines.append(
            "and the fusion is the way to spend it: the three-way fusion scores "
            "{:.3f} nDCG@{} against {:.3f} for the n-gram list alone.".format(
                noisy_three, k, noisy_alone
            )
        )
    return "\n".join(textwrap.fill(line, width=78) for line in lines)


def parse_rates(text: str) -> List[float]:
    rates = [0.0]
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if not 0.0 < value <= 1.0:
            raise SystemExit("rates must be in (0, 1]: {}".format(piece))
        rates.append(value)
    if len(rates) < 2:
        raise SystemExit("at least one non-zero rate is required")
    return rates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--rates",
        default="0.05,0.1,0.2",
        help="comma-separated character error rates; 0.0 is always included",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=7,
        help="number of noise draws to average at each non-zero rate",
    )
    parser.add_argument(
        "--ngram-n",
        type=int,
        default=4,
        help="n for the character n-gram index that feeds the third list",
    )
    args = parser.parse_args(argv)

    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")

    gold = load_questions(args.gold)
    rates = parse_rates(args.rates)
    seeds = list(range(args.seeds))

    engines = {
        "base": RAGEngine.from_corpus(args.corpus),
        "ngram": RAGEngine.from_corpus(
            args.corpus, config=EngineConfig(ngram=args.ngram_n)
        ),
    }

    print(
        "\nquery robustness: {} questions, {} documents, n={} character grams, "
        "{} seeds per rate".format(
            len(gold),
            len(engines["base"].documents),
            args.ngram_n,
            args.seeds,
        )
    )

    means, per_seed, realised = sweep(engines, gold, args.k, rates, seeds)

    print("\nrealised noise (mean character error rate over the gold questions)")
    print(
        format_table(
            {
                "{:.0%}".format(rate): {"realised": realised[rate]}
                for rate in rates
            },
            label="requested",
        )
    )
    print(
        "\nthe realised rate trails the requested one because tokens shorter than\n"
        "four characters are never corrupted and the edit budget rounds down."
    )

    for rate in rates:
        title = "clean" if rate == 0.0 else "{:.0%} character error rate".format(rate)
        print("\nretrieval accuracy - {} (k={})".format(title, args.k))
        print(format_table(means[rate]))

    worst = rates[-1]
    print(
        "\nnDCG@{} retention at {:.0%} noise (noisy / that method's own clean score)".format(
            args.k, worst
        )
    )
    print(retention_table(means, per_seed, 0.0, worst, args.k))
    print(
        "\nretention is the column to read, not the noisy score: a method can hold a\n"
        "higher absolute nDCG while losing more of what it started with. A retention\n"
        "gap narrower than the sd(seeds) column is a draw, not a result."
    )
    print("\n" + verdict(means, 0.0, worst, args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
