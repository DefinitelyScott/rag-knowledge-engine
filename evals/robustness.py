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

Four tables:

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

3. **Paired significance under noise.** ``sd(seeds)`` bounds only the variance
   from drawing the noise. The larger uncertainty is which questions happen to
   be in the gold set, and until this table existed the README named that as
   an open gap. Each question's nDCG is averaged over the noise draws, so the
   question stays the paired unit, and the same paired permutation test and
   bootstrap that ``evals/significance.py`` uses are run on three pairs at the
   clean rate and at the noisiest rate, with Holm's correction over those six.
   The test is conditional on the noise *distribution*: it asks whether the
   ranking would survive a different gold set under the same typo model, not a
   different typo model.

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
from ragkb.metrics import aggregate, mean, ndcg_at_k  # noqa: E402
from ragkb.perturb import TypoConfig, character_error_rate, perturb  # noqa: E402
from ragkb.significance import Comparison, compare, holm_bonferroni  # noqa: E402

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

#: The pairs the significance table tests, as ``(a, b)`` with the delta read as
#: ``a - b``. One per claim the README makes about this harness: the third list
#: alone against the fusion that contains it, the three-way fusion against the
#: two-way one, and the two-way fusion against its stronger input.
SIGNIFICANCE_PAIRS = (
    ("ngram", "hybrid3"),
    ("hybrid3", "hybrid2"),
    ("hybrid2", "bm25"),
)


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


def per_question_ndcg(
    engines: Dict[str, RAGEngine],
    gold: Sequence[Dict],
    k: int,
    rate: float,
    seeds: Sequence[int],
) -> Dict[str, List[float]]:
    """Each method's nDCG@k per gold question, averaged over the noise draws.

    Averaging inside the question rather than across it keeps the question as
    the unit a paired test resamples, which is what makes these vectors
    comparable with every other significance figure in the repo. At rate 0.0
    one seed is used, for the same reason :func:`sweep` uses one: every seed
    yields the identical query.
    """
    active_seeds = [seeds[0]] if rate == 0.0 else list(seeds)
    totals: Dict[str, List[float]] = {
        name: [0.0] * len(gold) for name, _, _ in METHOD_ROUTING
    }
    for seed in active_seeds:
        questions = corrupt(gold, TypoConfig(rate=rate, seed=seed))
        for name, engine_key, method in METHOD_ROUTING:
            engine = engines[engine_key]
            for index, (record, question) in enumerate(zip(gold, questions)):
                ranked = engine.ranked_doc_ids(question, k=k, method=method)
                totals[name][index] += ndcg_at_k(ranked, record["relevant"], k)
    return {
        name: [value / len(active_seeds) for value in values]
        for name, values in totals.items()
    }


def significance_rows(
    scores_by_rate: Dict[float, Dict[str, List[float]]],
    pairs: Sequence[Tuple[str, str]] = SIGNIFICANCE_PAIRS,
    permutations: int = 10000,
    seed: int = 0,
) -> List[Tuple[float, str, str, Comparison, float, bool]]:
    """Paired tests for every pair at every rate, Holm-corrected as one family.

    Returns ``(rate, a, b, comparison, adjusted_p, reject)`` rows in rate order.
    The family is every row printed, so adding a rate or a pair tightens the
    correction on all of them rather than letting the new rows ride for free.
    """
    raw: List[Tuple[float, str, str, Comparison]] = []
    for rate in sorted(scores_by_rate):
        scores = scores_by_rate[rate]
        for a, b in pairs:
            raw.append(
                (rate, a, b, compare(scores[a], scores[b], permutations=permutations, seed=seed))
            )
    adjusted = holm_bonferroni([row[3].p_value for row in raw])
    return [
        (rate, a, b, comparison, p_adj, reject)
        for (rate, a, b, comparison), (p_adj, reject) in zip(raw, adjusted)
    ]


def format_significance(rows, k: int) -> str:
    header = "{:<8}{:<20}{:>9}{:>21}{:>10}{:>14}".format(
        "rate", "pair", "delta", "95% CI", "p_holm", "W/L/T"
    )
    lines = [header, "-" * len(header)]
    for rate, a, b, comparison, p_adj, _ in rows:
        lines.append(
            "{:<8}{:<20}{:>+9.3f}{:>21}{:>10.3f}{:>14}".format(
                "clean" if rate == 0.0 else "{:.0%}".format(rate),
                "{} vs {}".format(a, b),
                comparison.delta,
                "[{:+.3f}, {:+.3f}]".format(comparison.ci_low, comparison.ci_high),
                p_adj,
                "{}/{}/{}".format(comparison.wins, comparison.losses, comparison.ties),
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

    scores_by_rate = {
        rate: per_question_ndcg(engines, gold, args.k, rate, seeds)
        for rate in (0.0, worst)
    }
    rows = significance_rows(scores_by_rate)
    print(
        "\npaired significance on nDCG@{} (per-question mean over {} noise draws;\n"
        "permutation test + bootstrap CI, Holm over all {} rows)".format(
            args.k, args.seeds, len(rows)
        )
    )
    print(format_significance(rows, args.k))
    print(
        "\nsd(seeds) above is the noise-draw spread; this table is the gold-set\n"
        "spread, which is the larger of the two and the one a new question moves."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
