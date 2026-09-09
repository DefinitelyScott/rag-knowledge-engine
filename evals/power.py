"""How many questions would it take? Sample-size analysis for this gold set.

``evals/significance.py`` closes by telling the reader that the fix for an
UNTESTABLE verdict is "a bigger and more discriminative gold set". This script
is the other half of that sentence: it prices the advice.

Three blocks, in order of how much they depend on the data:

1. **What this gold set can resolve.** Fixed by ``n`` and the observed
   discordance rate alone, before any method is compared. It answers "what
   effect would have had to exist for the tables in this repo to have found it?"
2. **The grid.** Required questions across a range of hypothesised truths. This
   block looks at no results at all and would print the same numbers if every
   retriever in the project were replaced tomorrow. It is the reusable finding.
3. **The claims.** The comparisons the README asserts, re-run for their real
   win/loss counts, with the sample size each would need *if its observed split
   were the true one*.

Block 3 carries a hazard worth naming twice. Feeding observed counts back in is
**not** retrospective power, which is circular and uninformative - observed
power is a monotone function of the p-value, so reporting it restates the result
and calls it a diagnosis. The number printed here is conditional and
prospective: *if the truth were this, this is what it would cost to see it.*
The observed split is a hypothesis on eight disagreements, and eight
disagreements pin a proportion to roughly a third either way, so the sample
sizes in that block carry error bars far wider than the ones in block 2.

Usage:
    python evals/power.py
    python evals/power.py --target-power 0.9 --alpha 0.01
    python evals/power.py --metric hit@5 --slice hard
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.power import (  # noqa: E402
    minimum_discordant_pairs,
    rates_from_counts,
    required_questions,
    sign_test_power,
    smallest_detectable_win_share,
)
from ragkb.significance import compare, resolution  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Reuse the sibling harness rather than restating its routing table, gold
# loader and metric dispatch. Duplicating them is how two scripts end up
# scoring different things under one label, which is the exact failure the
# METHOD_ROUTING comment in significance.py was written to prevent.
_spec = importlib.util.spec_from_file_location(
    "significance_eval", os.path.join(HERE, "significance.py")
)
significance_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(significance_eval)

#: Hypothesised discordance rates for the grid, spanning "these questions barely
#: separate the methods" to "a third of them do".
GRID_RATES = (0.05, 0.10, 0.15, 0.20, 0.30)

#: Hypothesised win shares. 0.55 is a genuinely small effect; 0.90 is a rout.
GRID_SHARES = (0.55, 0.60, 0.65, 0.70, 0.80, 0.90)


def format_count(value):
    return "-" if value is None else str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--ngram-n", type=int, default=4)
    parser.add_argument("--metric", default="ndcg@5", help="metric for the claims block")
    parser.add_argument("--slice", default="all", help="gold slice for the claims block")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--max-questions", type=int, default=20000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from ragkb.engine import EngineConfig, RAGEngine

    gold = significance_eval.load_gold(args.gold)
    subset = significance_eval.select_slice(gold, args.slice)
    if not subset:
        raise SystemExit("empty slice: {}".format(args.slice))

    print(
        "sample-size analysis - alpha={:.2f}, target power={:.0%}, exact two-sided "
        "sign test".format(args.alpha, args.target_power)
    )
    print(
        "gold set: {} questions total, {} in the {} slice, one question is worth "
        "{:.3f}".format(len(gold), len(subset), args.slice, resolution(len(subset)))
    )

    engines = {
        "base": RAGEngine.from_corpus(
            args.corpus,
            config=EngineConfig(
                target_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
                rrf_k=args.rrf_k,
            ),
        ),
        "ngram": RAGEngine.from_corpus(
            args.corpus,
            config=EngineConfig(
                target_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
                rrf_k=args.rrf_k,
                ngram=args.ngram_n,
            ),
        ),
    }
    ranked = significance_eval.rankings_by_method(
        engines, significance_eval.METHOD_ROUTING, subset, args.k
    )
    relevancies = [record["relevant"] for record in subset]
    scores = {
        method: [
            significance_eval.metric_value(args.metric, ranking, relevant)
            for ranking, relevant in zip(ranked[method], relevancies)
        ]
        for method, _, _ in significance_eval.METHOD_ROUTING
    }

    results = {}
    for left, right, description in significance_eval.CLAIMS:
        results[(left, right)] = (
            compare(
                scores[left],
                scores[right],
                bootstrap_resamples=args.bootstrap_resamples,
                permutations=args.permutations,
                seed=args.seed,
            ),
            description,
        )

    observed_rates = [
        result.discordant / float(result.n) for result, _ in results.values()
    ]
    typical_rate = (
        sum(observed_rates) / len(observed_rates) if observed_rates else 0.0
    )

    # ---- block 1 -----------------------------------------------------------
    n = len(subset)
    print("\n" + "=" * 72)
    print("1. WHAT THIS GOLD SET CAN RESOLVE")
    print("=" * 72)
    print(
        "The three claim comparisons disagree on {:.1%} of questions on average, "
        "so a\ntypical comparison here has about {:.0f} discordant pairs out of "
        "{}.".format(typical_rate, typical_rate * n, n)
    )
    print(
        "Fewer than {} disagreements can never reach p<{:.2f} at any sample size."
        .format(minimum_discordant_pairs(args.alpha), args.alpha)
    )
    detectable = smallest_detectable_win_share(
        n, typical_rate, alpha=args.alpha, target_power=args.target_power
    )
    if detectable is None:
        print(
            "\nAt this discordance rate, {} questions cannot reach {:.0%} power "
            "against ANY\neffect size - not even a method that wins every single "
            "disagreement.".format(n, args.target_power)
        )
    else:
        print(
            "\nSmallest effect {} questions could detect at {:.0%} power: "
            "win share {:.3f}.".format(n, args.target_power, detectable)
        )
        print(
            "That is the bar: unless one method wins {:.0%} of the questions the "
            "two\ndisagree on, this gold set will usually fail to notice."
            .format(detectable)
        )
    print(
        "\nFor reference, power to detect a genuinely small effect (win share "
        "0.60)\nat this rate and size: {:.1%}.".format(
            sign_test_power(n, typical_rate, 0.60, args.alpha)
        )
    )

    # ---- block 2 -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("2. REQUIRED QUESTIONS BY ASSUMED TRUTH  (independent of any result)")
    print("=" * 72)
    header = "{:>12}".format("discordance") + "".join(
        "{:>9}".format("q={:.2f}".format(share)) for share in GRID_SHARES
    )
    print(header)
    print("-" * len(header))
    for rate in GRID_RATES:
        row = "{:>11.0%} ".format(rate)
        for share in GRID_SHARES:
            row += "{:>9}".format(
                format_count(
                    required_questions(
                        rate,
                        share,
                        alpha=args.alpha,
                        target_power=args.target_power,
                        max_questions=args.max_questions,
                    )
                )
            )
        print(row)
    print(
        "\n'-' means more than {} questions. q is the share of DISAGREEMENTS the "
        "better\nmethod wins; 0.50 is the null.".format(args.max_questions)
    )
    print(
        "\nRead the rows against the columns. Halving the effect roughly "
        "quadruples the\nquestions needed, but so does dividing the discordance "
        "rate by four - and the\ndiscordance rate is the cheaper lever, because "
        "it is bought by writing questions\nthat separate the methods rather "
        "than merely more of them. A question both\nmethods already answer "
        "correctly adds to n and contributes exactly nothing to\nthe test."
    )

    # ---- block 3 -----------------------------------------------------------
    print("\n" + "=" * 72)
    print(
        "3. THE PROJECT'S CLAIMS, PRICED  ({} / {} slice)".format(
            args.metric, args.slice
        )
    )
    print("=" * 72)
    row_format = "{:<22}{:>12}{:>10}{:>12}{:>14}"
    print(row_format.format("comparison", "W/L/T", "rate", "obs. share", "questions"))
    print("-" * 70)
    for (left, right), (result, _description) in results.items():
        rate, share = rates_from_counts(result.wins, result.losses, result.n)
        needed = (
            required_questions(
                rate,
                share,
                alpha=args.alpha,
                target_power=args.target_power,
                max_questions=args.max_questions,
            )
            if result.discordant
            else None
        )
        print(
            row_format.format(
                "{} vs {}".format(left, right),
                "{}/{}/{}".format(result.wins, result.losses, result.ties),
                "{:.1%}".format(rate),
                "{:.2f}".format(share),
                format_count(needed),
            )
        )
    print(
        "\nNOT retrospective power. Each figure answers 'if the observed split "
        "were the\ntrue one, how many questions would that effect need?' - a "
        "conditional\nstatement about a hypothesis, not a verdict on the test "
        "that produced it.\nObserved power is a monotone function of the p-value "
        "and would say nothing the\nsignificance table has not already said."
    )
    print(
        "An observed share near 0.50 means the estimated effect is near zero, so "
        "the\nrequired sample diverges - correctly. It reports 'no effect worth "
        "sizing was\nseen here', not 'this many questions would settle it'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
