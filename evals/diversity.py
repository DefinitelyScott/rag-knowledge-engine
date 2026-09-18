"""Does diversity re-ranking move recall - and was the MMR table measuring one system?

The MMR section of the README opens with the argument for the whole technique:

> A top-5 made of five paraphrases of one passage scores exactly the same as
> five complementary ones, but it is worth far less to an answerer.

That is a claim about **recall**. A result set that spends all five slots on one
document cannot hold the second document a two-document question needs, and
recall@k is the only metric in ``ragkb/metrics.py`` that notices. The sweep that
follows the paragraph reports hit@1, hit@5, MRR, redundancy and distinct
documents. It does not report recall, even though ``diversity_report`` in
``evals/evaluate.py`` computes it and drops it on the way out. So the section
argues for diversity on recall grounds and then scores it on metrics that are
structurally unable to see the thing it argued.

This script measures the missing column, and in the course of doing so found
that the table had a second and larger problem.

**Recall only carries information on the multi-document questions.** For a
question with exactly one relevant document, recall@k is 1.0 when that document
is in the top k and 0.0 otherwise - which is the definition of hit@k. The two
metrics are the *same number* on those questions, so averaging recall over the
whole gold set mostly re-reports hit@k with a minority of questions perturbing
it. Of the gold questions, only a minority have more than one relevant document,
and every one of them sits in the hard slice. That subset is the entire
independent content of recall@k, and it is what this script sweeps. The identity
is checked numerically rather than asserted, because it is the premise the rest
of the output rests on.

**The accuracy columns and the diversity columns were measured at different
selection depths.** MMR is greedy against a running penalty, so its output
depends on how many items it is asked to select: choosing 5 of 25 candidates is
an aggressive diversity selection, and choosing 20 of 25 can only drop five and
is close to inert. ``diversity_report`` takes its accuracy columns from
``ranked_doc_ids``, which searches at ``k * 4`` to have enough chunks left after
deduplicating to documents, and takes ``redundancy`` and ``distinct_docs`` from
``search`` at ``k``. Those are different configurations of the same re-ranker,
printed as one row. That is why the accuracy half of the table looked flat while
the diversity half moved: the flat half was measured under a setting where MMR
has almost nothing to do. ``depth_audit`` prints both depths side by side so the
gap is visible rather than inferred.

None of this says MMR helps. It says the table could not have shown it either
way. What the sweep below actually finds is reported as it comes.

Usage:
    python evals/diversity.py
    python evals/diversity.py --lambdas 0.9,0.7,0.5,0.3
    python evals/diversity.py --k 5 --slice multi
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine, read_corpus  # noqa: E402
from ragkb.metrics import hit_at_k, mean, recall_at_k  # noqa: E402
from ragkb.power import clopper_pearson, mcnemar_exact  # noqa: E402

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


def multi_document(gold: Sequence[Dict]) -> List[Dict]:
    """The questions with more than one relevant document.

    The only questions on which recall@k is not a restatement of hit@k, and
    therefore the only ones a diversity re-ranker can be scored on without the
    result being dominated by questions that cannot express the difference.
    """
    return [record for record in gold if len(record["relevant"]) > 1]


def recall_equals_hit(
    engine: RAGEngine, gold: Sequence[Dict], k: int
) -> Tuple[int, int]:
    """Check recall@k == hit@k on every single-relevant-document question.

    Returns ``(checked, mismatches)``. This is a premise of everything below -
    that averaging recall over the full gold set mostly re-reports hit@k - and a
    premise worth measuring rather than reasoning about, because it is cheap and
    because a mismatch would mean the gold set contains a duplicate relevant id
    or the metric does not do what its docstring says.
    """
    checked = 0
    mismatches = 0
    for record in gold:
        if len(record["relevant"]) != 1:
            continue
        ranked = engine.ranked_doc_ids(record["question"], k=k)
        checked += 1
        if recall_at_k(ranked, record["relevant"], k) != hit_at_k(
            ranked, record["relevant"], k
        ):
            mismatches += 1
    return checked, mismatches


def baseline_doc_lists(
    engine: RAGEngine, gold: Sequence[Dict], k: int
) -> List[List[str]]:
    """The top-``k`` document lists with re-ranking off, for churn comparisons."""
    return [engine.ranked_doc_ids(record["question"], k=k) for record in gold]


def depth_audit(
    engine: RAGEngine,
    gold: Sequence[Dict],
    k: int,
    mmr_lambda: Optional[float],
    baseline: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Diversity read-outs at both selection depths, for one lambda.

    ``shallow`` is ``search(k)`` - the depth the README's ``distinct_docs`` and
    ``redundancy`` columns are measured at. ``deep`` is ``search(k * 4)``, the
    depth ``ranked_doc_ids`` uses and therefore the one every accuracy column in
    the same table was actually produced under. ``kept_docs`` is how many
    distinct documents survive the truncation to ``k`` documents that evaluation
    applies.

    ``kept_docs`` alone would be misleading, and ``churn`` is the guard against
    misreading it. The truncation takes the first ``k`` *distinct* documents, so
    whenever the candidate pool holds at least ``k`` of them the count comes out
    at exactly ``k`` for every lambda - which says nothing about whether the
    documents are the same ones. ``churn`` counts how many of the ``k`` differ
    from the re-ranking-off list, and separates "MMR changes nothing a
    document-level metric can see" from "MMR swaps documents and the swaps do
    not pay".
    """
    deep_k = max(k * 4, k)
    shallow_docs, shallow_redundancy = [], []
    deep_docs, kept_docs, churn = [], [], []
    for record, base in zip(gold, baseline):
        shallow = engine.search(record["question"], k=k, mmr_lambda=mmr_lambda)
        shallow_docs.append(float(len({result.doc_id for result in shallow})))
        shallow_redundancy.append(engine.retriever.redundancy(shallow))

        deep = engine.search(record["question"], k=deep_k, mmr_lambda=mmr_lambda)
        deep_docs.append(float(len({result.doc_id for result in deep})))

        kept = engine.ranked_doc_ids(record["question"], k=k, mmr_lambda=mmr_lambda)
        kept_docs.append(float(len(kept)))
        churn.append(float(len(set(kept) - set(base))))
    return {
        "shallow_docs": mean(shallow_docs),
        "shallow_redund": mean(shallow_redundancy),
        "deep_docs": mean(deep_docs),
        "kept_docs": mean(kept_docs),
        "churn": mean(churn),
    }


def recall_profile(
    engine: RAGEngine, gold: Sequence[Dict], k: int, mmr_lambda: Optional[float]
) -> Dict[str, object]:
    """Per-question recall@k plus the document-level found/needed counts."""
    per_question: List[float] = []
    found = 0
    needed = 0
    for record in gold:
        ranked = engine.ranked_doc_ids(record["question"], k=k, mmr_lambda=mmr_lambda)
        per_question.append(recall_at_k(ranked, record["relevant"], k))
        relevant = set(record["relevant"])
        found += len(relevant & set(ranked[:k]))
        needed += len(relevant)
    return {
        "per_question": per_question,
        "recall": mean(per_question),
        "found": found,
        "needed": needed,
    }


def paired_change(
    baseline: Sequence[float], candidate: Sequence[float]
) -> Tuple[int, int, int, float]:
    """Questions improved, worsened, tied, and the exact paired p-value.

    Only the questions whose recall *moved* carry information about whether the
    re-ranker changed anything, which is the same argument :func:`mcnemar_exact`
    is built on: a question scoring identically under both settings is
    uninformative, not evidence of equivalence.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired series must be the same length")
    improved = sum(1 for base, new in zip(baseline, candidate) if new > base)
    worsened = sum(1 for base, new in zip(baseline, candidate) if new < base)
    tied = len(baseline) - improved - worsened
    return improved, worsened, tied, mcnemar_exact(improved, worsened)


def format_depth_table(rows: Sequence[Tuple[str, Dict[str, float]]]) -> str:
    columns = ["shallow_docs", "shallow_redund", "deep_docs", "kept_docs", "churn"]
    width = max([len("lambda")] + [len(name) for name, _ in rows]) + 2
    header = "{:<{w}}".format("lambda", w=width) + "".join(
        "{:>16}".format(name) for name in columns
    )
    lines = [header, "-" * len(header)]
    for name, report in rows:
        lines.append(
            "{:<{w}}".format(name, w=width)
            + "".join("{:>16.3f}".format(report[column]) for column in columns)
        )
    return "\n".join(lines)


def format_recall_table(rows: Sequence[Tuple[str, Dict[str, object]]]) -> str:
    width = max([len("lambda")] + [len(name) for name, _ in rows]) + 2
    header = (
        "{:<{w}}".format("lambda", w=width)
        + "{:>10}".format("recall")
        + "{:>14}".format("docs found")
        + "{:>18}".format("95% interval")
        + "{:>10}".format("W/L/T")
        + "{:>10}".format("p")
    )
    lines = [header, "-" * len(header)]
    for name, report in rows:
        low, high = report["interval"]  # type: ignore[index]
        counts = "{}/{}/{}".format(report["improved"], report["worsened"], report["tied"])
        lines.append(
            "{:<{w}}".format(name, w=width)
            + "{:>10.3f}".format(report["recall"])  # type: ignore[arg-type]
            + "{:>14}".format("{}/{}".format(report["found"], report["needed"]))
            + "{:>18}".format("{:.3f}-{:.3f}".format(low, high))
            + "{:>10}".format(counts)
            + "{:>10.3f}".format(report["p"])  # type: ignore[arg-type]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--lambdas",
        default="1.0,0.9,0.7,0.5,0.3",
        help="comma-separated MMR lambda values to sweep",
    )
    parser.add_argument(
        "--slice",
        choices=("multi", "all"),
        default="multi",
        help="'multi' scores only the multi-document questions (the default, and "
        "the only ones where recall@k differs from hit@k); 'all' scores the "
        "whole gold set for comparison",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    gold = load_gold(args.gold)
    documents = read_corpus(args.corpus)
    engine = RAGEngine(documents, EngineConfig())

    multi = multi_document(gold)
    single = len(gold) - len(multi)
    print(
        "corpus: {} documents, {} chunks   gold: {} questions, "
        "{} with one relevant document, {} with more".format(
            len(documents), len(engine.chunks), len(gold), single, len(multi)
        )
    )

    checked, mismatches = recall_equals_hit(engine, gold, args.k)
    print(
        "recall@{k} == hit@{k} on {checked}/{checked} single-document questions "
        "({mismatches} mismatches) - so all independent recall information "
        "lives in the {multi} multi-document questions.".format(
            k=args.k, checked=checked, mismatches=mismatches, multi=len(multi)
        )
    )

    scored = multi if args.slice == "multi" else gold
    if not scored:
        print("\nno questions in the requested slice; nothing to sweep.")
        return 0

    lambdas = [value.strip() for value in args.lambdas.split(",") if value.strip()]

    print(
        "\nselection depth audit ({} questions): where the diversification goes".format(
            len(scored)
        )
    )
    base_lists = baseline_doc_lists(engine, scored, args.k)
    depth_rows = [("off", depth_audit(engine, scored, args.k, None, base_lists))]
    for value in lambdas:
        depth_rows.append(
            (value, depth_audit(engine, scored, args.k, float(value), base_lists))
        )
    print(format_depth_table(depth_rows))
    print(
        "shallow_* is search(k={k}) - the depth the README's diversity columns use.\n"
        "deep_docs is search(k={deep}) - the depth every accuracy column in that\n"
        "same table was measured under. kept_docs is what survives truncation to\n"
        "{k} documents, and is what any doc-level metric can actually see; churn\n"
        "is how many of those {k} are different documents from the `off` list,\n"
        "which is the part a flat kept_docs column cannot tell you.".format(
            k=args.k, deep=max(args.k * 4, args.k)
        )
    )

    print(
        "\nrecall@{} on the {} slice ({} questions), paired against lambda off".format(
            args.k, args.slice, len(scored)
        )
    )
    baseline = recall_profile(engine, scored, args.k, None)
    rows: List[Tuple[str, Dict[str, object]]] = []
    for label, profile in [("off", baseline)] + [
        (value, recall_profile(engine, scored, args.k, float(value)))
        for value in lambdas
    ]:
        improved, worsened, tied, p_value = paired_change(
            baseline["per_question"],  # type: ignore[arg-type]
            profile["per_question"],  # type: ignore[arg-type]
        )
        rows.append(
            (
                label,
                {
                    "recall": profile["recall"],
                    "found": profile["found"],
                    "needed": profile["needed"],
                    "interval": clopper_pearson(
                        int(profile["found"]), int(profile["needed"])  # type: ignore[arg-type]
                    ),
                    "improved": improved,
                    "worsened": worsened,
                    "tied": tied,
                    "p": p_value,
                },
            )
        )
    print(format_recall_table(rows))
    print(
        "W/L/T counts questions whose recall@{} moved against `off`, and p is the\n"
        "exact paired sign test over the ones that moved. The interval is\n"
        "Clopper-Pearson on documents found out of documents needed; it treats\n"
        "those documents as independent draws, which they are not within a\n"
        "question, so read it as a scale rather than a decision rule - the W/L/T\n"
        "column and its p-value are the paired quantities.".format(args.k)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
