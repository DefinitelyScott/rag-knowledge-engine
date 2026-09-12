"""Audit the gold set's hand-assigned `difficulty` label, then stratify by a measured one.

Three questions, in the order they have to be answered:

1. **Does the label mean anything?** The ``easy``/``hard`` split in
   ``evals/gold.jsonl`` was assigned by hand and every hard-slice number in the
   README rests on it. Section 1 tests it against measured containment (see
   :mod:`ragkb.difficulty`) with an unpaired label-shuffle permutation test.

2. **Is it one axis or two?** Section 2 crosstabs the label against the size of
   the answer key with an exact Fisher test. If the label is perfectly
   confounded with "how many documents answer this question", then a hard-slice
   result is a blend of a lexical effect and a multi-document effect, and the
   two have to be separated before either can be quoted. Section 3 separates
   them by holding the answer-key size fixed.

3. **Where does the third list's gain actually live?** Section 4 stratifies the
   ablation along measured containment rather than the hand label. The
   character n-gram list exists to match terms the word index scores at exactly
   zero, so its gain should concentrate at low containment and vanish at high.
   That is a falsifiable prediction about a mechanism, and it is a stronger
   claim than any aggregate delta because it says *where* the effect must be as
   well as how big.

Section 4 is the one worth reading first. An aggregate that improves tells you
a change helped; a gain that appears exactly in the band its mechanism predicts
and nowhere else tells you *why*, and would have been visible as a
contradiction had the mechanism been wrong.

Usage:
    python evals/difficulty.py
    python evals/difficulty.py --metric ndcg@5
    python evals/difficulty.py --edges 0.3,0.6
    python evals/difficulty.py --seed 11 --resamples 50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.difficulty import (  # noqa: E402
    BAND_EDGES,
    Overlap,
    band_label,
    group_by_band,
    label_multidoc_table,
    mean,
    overlaps,
)
from ragkb.engine import EngineConfig, RAGEngine, read_corpus  # noqa: E402
from ragkb.metrics import aggregate  # noqa: E402
from ragkb.power import clopper_pearson, fisher_exact  # noqa: E402
from ragkb.significance import two_sample_permutation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: The three systems the stratification compares. `hybrid3` is the shipped
#: configuration; `bm25` is the single-list baseline the third list has to beat;
#: `hybrid2` isolates whether the gain came from the third list or merely from
#: fusing more than one thing.
SYSTEMS = ("bm25", "hybrid2", "hybrid3")


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


def build_engines(documents: Mapping[str, str], ngram_weight: float) -> Dict:
    """One two-list engine and one three-list engine, built once and reused.

    The three-list engine carries the weighting the README ships
    (``ngram=4``), because the question is where the *shipped* system's gain
    lives, not where an unweighted variant's would.
    """
    two = RAGEngine(dict(documents), EngineConfig())
    three = RAGEngine(
        dict(documents),
        EngineConfig(ngram=4, fusion_weights={"ngram": ngram_weight}),
    )
    return {"bm25": (two, "bm25"), "hybrid2": (two, "hybrid"), "hybrid3": (three, "hybrid")}


def score(
    engines: Dict, subset: Sequence[Mapping], system: str, k: int, metric: str
) -> float:
    engine, method = engines[system]
    if not subset:
        return float("nan")
    rankings = [
        engine.ranked_doc_ids(record["question"], k=k, method=method)
        for record in subset
    ]
    relevancies = [record["relevant"] for record in subset]
    return aggregate(rankings, relevancies, ks=(1, 3, k))[metric]


def format_row(label: str, n: int, values: Sequence[float], width: int = 34) -> str:
    cells = "".join(
        "{:>15}".format("-" if value != value else "{:.3f}".format(value))
        for value in values
    )
    return "{:<{}} {:>4}{}".format(label, width, n, cells)


def format_header(width: int = 34) -> str:
    head = "{:<{}} {:>4}".format("slice", width, "n") + "".join(
        "{:>15}".format(name) for name in SYSTEMS
    )
    return head + "\n" + "-" * len(head)


def slice_table(
    engines: Dict,
    groups: Sequence[Tuple[str, Sequence[Mapping]]],
    k: int,
    metric: str,
) -> str:
    lines = [format_header()]
    for label, subset in groups:
        values = [score(engines, subset, system, k, metric) for system in SYSTEMS]
        lines.append(format_row(label, len(subset), values))
    return "\n".join(lines)


def label_audit(rows: Sequence[Overlap], seed: int, resamples: int) -> str:
    """Section 1: does the hand label separate on measured containment?"""
    easy = [row for row in rows if row.label != "hard"]
    hard = [row for row in rows if row.label == "hard"]
    out = [
        "{:<12}{:>5}{:>14}{:>18}{:>14}".format(
            "label", "n", "containment", "idf_containment", "mean_terms"
        ),
        "-" * 63,
    ]
    for name, group in (("easy", easy), ("hard", hard)):
        out.append(
            "{:<12}{:>5}{:>14.3f}{:>18.3f}{:>14.1f}".format(
                name,
                len(group),
                mean([row.containment for row in group]),
                mean([row.idf_containment for row in group]),
                mean([float(row.terms) for row in group]),
            )
        )
    p_plain = two_sample_permutation(
        [row.containment for row in easy],
        [row.containment for row in hard],
        resamples=resamples,
        seed=seed,
    )
    p_idf = two_sample_permutation(
        [row.idf_containment for row in easy],
        [row.idf_containment for row in hard],
        resamples=resamples,
        seed=seed,
    )
    floor = 1.0 / (resamples + 1)
    out.append("")
    out.append(
        "label-shuffle permutation test, {} resamples, seed {}:".format(resamples, seed)
    )
    for name, p_value in (("containment", p_plain), ("idf_containment", p_idf)):
        shown = (
            "< {:.2e}".format(floor) if p_value <= floor else "{:.4f}".format(p_value)
        )
        verdict = "SEPARATES" if p_value < 0.05 else "NOT SHOWN"
        out.append("  {:<18} p = {:<12} {}".format(name, shown, verdict))
    return "\n".join(out)


def confound_audit(rows: Sequence[Overlap]) -> str:
    """Section 2: is the label an axis of its own, or a proxy for answer-key size?"""
    hard_multi, hard_single, easy_multi, easy_single = label_multidoc_table(rows)
    p_value = fisher_exact(hard_multi, hard_single, easy_multi, easy_single)
    out = [
        "{:<12}{:>14}{:>16}".format("label", "multi-document", "single-document"),
        "-" * 42,
        "{:<12}{:>14}{:>16}".format("hard", hard_multi, hard_single),
        "{:<12}{:>14}{:>16}".format("easy", easy_multi, easy_single),
        "",
        "Fisher exact test of independence: p = {:.3e}".format(p_value),
    ]
    total_multi = hard_multi + easy_multi
    if total_multi:
        share = hard_multi / total_multi
        low, high = clopper_pearson(hard_multi, total_multi)
        out.append(
            "share of multi-document questions labelled hard: "
            "{:.3f} (95% CI {:.3f} - {:.3f})".format(share, low, high)
        )
    if p_value < 0.05:
        out.append(
            "CONFOUNDED: the label is not independent of answer-key size, so a "
            "hard-slice number mixes a lexical effect with a multi-document one."
        )
    else:
        out.append("the two axes are not shown to be dependent at this corpus size.")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(ROOT, "evals", "gold.jsonl"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--metric",
        default="hit@1",
        help="metric to stratify; any key aggregate() returns, e.g. ndcg@5, mrr",
    )
    parser.add_argument(
        "--edges",
        default=",".join("{:g}".format(edge) for edge in BAND_EDGES),
        help="comma-separated containment band edges",
    )
    parser.add_argument(
        "--length-edges",
        default="7,10,14",
        help="term-count edges for the negative-control table, chosen to give "
        "band counts comparable to the containment bands",
    )
    parser.add_argument("--ngram-weight", type=float, default=4.0)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    edges = tuple(float(part) for part in args.edges.split(",") if part.strip())
    length_edges = tuple(
        float(part) for part in args.length_edges.split(",") if part.strip()
    )
    documents = read_corpus(args.corpus)
    gold = load_gold(args.gold)
    rows = overlaps(gold, documents)
    by_id = {row.question_id: row for row in rows}

    print(
        "corpus: {} documents   gold set: {} questions   k={}   metric={}".format(
            len(documents), len(gold), args.k, args.metric
        )
    )
    print("containment bands: " + ", ".join(band_label(i, edges) for i in range(len(edges) + 1)))
    print()

    print("1. does the hand-assigned difficulty label measure anything?")
    print(label_audit(rows, args.seed, args.resamples))
    print()

    print("2. is it one axis, or a proxy for how many documents answer the question?")
    print(confound_audit(rows))
    print()

    engines = build_engines(documents, args.ngram_weight)

    print("3. the hand label, with answer-key size held fixed")
    single = [record for record in gold if len(record["relevant"]) == 1]
    groups = [
        ("hand easy (all)", [r for r in gold if r["difficulty"] != "hard"]),
        ("hand hard (all)", [r for r in gold if r["difficulty"] == "hard"]),
        ("hand easy, single-document", [r for r in single if r["difficulty"] != "hard"]),
        ("hand hard, single-document", [r for r in single if r["difficulty"] == "hard"]),
        (
            "hand hard, multi-document",
            [r for r in gold if r["difficulty"] == "hard" and len(r["relevant"]) > 1],
        ),
    ]
    print(slice_table(engines, groups, args.k, args.metric))
    print()
    print(
        "the two hard sub-slices are the point: if they disagree, the reported\n"
        "hard-slice figure is an average over sub-populations of unequal difficulty."
    )
    print()

    print("4. measured containment (idf-weighted), all questions")
    values = [by_id[record["id"]].idf_containment for record in gold]
    print(slice_table(engines, group_by_band(gold, values, edges), args.k, args.metric))
    print()
    print("   the same bands, single-document questions only")
    single_values = [by_id[record["id"]].idf_containment for record in single]
    print(
        slice_table(engines, group_by_band(single, single_values, edges), args.k, args.metric)
    )
    print()
    print(
        "prediction under test: the n-gram list matches terms the word index scores\n"
        "at exactly zero, so hybrid3 - bm25 should be largest in the lowest band and\n"
        "fall towards zero in the highest. A flat profile would mean the third list is\n"
        "buying something other than the lexical gap it was added for."
    )
    print()

    print("5. negative control: the same stratification along question length")
    length_values = [float(by_id[record["id"]].terms) for record in single]
    print(
        slice_table(
            engines,
            group_by_band(single, length_values, length_edges, upper=None),
            args.k,
            args.metric,
        )
    )
    print()
    print(
        "why this table is here: hard questions in this gold set are also longer\n"
        "questions ({} terms on average against {}), and containment's denominator is\n"
        "the term count, so a sceptic is owed the alternative explanation that the\n"
        "profile above is about length rather than lexical reach. Length is a\n"
        "genuine rival hypothesis and this is the table that tests it: if it\n"
        "reproduces the monotone collapse, containment is a restatement of length\n"
        "and adds nothing.".format(
            "{:.1f}".format(
                mean([float(r.terms) for r in rows if r.label == "hard"])
            ),
            "{:.1f}".format(
                mean([float(r.terms) for r in rows if r.label != "hard"])
            ),
        )
    )
    print()
    print(
        "caveat, stated where it applies: containment measures presence of the\n"
        "question's terms in its answer, not their discriminativeness against the\n"
        "rest of the corpus. High containment does not imply an easy question, and\n"
        "the top band is where that gap shows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
