"""Is the n-gram list's lead an artefact of RRF ignoring scores?

The README's standing finding is that the character n-gram list alone beats
every fusion that contains it. Weighted RRF was one answer to that ("trust the
strong list more"). This is the other obvious one: RRF reads only ranks, so it
cannot tell a list that was sure of its top hit from one that was guessing.
CombSUM and CombMNZ over normalised scores can. If the dilution is a property
of *rank-only* arithmetic, score fusion should close the gap to ``ngram``.

What this script does, in order:

1. **Identity check.** Re-derives two-list and three-list RRF through
   :mod:`ragkb.scorefusion` and asserts that every question's ranked document
   list is identical to ``RAGEngine.ranked_doc_ids``. If that fails, nothing
   below is comparing like with like, so the script exits non-zero.
2. **Descriptive table** for all, easy and hard slices: RRF, CombSUM and
   CombMNZ under min-max and z-score normalisation, over two and three lists,
   with ``ngram`` alone as the control.
3. **Two pre-specified paired tests** on all-slice nDCG@5, Holm-corrected as a
   family of two. The score-fusion arm is fixed to CombSUM with min-max
   normalisation over three lists, because that is the textbook default (Fox &
   Shaw 1994, Lee 1997) - *not* because it is the best row in the table. An
   exploratory pass ran before this script was written and is why that
   distinction is stated rather than assumed: picking the arm after seeing the
   table is the selection bias `evals/holdout.py` exists to price.

   - ``combsum3 vs hybrid3``: does reading scores beat ranks on the same lists?
   - ``combsum3 vs ngram``: does it recover the lead that RRF gave away?

Usage:
    python evals/score_fusion.py
    python evals/score_fusion.py --permutations 50000 --seed 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.metrics import (  # noqa: E402
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragkb.scorefusion import ranked_doc_ids  # noqa: E402
from ragkb.significance import compare, holm_bonferroni  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TWO = ("bm25", "vector")
THREE = ("bm25", "vector", "ngram")

#: (row label, lists, combination, normalisation). ``rrf`` rows come from the
#: engine itself, not from :mod:`ragkb.scorefusion`; the identity check is what
#: licenses putting them in one table with the rows that do.
SCORE_ROWS = (
    ("combsum2 minmax", TWO, "combsum", "minmax"),
    ("combsum2 zscore", TWO, "combsum", "zscore"),
    ("combmnz2 minmax", TWO, "combmnz", "minmax"),
    ("combmnz2 zscore", TWO, "combmnz", "zscore"),
    ("combsum3 minmax", THREE, "combsum", "minmax"),
    ("combsum3 zscore", THREE, "combsum", "zscore"),
    ("combmnz3 minmax", THREE, "combmnz", "minmax"),
    ("combmnz3 zscore", THREE, "combmnz", "zscore"),
)

#: The pre-specified score-fusion arm, and what it is tested against.
PRIMARY = "combsum3 minmax"
PRIMARY_TESTS = ("hybrid3", "ngram")
PRIMARY_METRIC = "ndcg@5"

METRICS = ("hit@1", "hit@5", "recall@5", "ndcg@5", "mrr")


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


def score(metric: str, ranked: Sequence[str], relevant: Sequence[str]) -> float:
    if metric == "mrr":
        return reciprocal_rank(ranked, relevant)
    name, _, depth = metric.partition("@")
    at = int(depth)
    if name == "hit":
        return hit_at_k(ranked, relevant, at)
    if name == "recall":
        return recall_at_k(ranked, relevant, at)
    if name == "ndcg":
        return ndcg_at_k(ranked, relevant, at)
    raise ValueError("unknown metric: {}".format(metric))


def run_all(
    gold: Sequence[Dict], k: int
) -> Tuple[Dict[str, List[List[str]]], List[str]]:
    """Every row's ranked document list for every question, plus identity failures."""
    base = RAGEngine.from_corpus(os.path.join(ROOT, "corpus"))
    three = RAGEngine.from_corpus(os.path.join(ROOT, "corpus"), EngineConfig(ngram=4))
    candidates = base.config.candidates

    def via_engine(engine: RAGEngine, method: str) -> Callable[[str], List[str]]:
        return lambda q: engine.ranked_doc_ids(q, k=k, method=method)

    def via_module(retriever, lists, combination, normalisation):
        return lambda q: ranked_doc_ids(
            retriever, q, lists, combination, normalisation, candidates, k
        )

    producers: List[Tuple[str, Callable[[str], List[str]]]] = [
        ("hybrid2", via_engine(base, "hybrid")),
        ("hybrid3", via_engine(three, "hybrid")),
        ("ngram", via_engine(three, "ngram")),
        ("rrf2 (module)", via_module(base.retriever, TWO, "rrf", "minmax")),
        ("rrf3 (module)", via_module(three.retriever, THREE, "rrf", "minmax")),
    ]
    for label, lists, combination, normalisation in SCORE_ROWS:
        retriever = three.retriever if "ngram" in lists else base.retriever
        producers.append((label, via_module(retriever, lists, combination, normalisation)))

    rankings = {
        label: [produce(record["question"]) for record in gold]
        for label, produce in producers
    }
    failures = []
    for engine_row, module_row in (("hybrid2", "rrf2 (module)"), ("hybrid3", "rrf3 (module)")):
        for record, left, right in zip(gold, rankings[engine_row], rankings[module_row]):
            if left != right:
                failures.append("{}: {} vs {} on {}".format(
                    record["id"], engine_row, module_row, record["question"][:60]
                ))
    return rankings, failures


def per_query(
    rankings: Sequence[Sequence[str]], gold: Sequence[Dict], metric: str
) -> List[float]:
    return [score(metric, r, g["relevant"]) for r, g in zip(rankings, gold)]


def table(rankings: Dict[str, List[List[str]]], gold: Sequence[Dict], slice_: str) -> str:
    keep = [i for i, g in enumerate(gold) if slice_ == "all" or g["difficulty"] == slice_]
    sub_gold = [gold[i] for i in keep]
    lines = [
        "{} ({} questions)".format(slice_, len(keep)),
        "{:<18}".format("fusion") + "".join("{:>11}".format(m) for m in METRICS),
        "-" * (18 + 11 * len(METRICS)),
    ]
    for label, rows in rankings.items():
        if "(module)" in label:
            continue
        sub = [rows[i] for i in keep]
        cells = []
        for metric in METRICS:
            values = per_query(sub, sub_gold, metric)
            cells.append("{:>11.3f}".format(sum(values) / len(values)))
        lines.append("{:<18}".format(label) + "".join(cells))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    rankings, failures = run_all(gold, args.k)

    print("identity check: module RRF vs engine RRF on {} questions".format(len(gold)))
    if failures:
        print("FAILED on {} question(s):".format(len(failures)))
        for line in failures[:10]:
            print("  " + line)
        return 1
    print("  two lists:   identical on every question")
    print("  three lists: identical on every question\n")

    for slice_ in ("all", "easy", "hard"):
        print(table(rankings, gold, slice_))
        print()

    print("pre-specified tests, all slice, {} (Holm over {})".format(
        PRIMARY_METRIC, len(PRIMARY_TESTS)))
    primary = per_query(rankings[PRIMARY], gold, PRIMARY_METRIC)
    results = []
    for other in PRIMARY_TESTS:
        results.append((other, compare(
            primary,
            per_query(rankings[other], gold, PRIMARY_METRIC),
            permutations=args.permutations,
            seed=args.seed,
        )))
    adjusted = holm_bonferroni([result.p_value for _, result in results])
    for (other, result), (p_holm, reject) in zip(results, adjusted):
        print("  {} vs {:<8} {}  p_holm={:.3f}  {}".format(
            PRIMARY, other, result.describe(), p_holm,
            "SUPPORTED" if reject else "NOT SHOWN",
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
