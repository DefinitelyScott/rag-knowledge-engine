"""Leave-one-document-out jackknife: is a method gap broad, or a few documents?

Every significance test in this repository resamples *questions*. That answers
"would a different draw of questions have given the same delta?". It cannot
answer a different, equally fair question that the last three corpus increments
kept raising: "would a different corpus have given the same delta?". Adding two
documents moved ``hybrid3 vs hybrid2`` by +0.019 nDCG@5 on a question set that
was not edited, and adding two others had moved it by -0.009. A delta that
changes sign when forty documents become thirty-eight is sitting on its noise
floor, but the question-resampling tests cannot say whether that floor is a
broad, shallow effect or a handful of documents pulling in opposite directions.

This script prices that. For each document ``d`` in the corpus:

* ``Q_d`` is the set of gold questions whose answer key does **not** name ``d``.
  A question that names ``d`` has part of its answer key missing once ``d`` is
  gone, so it would score as a miss for reasons unrelated to retrieval; it is
  dropped from this document's comparison rather than scored against a hole.
* ``A_d`` is the mean per-question score difference (left minus right) on
  ``Q_d`` with the **full** corpus.
* ``B_d`` is the same quantity with ``d`` **removed from the corpus**.
* ``influence_d = A_d - B_d`` is what the presence of ``d`` as a competitor did
  to the gap, on questions that were identical in both arms. Positive means
  ``d`` helped the left method relative to the right.

Because the question set is held fixed within each ``d``, the influence is a
pure corpus effect. It is not a pure *document* effect: removing a document
also changes every term's IDF and the BM25 length normaliser for the rest of
the corpus, so a small non-zero influence on a document no question touches is
expected and is itself informative (it is the cost of the statistics moving).

The summary reports the full-corpus delta, the distribution of influences, the
largest movers, how many removals would flip the sign of the delta, and a
jackknife standard error over documents. The pseudo leave-one-out value is
defined as ``delta_full - influence_d`` so that every term sits on the same
question set; that is an approximation of the textbook jackknife (which holds
the unit of resampling, here the document, and nothing else), and the output
says so rather than presenting it as exact.

This measures; it does not judge. Influences clustered tightly around zero say
the delta is broad. A few large ones say it is a few documents' vocabulary.
Both are findings.

Usage:
    python evals/jackknife.py
    python evals/jackknife.py --pair hybrid2,bm25 --metric hit@1
    python evals/jackknife.py --top 8
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import read_corpus  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The same sibling harnesses ``corpus_lever.py`` reuses, for the same reason: a
# second script that resolves "hybrid3" independently is a second script that
# will eventually disagree about what it means.
significance_eval = _load("significance_eval", "significance.py")
lever = _load("corpus_lever_eval", "corpus_lever.py")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = mean(values)
    return math.sqrt(sum((v - centre) ** 2 for v in values) / (len(values) - 1))


def gap_scores(engines, routing, gold, left, right, metric, k) -> List[float]:
    """Per-question ``left - right`` score difference, in question order."""
    ranked = significance_eval.rankings_by_method(engines, routing, gold, k)
    return [
        significance_eval.metric_value(metric, ranked[left][i], record["relevant"])
        - significance_eval.metric_value(metric, ranked[right][i], record["relevant"])
        for i, record in enumerate(gold)
    ]


def questions_without(gold: Sequence[Dict], doc_id: str) -> List[int]:
    """Indices of the questions whose answer key does not name ``doc_id``."""
    return [i for i, record in enumerate(gold) if doc_id not in record["relevant"]]


def influence(
    full_gap: Sequence[float], without_gap: Sequence[float]
) -> Tuple[float, int]:
    """Influence of one document and how many questions' gap it changed.

    Both vectors cover the *same* questions in the same order. Returns
    ``(mean difference, changed count)``.
    """
    if len(full_gap) != len(without_gap):
        raise ValueError("gap vectors must cover the same questions")
    if not full_gap:
        raise ValueError("no questions to compare")
    changed = sum(1 for a, b in zip(full_gap, without_gap) if abs(a - b) > 1e-12)
    return mean(full_gap) - mean(without_gap), changed


def jackknife_se(delta_full: float, influences: Sequence[float]) -> float:
    """Jackknife standard error over documents.

    The leave-one-out value for document ``i`` is ``delta_full - influence_i``,
    which puts every term on the full question set. The textbook estimator is
    ``sqrt((n - 1) / n * sum((theta_i - mean(theta)) ** 2))``.
    """
    n = len(influences)
    if n < 2:
        raise ValueError("a jackknife needs at least two documents")
    leave_out = [delta_full - value for value in influences]
    centre = mean(leave_out)
    return math.sqrt((n - 1) / n * sum((v - centre) ** 2 for v in leave_out))


def sign_flips(delta_full: float, influences: Sequence[float]) -> List[int]:
    """Indices of documents whose removal would flip the sign of the delta."""
    if delta_full == 0:
        return []
    return [
        i
        for i, value in enumerate(influences)
        if (delta_full - value) * delta_full < 0
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("--pair", default="hybrid3,hybrid2")
    parser.add_argument("--metric", default="ndcg@5")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--top", type=int, default=6, help="movers to list per sign")
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--ngram-n", type=int, default=4)
    args = parser.parse_args(argv)

    left, _, right = args.pair.partition(",")
    left, right = left.strip(), right.strip()
    known = {entry[0] for entry in significance_eval.METHOD_ROUTING}
    unknown = [name for name in (left, right) if name not in known]
    if unknown:
        raise SystemExit("unknown methods: {}".format(",".join(unknown)))
    routing = [e for e in significance_eval.METHOD_ROUTING if e[0] in (left, right)]

    documents = read_corpus(args.corpus)
    gold = significance_eval.load_gold(args.gold)
    config = dict(
        target_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        rrf_k=args.rrf_k,
    )

    full_engines = lever.build_engines(documents, routing, ngram_n=args.ngram_n, **config)
    full_gap = gap_scores(full_engines, routing, gold, left, right, args.metric, args.k)
    delta_full = mean(full_gap)

    rows = []  # (doc_id, n_questions, influence, changed)
    for doc_id in sorted(documents):
        keep = questions_without(gold, doc_id)
        if not keep:
            continue
        subset = [gold[i] for i in keep]
        reduced = {d: t for d, t in documents.items() if d != doc_id}
        engines = lever.build_engines(reduced, routing, ngram_n=args.ngram_n, **config)
        without_gap = gap_scores(engines, routing, subset, left, right, args.metric, args.k)
        value, changed = influence([full_gap[i] for i in keep], without_gap)
        rows.append((doc_id, len(keep), value, changed))

    values = [row[2] for row in rows]
    se = jackknife_se(delta_full, values)
    flips = sign_flips(delta_full, values)
    helped = [r for r in rows if r[2] > 1e-12]
    hurt = [r for r in rows if r[2] < -1e-12]

    print(
        "{} vs {} on {}, k={}: {} documents, {} questions".format(
            left, right, args.metric, args.k, len(documents), len(gold)
        )
    )
    print("full-corpus delta: {:+.4f}".format(delta_full))
    print(
        "influence (gap with d present minus gap with d removed, questions fixed): "
        "mean {:+.4f}, sd {:.4f}, range [{:+.4f}, {:+.4f}]".format(
            mean(values), sample_sd(values), min(values), max(values)
        )
    )
    print(
        "{} documents helped the left method, {} hurt it, {} left every score "
        "exactly unchanged".format(len(helped), len(hurt), len(rows) - len(helped) - len(hurt))
    )
    print(
        "document-jackknife SE of the delta: {:.4f}  (delta / SE = {:.2f})".format(
            se, delta_full / se if se else float("inf")
        )
    )
    print(
        "removals that flip the sign of the delta: {}{}".format(
            len(flips), "  (" + ", ".join(rows[i][0] for i in flips) + ")" if flips else ""
        )
    )
    ordered = sorted(rows, key=lambda r: r[2])
    print("\nlargest negative influence (document hurt the left method):")
    for doc_id, n, value, changed in ordered[: args.top]:
        print("  {:<28} {:+.4f}  ({} of {} questions changed)".format(doc_id, value, changed, n))
    print("largest positive influence (document helped the left method):")
    for doc_id, n, value, changed in reversed(ordered[-args.top :]):
        print("  {:<28} {:+.4f}  ({} of {} questions changed)".format(doc_id, value, changed, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
