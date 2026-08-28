"""Measure the abstention gate against answerable and unanswerable questions.

Every other eval in this repo asks how well retrieval finds the answer. This
one asks the question underneath it: does the system know when there *is* no
answer? A retriever with a perfect hit@5 still returns five passages for a
question about Kubernetes, and the extractive answerer will happily quote one
of them with a citation attached.

Two question sets are needed, and only one of them already existed:

* ``gold.jsonl`` - answerable. Every abstention here is a false alarm.
* ``unanswerable.jsonl`` - questions the corpus genuinely does not cover. Every
  answer here is the failure the gate exists to prevent.

The unanswerable set is split into ``off-domain`` (Kubernetes, Raft, React -
different subject entirely) and ``near-domain`` (SPLADE, dense-passage-retrieval
training, sharding - information retrieval questions this corpus happens not to
answer). That split is the point of the file. Off-domain questions are made of
words the corpus never indexed and any threshold catches them; near-domain
questions are built almost entirely from vocabulary the corpus knows, so they
are the only rows that say anything about whether the gate works. A headline
number over the pooled set would be flattered by the easy half, so the two are
always reported separately.

The sweep reports, at each threshold:

``answer_rate``  fraction of answerable questions still answered (higher better)
``abstain_off``  fraction of off-domain questions declined (higher better)
``abstain_near`` fraction of near-domain questions declined (higher better)
``accuracy``     over the pooled set, weighting both error types equally
``f1``           F1 of the "answer an answerable question" decision

One more column matters more than any of them. Some answerable questions the
retriever simply misses - it returns five chunks, none from a relevant
document - and on those the ideal gate abstains, because answering from
evidence that does not contain the answer is the same failure whether or not
the corpus could have answered it. ``caught_misses`` reports the share of the
retriever's own misses that the gate declines, and ``lost_hits`` the share of
successful retrievals it throws away. A gate that only ever traded one for the
other at the same rate would be worth nothing.

Usage:
    python evals/abstention_eval.py [--corpus corpus] [--k 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.abstain import AbstentionConfig, assess  # noqa: E402
from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.metrics import mean  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError("question set is empty: {}".format(path))
    return records


def score_questions(
    engine: RAGEngine, questions: Sequence[str], k: int
) -> List[float]:
    """The answerability score of each question, computed once and reused.

    The threshold sweep only compares these numbers against a line, so
    retrieval runs once per question rather than once per question per
    threshold.
    """
    config = AbstentionConfig()
    scores = []
    for question in questions:
        results = engine.search(question, k=k)
        scores.append(assess(engine.retriever, question, results, config=config).score)
    return scores


def retrieval_hit(engine: RAGEngine, record: Dict, k: int) -> bool:
    """Did retrieval put any relevant document in the top ``k``?"""
    ranked = set(engine.ranked_doc_ids(record["question"], k=k))
    return bool(ranked & set(record["relevant"]))


def rate(values: Sequence[bool]) -> float:
    return mean([1.0 if value else 0.0 for value in values]) if values else float("nan")


def sweep_row(
    threshold: float,
    answerable_scores: Sequence[float],
    hits: Sequence[bool],
    by_kind: Dict[str, List[float]],
) -> Dict[str, float]:
    answered = [score >= threshold for score in answerable_scores]
    off = by_kind.get("off-domain", [])
    near = by_kind.get("near-domain", [])

    true_positive = sum(1 for value in answered if value)
    false_negative = len(answered) - true_positive
    false_positive = sum(
        1 for score in off + near if score >= threshold
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / len(answered) if answered else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    total = len(answered) + len(off) + len(near)
    correct = true_positive + (len(off) + len(near) - false_positive)
    # Pooled accuracy is not the number to tune on: there are three times as
    # many answerable questions as unanswerable ones, so it is maximised by a
    # gate that barely fires. Balanced accuracy weights the two classes
    # equally and is what the recommendation below is chosen on.
    abstain_rate = (
        (len(off) + len(near) - false_positive) / (len(off) + len(near))
        if off or near
        else 0.0
    )

    caught = [
        score < threshold
        for score, hit in zip(answerable_scores, hits)
        if not hit
    ]
    lost = [
        score < threshold for score, hit in zip(answerable_scores, hits) if hit
    ]
    return {
        "answer_rate": recall,
        "abstain_off": rate([score < threshold for score in off]),
        "abstain_near": rate([score < threshold for score in near]),
        "accuracy": correct / total if total else 0.0,
        "balanced": (recall + abstain_rate) / 2.0,
        "f1": f1,
        "caught_misses": rate(caught),
        "lost_hits": rate(lost),
        "_false_negative": float(false_negative),
    }


def format_table(rows: Sequence[tuple], columns: Sequence[str]) -> str:
    header = "{:<10}".format("threshold") + "".join(
        "{:>14}".format(name) for name in columns
    )
    lines = [header, "-" * len(header)]
    for label, row in rows:
        lines.append(
            "{:<10}".format(label)
            + "".join(
                "{:>14}".format(
                    "  n/a" if row[name] != row[name] else "{:.3f}".format(row[name])
                )
                for name in columns
            )
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument(
        "--unanswerable", default=os.path.join(HERE, "unanswerable.jsonl")
    )
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        default="0.00,0.20,0.24,0.26,0.28,0.30,0.32,0.36,0.40,0.50",
        help="comma-separated thresholds to sweep",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="how many borderline questions to print at the default threshold",
    )
    args = parser.parse_args(argv)

    gold = load_jsonl(args.gold)
    unanswerable = load_jsonl(args.unanswerable)
    engine = RAGEngine.from_corpus(args.corpus, config=EngineConfig())

    print(
        "{} answerable questions, {} unanswerable ({} off-domain, "
        "{} near-domain)   k={}".format(
            len(gold),
            len(unanswerable),
            sum(1 for r in unanswerable if r.get("kind") == "off-domain"),
            sum(1 for r in unanswerable if r.get("kind") == "near-domain"),
            args.k,
        )
    )

    answerable_scores = score_questions(
        engine, [record["question"] for record in gold], args.k
    )
    hits = [retrieval_hit(engine, record, args.k) for record in gold]
    unanswerable_scores = score_questions(
        engine, [record["question"] for record in unanswerable], args.k
    )
    by_kind: Dict[str, List[float]] = {}
    for record, score in zip(unanswerable, unanswerable_scores):
        by_kind.setdefault(record.get("kind", "off-domain"), []).append(score)

    print(
        "mean score: answerable {:.3f}, off-domain {:.3f}, near-domain "
        "{:.3f}".format(
            mean(answerable_scores),
            mean(by_kind.get("off-domain", [])),
            mean(by_kind.get("near-domain", [])),
        )
    )
    print(
        "retrieval already misses {} of the {} answerable questions at k={}".format(
            sum(1 for hit in hits if not hit), len(hits), args.k
        )
    )

    thresholds = [
        float(value) for value in args.thresholds.split(",") if value.strip()
    ]
    rows = [
        ("{:g}".format(threshold), sweep_row(threshold, answerable_scores, hits, by_kind))
        for threshold in thresholds
    ]
    columns = [
        "answer_rate",
        "abstain_off",
        "abstain_near",
        "accuracy",
        "balanced",
        "f1",
        "caught_misses",
        "lost_hits",
    ]
    print("\nabstention threshold sweep")
    print(format_table(rows, columns))

    best_label, best_row = max(rows, key=lambda pair: pair[1]["balanced"])
    print(
        "\nbest balanced accuracy at threshold {}: {:.3f} "
        "(answers {:.0%} of answerable, declines {:.0%} of off-domain and "
        "{:.0%} of near-domain)".format(
            best_label,
            best_row["balanced"],
            best_row["answer_rate"],
            best_row["abstain_off"],
            best_row["abstain_near"],
        )
    )
    print(
        "at that point the gate declines {:.0%} of the questions retrieval "
        "already misses and {:.0%} of the ones it gets right".format(
            best_row["caught_misses"], best_row["lost_hits"]
        )
    )

    default = AbstentionConfig().threshold
    refused = [
        (record["id"], score)
        for record, score in zip(gold, answerable_scores)
        if score < default
    ]
    admitted = [
        (record["id"], record.get("kind", ""), score)
        for record, score in zip(unanswerable, unanswerable_scores)
        if score >= default
    ]
    print(
        "\nat the shipped threshold {:g}, wrongly refused: {}".format(
            default,
            ", ".join(
                "{} ({:.2f})".format(qid, score)
                for qid, score in sorted(refused, key=lambda pair: pair[1])[
                    : args.examples
                ]
            )
            or "none",
        )
    )
    print(
        "wrongly answered: {}".format(
            ", ".join(
                "{} {} ({:.2f})".format(qid, kind, score)
                for qid, kind, score in sorted(
                    admitted, key=lambda triple: -triple[2]
                )[: args.examples]
            )
            or "none"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
