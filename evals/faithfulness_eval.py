"""Score answer faithfulness and citation accuracy over evals/gold.jsonl.

``evaluate.py`` measures whether the right passages come back. This script
measures what happens after that: whether the answer stays inside those
passages and whether its citations point at the documents that produced it.

Two things are reported per k:

* the four faithfulness measures from :mod:`ragkb.faithfulness`, averaged over
  the gold questions;
* ``cited_gold``, the share of answers whose citation list contains a document
  the gold set marks relevant. This is the only column that uses the labels,
  and it separates "faithful to what was retrieved" from "faithful to what was
  retrieved, which was also right".

Usage:
    python evals/faithfulness_eval.py [--corpus corpus] [--ks 3,4,5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.answerer import ExtractiveAnswerer  # noqa: E402
from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.faithfulness import evaluate_answer, mean_report  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

COLUMNS = (
    "sentence_grounding",
    "token_grounding",
    "citation_precision",
    "citation_recall",
    "cited_gold",
)


def load_gold(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError("gold set is empty: {}".format(path))
    return records


def score_at_k(
    engine: RAGEngine,
    answerer: ExtractiveAnswerer,
    gold: Sequence[Dict],
    k: int,
) -> Dict[str, float]:
    reports = []
    cited_gold = []
    worst = []
    for record in gold:
        results = engine.search(record["question"], k=k)
        answer = answerer.answer(record["question"], results)
        report = evaluate_answer(answer, results)
        reports.append(report)
        cited_gold.append(
            1.0 if set(answer.citations) & set(record["relevant"]) else 0.0
        )
        worst.append((report.token_grounding, record["id"]))
    summary = mean_report(reports)
    summary["cited_gold"] = sum(cited_gold) / len(cited_gold)
    summary["_worst"] = sorted(worst)[:3]
    return summary


def format_table(rows: Sequence[tuple]) -> str:
    header = "{:<6}".format("k") + "".join(
        "{:>20}".format(name) for name in COLUMNS
    )
    lines = [header, "-" * len(header)]
    for k, summary in rows:
        lines.append(
            "{:<6}".format(k)
            + "".join("{:>20.3f}".format(summary[name]) for name in COLUMNS)
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("--ks", default="3,4,5", help="comma-separated k values")
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--mmr-lambda", type=float, default=None)
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    engine = RAGEngine.from_corpus(
        args.corpus, config=EngineConfig(mmr_lambda=args.mmr_lambda)
    )
    answerer = ExtractiveAnswerer(max_sentences=args.max_sentences)

    print(
        "questions: {}   answerer: extractive (max {} sentences)   mmr: {}\n".format(
            len(gold),
            args.max_sentences,
            "off" if args.mmr_lambda is None else args.mmr_lambda,
        )
    )

    ks = [int(value) for value in args.ks.split(",") if value.strip()]
    rows = [(k, score_at_k(engine, answerer, gold, k)) for k in ks]
    print(format_table(rows))

    last_k, last = rows[-1]
    print(
        "\nweakest token grounding at k={}: {}".format(
            last_k,
            ", ".join(
                "{} ({:.2f})".format(qid, score) for score, qid in last["_worst"]
            ),
        )
    )
    print(
        "\nsentence_grounding is 1.000 by construction for the extractive "
        "answerer;\nit is a regression guard, not a quality score. The measure "
        "to watch for an\nLLM answerer is token_grounding."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
