"""Sweep the index-time parameters that every other evaluation takes as given.

`evals/evaluate.py` sweeps the query-time knobs - expansion alpha, re-ranking
alpha, MMR lambda - against one fixed index. But that index is itself built from
three numbers that have been constants since the first commit: `target_tokens`
(how long a chunk is), `overlap_tokens` (how much of the previous chunk it
repeats) and `rrf_k` (the constant in `1 / (rrf_k + rank)`). Nothing in the
project has ever checked whether they are good, which makes them the weakest
claim in the README.

The two kinds of knob need different machinery, and separating them is the
point of this script:

- **Chunking is index-time.** Changing `target_tokens` or `overlap_tokens`
  changes what a document is cut into, so both indexes have to be re-fitted from
  scratch. Every row here is a full rebuild, and the table reports the resulting
  corpus shape (chunk count, mean chunk length) next to the accuracy, because
  the two move together and reading either alone is misleading.
- **`rrf_k` is query-time.** It only rescales an existing pair of ranked lists,
  so one fitted engine serves the whole sweep. That table carries an extra
  `changed` column - how many of the gold questions' top-k orderings actually
  differ from the default - because an accuracy table alone cannot distinguish
  "this setting is equally good" from "this setting does nothing at all".

Every row is reported as a delta against the current defaults, so the question
the script exists to answer - *are the shipped constants defensible?* - is
answered in the table rather than in prose.

Usage:
    python evals/tune.py
    python evals/tune.py --chunk-specs 100:25,120:30,150:40 --rrf-ks 1,10,60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine  # noqa: E402
from ragkb.metrics import aggregate  # noqa: E402
from ragkb.text import parse_chunk_specs  # noqa: E402,F401  (re-exported)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: The constants the engine ships with, and therefore the baseline every row is
#: measured against.
DEFAULT_CHUNK_TOKENS = EngineConfig.target_tokens
DEFAULT_OVERLAP_TOKENS = EngineConfig.overlap_tokens
DEFAULT_RRF_K = EngineConfig.rrf_k

#: Columns kept in the sweep tables. hit@1 and MRR read early precision, hit@5
#: and recall@5 read reach, nDCG@5 reads how the whole relevant set is ordered.
COLUMNS = ("hit@1", "hit@3", "hit@5", "recall@5", "ndcg@5", "mrr")


def load_gold(path: str) -> List[Dict]:
    """Read the gold set, defaulting unlabelled records to the easy slice."""
    records: List[Dict] = []
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


def score(
    engine: RAGEngine, gold: Sequence[Dict], k: int
) -> Dict[str, float]:
    """Accuracy for one engine over one slice of the gold set."""
    rankings = [
        engine.ranked_doc_ids(record["question"], k=k) for record in gold
    ]
    report = aggregate(
        rankings, [record["relevant"] for record in gold], ks=(1, 3, k)
    )
    return {name.format(k): report[name.format(k)] for name in COLUMNS}


def top_k_orderings(
    engine: RAGEngine, gold: Sequence[Dict], k: int
) -> List[Tuple[int, ...]]:
    """The chunk indices each question retrieves, in rank order.

    Two settings that produce identical lists here are the same retriever with
    different arithmetic, however their accuracy tables happen to read.
    """
    return [
        tuple(result.chunk_index for result in engine.search(record["question"], k=k))
        for record in gold
    ]


def sweep_chunking(
    documents: Dict[str, str],
    gold: Sequence[Dict],
    specs: Sequence[Tuple[int, int]],
    k: int,
) -> List[Tuple[str, Dict[str, float]]]:
    """Rebuild the index once per chunking setting and score each one.

    ``chunks`` and ``mean_tok`` are carried alongside the accuracy columns
    because they are the mechanism: a smaller target makes more, shorter chunks,
    which raises the chance that *some* chunk of the right document lands in the
    top k while lowering the chance that any one of them carries the whole
    answer.
    """
    rows: List[Tuple[str, Dict[str, float]]] = []
    for target, overlap in specs:
        engine = RAGEngine(
            documents,
            config=EngineConfig(target_tokens=target, overlap_tokens=overlap),
        )
        stats = engine.stats()
        report = {
            "chunks": stats["chunks"],
            "mean_tok": stats["mean_chunk_tokens"],
        }
        report.update(score(engine, gold, k))
        label = "{}:{}".format(target, overlap)
        if (target, overlap) == (DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS):
            label += "*"
        rows.append((label, report))
    return rows


def sweep_rrf_k(
    engine: RAGEngine,
    gold: Sequence[Dict],
    values: Sequence[int],
    k: int,
) -> List[Tuple[str, Dict[str, float]]]:
    """Score each ``rrf_k`` on one already-fitted engine.

    ``rrf_k`` never touches the indexes, only the fusion arithmetic, so the
    engine is mutated in place and restored afterwards. The ``changed`` column
    counts questions whose top-k ordering differs from the default - the
    measurement that separates "no better" from "no different".
    """
    original = engine.retriever.rrf_k
    try:
        engine.retriever.rrf_k = DEFAULT_RRF_K
        baseline = top_k_orderings(engine, gold, k)
        rows: List[Tuple[str, Dict[str, float]]] = []
        for value in values:
            engine.retriever.rrf_k = value
            report = score(engine, gold, k)
            report["changed"] = float(
                sum(
                    1
                    for before, after in zip(
                        baseline, top_k_orderings(engine, gold, k)
                    )
                    if before != after
                )
            )
            label = str(value) + ("*" if value == DEFAULT_RRF_K else "")
            rows.append((label, report))
        return rows
    finally:
        engine.retriever.rrf_k = original


#: Columns describing the shape of the run rather than its quality. They are
#: printed plain: a delta on "how many chunks" is not a result, it is the cause.
SHAPE_COLUMNS = ("chunks", "mean_tok", "changed")
INTEGER_COLUMNS = ("chunks", "changed")


def format_table(
    rows: Sequence[Tuple[str, Dict[str, float]]],
    label: str,
    baseline: Optional[Dict[str, float]] = None,
) -> str:
    """Render a sweep, annotating every accuracy cell with its delta.

    The delta is the whole point of the table, so it sits in the cell rather
    than in a second table the reader has to align by eye. Shape columns are
    left plain so the eye can separate what was changed from what it cost.
    """
    columns = list(rows[0][1].keys())
    width = 17 if baseline else 12
    header = "{:<10}".format(label) + "".join(
        "{:>{w}}".format(name, w=10 if name in SHAPE_COLUMNS else width)
        for name in columns
    )
    lines = [header, "-" * len(header)]
    for row_label, report in rows:
        cells = []
        for name in columns:
            value = report[name]
            if name in INTEGER_COLUMNS:
                text = "{:d}".format(int(round(value)))
            elif name in SHAPE_COLUMNS:
                text = "{:.1f}".format(value)
            else:
                text = "{:.3f}".format(value)
                if baseline:
                    delta = value - baseline[name]
                    text += (
                        " {:+.3f}".format(delta) if abs(delta) >= 5e-4 else "  0    "
                    )
            cells.append(
                "{:>{w}}".format(text, w=10 if name in SHAPE_COLUMNS else width)
            )
        lines.append("{:<10}".format(row_label) + "".join(cells))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--chunk-specs",
        default="60:15,80:20,100:25,120:0,120:15,120:30,120:60,150:40,180:45,200:50",
        help="comma-separated target:overlap chunking settings to sweep",
    )
    parser.add_argument(
        "--rrf-ks",
        default="1,5,10,20,60,200,1000",
        help="comma-separated rrf_k values to sweep",
    )
    parser.add_argument(
        "--slice",
        default="all",
        choices=("all", "easy", "hard"),
        help="which slice of the gold set to sweep against",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=None,
        metavar="N",
        help=(
            "fuse a character N-gram list as well, and sweep rrf_k against "
            "three lists instead of two. The two-list sweep found rrf_k inert "
            "because BM25 and TF-IDF index the same words and rarely disagree; "
            "this is the flag that tests whether that was the reason"
        ),
    )
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    if args.slice != "all":
        gold = [r for r in gold if r.get("difficulty") == args.slice]
        if not gold:
            raise SystemExit("no {} questions in the gold set".format(args.slice))

    from ragkb.engine import read_corpus

    documents = read_corpus(args.corpus)
    print(
        "corpus: {} documents   gold: {} questions ({} slice)   k={}".format(
            len(documents), len(gold), args.slice, args.k
        )
    )
    print(
        "defaults marked *: target_tokens={}, overlap_tokens={}, rrf_k={}".format(
            DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS, DEFAULT_RRF_K
        )
    )

    specs = parse_chunk_specs(args.chunk_specs)
    if specs:
        rows = sweep_chunking(documents, gold, specs, args.k)
        default_label = "{}:{}*".format(
            DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS
        )
        baseline = dict(
            next(
                (report for label, report in rows if label == default_label),
                rows[0][1],
            )
        )
        print("\nchunking (index rebuilt per row, hybrid, k={})".format(args.k))
        print(format_table(rows, "tgt:ovl", baseline=baseline))

    values = [int(v) for v in args.rrf_ks.split(",") if v.strip()]
    if values:
        engine = RAGEngine(documents, config=EngineConfig(ngram=args.ngram))
        rows = sweep_rrf_k(engine, gold, values, args.k)
        baseline = dict(
            next(
                (
                    report
                    for label, report in rows
                    if label == "{}*".format(DEFAULT_RRF_K)
                ),
                rows[0][1],
            )
        )
        print(
            "\nrrf_k (one fitted index, hybrid over {} lists, k={})".format(
                len(engine.retriever._lists()), args.k
            )
        )
        print(format_table(rows, "rrf_k", baseline=baseline))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
