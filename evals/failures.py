"""Explain the questions the retriever gets wrong, one failure at a time.

`evals/evaluate.py` ends by printing the ids it missed. That list is where the
interesting work starts and where the project has so far stopped: knowing that
`h01` fails says nothing about *why*, and the four levers this repo already
ships - expansion, the cross-encoder, the third n-gram list, MMR - are cheap to
turn on and expensive to reason about. Without a failure account, choosing
between them is guesswork dressed up as tuning.

So this script takes every (question, relevant document) pair the chosen method
misses at k and asks three questions of it.

**Where is the document instead?** Not "outside the top 5" but its actual rank
among all documents, alongside its rank in each component list. That single
comparison settles the most common suspicion about a fusion: whether RRF threw
away something BM25 or TF-IDF had already found. A pair where the fused rank
equals both component ranks is not a fusion failure at all, however tempting it
is to blame the newest moving part.

**How far away is it?** A document sitting at rank 6 and a document sitting
last are the same miss to hit@5 and completely different problems. The first is
an ordering problem - the retriever found it and mis-sorted it - and re-ranking
is exactly the tool for that. The second means the scorers barely matched it,
and no amount of re-ordering a candidate pool can rescue a document that never
entered the pool. The `distance` column separates the two, and the query-term
coverage column next to it - the share of the question's terms that appear
anywhere in the document - says whether there was anything to match on.

**Can anything already in the repo fix it?** This is the part that makes the
script actionable rather than merely descriptive. Every failing pair is re-run
under each lever the codebase offers, and the table reports the rank each one
achieves. A lever that pulls the document inside k has demonstrably solved that
failure; a lever that leaves it at rank 14 has demonstrably not, and no
aggregate table would have told them apart, because aggregates average a rescue
and a regression into a number near zero.

What is left over after that pass - failures no available lever recovers - is
the honest backlog. It is the only part of this output that argues for new work
rather than new defaults.

Usage:
    python evals/failures.py
    python evals/failures.py --method bm25 -k 3
    python evals/failures.py --no-recovery
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.crossencoder import RerankConfig  # noqa: E402
from ragkb.engine import EngineConfig, RAGEngine, read_corpus  # noqa: E402
from ragkb.expansion import ExpansionConfig  # noqa: E402
from ragkb.text import tokenize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Categories, in the order the classifier tries them. The first three are
#: about *what to do next*, not about severity, which is why "fusion-loss"
#: outranks distance: if the fusion dropped something a component had, the
#: distance is a symptom and the fusion is the bug.
FUSION_LOSS = "fusion-loss"
NEAR_MISS = "near-miss"
DEEP_MISS = "deep-miss"
UNRANKED = "unranked"

CATEGORIES = (FUSION_LOSS, NEAR_MISS, DEEP_MISS, UNRANKED)


@dataclass
class Failure:
    """One (question, relevant document) pair that the method missed at k."""

    question_id: str
    difficulty: str
    question: str
    doc_id: str
    #: Rank of ``doc_id`` in the method's document-level ranking, or ``None``
    #: when no chunk of it scored above zero and it was never ranked at all.
    rank: Optional[int]
    #: Rank of ``doc_id`` in each individual list ("bm25", "vector", ...).
    component_ranks: Dict[str, Optional[int]] = field(default_factory=dict)
    #: Share of the question's distinct terms that occur anywhere in the
    #: document. An upper bound on what any bag-of-words scorer could match.
    coverage: float = 0.0
    #: The documents that occupied the top k instead.
    displaced_by: List[str] = field(default_factory=list)
    category: str = DEEP_MISS
    #: lever name -> rank achieved under that lever (``None`` if unranked).
    recovery: Dict[str, Optional[int]] = field(default_factory=dict)

    def recovered_by(self, k: int) -> List[str]:
        """Levers that pull this document inside the top ``k``."""
        return [
            name
            for name, rank in sorted(self.recovery.items())
            if rank is not None and rank <= k
        ]


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


def query_coverage(question: str, document: str) -> float:
    """Share of the question's distinct terms that appear in ``document``.

    Deliberately crude: no IDF weighting, no scoring, just presence. It exists
    to answer one question a score cannot - "was there anything here to match
    at all?" - and weighting it would blur that into a second relevance
    judgement, which is what the retriever is already for.
    """
    terms = set(tokenize(question))
    if not terms:
        return 0.0
    present = set(tokenize(document))
    return sum(1 for term in terms if term in present) / len(terms)


def document_ranks(engine: RAGEngine, question: str, method: str, **kwargs) -> Dict[str, int]:
    """``doc_id -> 1-based rank`` over the whole corpus for one query.

    Ranking every document rather than the top k is the point: a failure
    analysis that could only see the top k would be unable to distinguish the
    two failure modes it exists to distinguish.
    """
    ordered = engine.ranked_doc_ids(
        question, k=len(engine.documents), method=method, **kwargs
    )
    return {doc_id: rank for rank, doc_id in enumerate(ordered, start=1)}


def classify(
    rank: Optional[int], component_ranks: Dict[str, Optional[int]], k: int
) -> str:
    """Label one failure by what would have to change to fix it."""
    if any(
        other is not None and other <= k for other in component_ranks.values()
    ) and (rank is None or rank > k):
        return FUSION_LOSS
    if rank is None:
        return UNRANKED
    if rank <= 2 * k:
        return NEAR_MISS
    return DEEP_MISS


def build_levers(
    documents: Dict[str, str], config: EngineConfig
) -> List[Tuple[str, RAGEngine, Dict]]:
    """The recovery configurations, as ``(name, engine, search kwargs)``.

    Two engines cover all four levers: expansion and re-ranking are query-time
    overrides on the shipped index, while both n-gram rows need a third index
    fitted, which is index-time work and cannot be a keyword argument.
    """
    base = RAGEngine(documents, config=config)
    ngram_config = EngineConfig(
        target_tokens=config.target_tokens,
        overlap_tokens=config.overlap_tokens,
        rrf_k=config.rrf_k,
        candidates=config.candidates,
        ngram=4,
    )
    with_ngram = RAGEngine(documents, config=ngram_config)
    return [
        ("expansion", base, {"expansion": ExpansionConfig()}),
        ("rerank", base, {"rerank": RerankConfig()}),
        ("mmr", base, {"mmr_lambda": 0.5}),
        ("hybrid3", with_ngram, {}),
        ("ngram-only", with_ngram, {"method": "ngram"}),
    ]


def analyse(
    engine: RAGEngine,
    gold: Sequence[Dict],
    k: int,
    method: str = "hybrid",
    levers: Optional[Sequence[Tuple[str, RAGEngine, Dict]]] = None,
) -> List[Failure]:
    """Every (question, relevant document) pair ``method`` misses at ``k``."""
    component_methods = [
        name for name in engine.retriever.methods if name != "hybrid"
    ]
    failures: List[Failure] = []
    for record in gold:
        question = record["question"]
        ranks = document_ranks(engine, question, method)
        missed = [
            doc_id
            for doc_id in record["relevant"]
            if ranks.get(doc_id) is None or ranks[doc_id] > k
        ]
        if not missed:
            continue
        # Only pay for the component and lever passes on questions that failed.
        components = {
            name: document_ranks(engine, question, name)
            for name in component_methods
        }
        top_k = [doc for doc, rank in sorted(ranks.items(), key=lambda p: p[1]) if rank <= k]
        for doc_id in missed:
            component_ranks = {
                name: table.get(doc_id) for name, table in components.items()
            }
            failure = Failure(
                question_id=record["id"],
                difficulty=record.get("difficulty", "easy"),
                question=question,
                doc_id=doc_id,
                rank=ranks.get(doc_id),
                component_ranks=component_ranks,
                coverage=query_coverage(question, engine.documents.get(doc_id, "")),
                displaced_by=top_k,
                category=classify(ranks.get(doc_id), component_ranks, k),
            )
            for name, lever_engine, kwargs in levers or ():
                lever_ranks = document_ranks(
                    lever_engine,
                    question,
                    kwargs.get("method", method),
                    **{key: value for key, value in kwargs.items() if key != "method"}
                )
                failure.recovery[name] = lever_ranks.get(doc_id)
            failures.append(failure)
    return failures


def format_failures(failures: Sequence[Failure], components: Sequence[str]) -> str:
    """One row per failing pair: where the document is, and how far that is."""
    header = "{:<6}{:<6}{:<16}{:>6}".format("id", "diff", "document", "rank")
    header += "".join("{:>9}".format(name) for name in components)
    header += "{:>10}{:>14}".format("coverage", "category")
    lines = [header, "-" * len(header)]
    for failure in failures:
        row = "{:<6}{:<6}{:<16}{:>6}".format(
            failure.question_id,
            failure.difficulty,
            failure.doc_id,
            "-" if failure.rank is None else failure.rank,
        )
        row += "".join(
            "{:>9}".format(
                "-" if failure.component_ranks.get(name) is None
                else failure.component_ranks[name]
            )
            for name in components
        )
        row += "{:>10.2f}{:>14}".format(failure.coverage, failure.category)
        lines.append(row)
    return "\n".join(lines)


def format_recovery(
    failures: Sequence[Failure], levers: Sequence[str], k: int
) -> str:
    """One row per failing pair: the rank each lever achieves, ``*`` if inside k."""
    header = "{:<6}{:<16}".format("id", "document")
    header += "".join("{:>12}".format(name) for name in levers)
    lines = [header, "-" * len(header)]
    for failure in failures:
        row = "{:<6}{:<16}".format(failure.question_id, failure.doc_id)
        for name in levers:
            rank = failure.recovery.get(name)
            if rank is None:
                cell = "-"
            else:
                cell = "{}{}".format(rank, "*" if rank <= k else "")
            row += "{:>12}".format(cell)
        lines.append(row)
    totals = "{:<6}{:<16}".format("", "recovered")
    for name in levers:
        recovered = sum(
            1
            for failure in failures
            if failure.recovery.get(name) is not None
            and failure.recovery[name] <= k
        )
        totals += "{:>12}".format("{}/{}".format(recovered, len(failures)))
    lines.append("-" * len(header))
    lines.append(totals)
    return "\n".join(lines)


def format_categories(failures: Sequence[Failure]) -> str:
    """How the failures distribute over categories, per difficulty slice."""
    slices = ["all"] + sorted({failure.difficulty for failure in failures})
    header = "{:<8}".format("slice") + "".join(
        "{:>14}".format(name) for name in CATEGORIES
    )
    lines = [header, "-" * len(header)]
    for label in slices:
        subset = [
            failure
            for failure in failures
            if label == "all" or failure.difficulty == label
        ]
        row = "{:<8}".format("{} ({})".format(label, len(subset)))
        row += "".join(
            "{:>14}".format(
                sum(1 for failure in subset if failure.category == category)
            )
            for category in CATEGORIES
        )
        lines.append(row)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--method",
        default="hybrid",
        help="the retrieval method whose failures are being explained",
    )
    parser.add_argument(
        "--slice",
        default="all",
        choices=("all", "easy", "hard"),
        help="which slice of the gold set to analyse",
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help=(
            "skip the recovery pass. It fits a second index and re-runs every "
            "failing pair five more times, which is the expensive half"
        ),
    )
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    if args.slice != "all":
        gold = [r for r in gold if r.get("difficulty") == args.slice]
        if not gold:
            raise SystemExit("no {} questions in the gold set".format(args.slice))

    documents = read_corpus(args.corpus)
    config = EngineConfig()
    engine = RAGEngine(documents, config=config)
    levers = [] if args.no_recovery else build_levers(documents, config)

    pairs = sum(len(record["relevant"]) for record in gold)
    failures = analyse(engine, gold, args.k, method=args.method, levers=levers)
    print(
        "corpus: {} documents   gold: {} questions, {} relevant pairs "
        "({} slice)   method={}   k={}".format(
            len(documents), len(gold), pairs, args.slice, args.method, args.k
        )
    )
    print(
        "{} of {} relevant pairs missed at k={} ({:.1%})".format(
            len(failures), pairs, args.k, len(failures) / pairs if pairs else 0.0
        )
    )
    if not failures:
        print("\nnothing to explain.")
        return 0

    components = [name for name in engine.retriever.methods if name != "hybrid"]
    print("\nwhere the missed document actually ranks (of {} documents)".format(
        len(documents)
    ))
    print(format_failures(failures, components))

    print("\nfailures by category")
    print(format_categories(failures))

    if levers:
        names = [name for name, _, _ in levers]
        print("\nrank under each available lever (* = inside k={})".format(args.k))
        print(format_recovery(failures, names, args.k))

        unrecovered = [
            failure for failure in failures if not failure.recovered_by(args.k)
        ]
        print(
            "\n{} of {} failures are recovered by at least one existing lever".format(
                len(failures) - len(unrecovered), len(failures)
            )
        )
        if unrecovered:
            print("nothing in the repo recovers these:")
            for failure in unrecovered:
                print(
                    "  {} -> {} (rank {}, coverage {:.2f}): {}".format(
                        failure.question_id,
                        failure.doc_id,
                        "-" if failure.rank is None else failure.rank,
                        failure.coverage,
                        failure.question,
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
