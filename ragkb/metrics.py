"""Retrieval evaluation metrics.

All metrics take a ranked list of predicted identifiers and the set of
identifiers judged relevant for that query, so they work equally well at
document level or chunk level.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Set


def _relevant_set(relevant: Iterable[str]) -> Set[str]:
    return set(relevant)


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant item appears in the top ``k``."""
    gold = _relevant_set(relevant)
    return 1.0 if any(item in gold for item in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top ``k``."""
    gold = _relevant_set(relevant)
    if not gold:
        return 0.0
    found = {item for item in ranked[:k] if item in gold}
    return len(found) / len(gold)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    gold = _relevant_set(relevant)
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in gold) / len(top)


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item, or 0.0 if none is retrieved."""
    gold = _relevant_set(relevant)
    for position, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / position
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(
    rankings: Sequence[Sequence[str]],
    relevancies: Sequence[Iterable[str]],
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """Compute mean hit@k, recall@k, precision@k and MRR over a query set."""
    if len(rankings) != len(relevancies):
        raise ValueError("rankings and relevancies must be the same length")
    report: Dict[str, float] = {}
    for k in ks:
        report["hit@{}".format(k)] = mean(
            [hit_at_k(r, g, k) for r, g in zip(rankings, relevancies)]
        )
        report["recall@{}".format(k)] = mean(
            [recall_at_k(r, g, k) for r, g in zip(rankings, relevancies)]
        )
        report["precision@{}".format(k)] = mean(
            [precision_at_k(r, g, k) for r, g in zip(rankings, relevancies)]
        )
    report["mrr"] = mean(
        [reciprocal_rank(r, g) for r, g in zip(rankings, relevancies)]
    )
    return report
