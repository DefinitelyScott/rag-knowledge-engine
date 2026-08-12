"""Retrieval evaluation metrics.

All metrics take a ranked list of predicted identifiers and the set of
identifiers judged relevant for that query, so they work equally well at
document level or chunk level.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set


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


def ndcg_at_k(
    ranked: Sequence[str],
    relevant: Iterable[str],
    k: int,
    gains: Optional[Mapping[str, float]] = None,
) -> float:
    """Normalized discounted cumulative gain at ``k``.

    hit@k and MRR only see the first relevant result, and recall@k does not
    care whether the second relevant document sits at rank 2 or rank ``k``.
    nDCG fills that gap: each relevant item earns its gain discounted by
    ``log2(rank + 1)``, and the total is normalised by the best score any
    ranking of the judged items could reach, so 1.0 means every relevant item
    is placed as high as possible.

    Relevance is binary by default (gain 1.0 for every item in ``relevant``);
    pass ``gains`` to override or extend with graded judgments. An item is
    credited only the first time it appears: the overlapping chunker means one
    document often fills several of the top ``k`` slots, and re-retrieving
    already-credited evidence should not score like ranking a second relevant
    document. The ideal ranking is likewise one slot per distinct item.
    """
    if k <= 0:
        return 0.0
    gold = _relevant_set(relevant)
    graded = dict(gains) if gains else {}

    def gain(item: str) -> float:
        if item in graded:
            return float(graded[item])
        return 1.0 if item in gold else 0.0

    seen: Set[str] = set()
    dcg = 0.0
    for position, item in enumerate(ranked[:k], start=1):
        if item in seen:
            continue
        seen.add(item)
        value = gain(item)
        if value > 0.0:
            dcg += value / math.log2(position + 1)

    universe = gold | set(graded)
    ideal = sorted((gain(item) for item in universe), reverse=True)[:k]
    idcg = sum(
        value / math.log2(position + 1)
        for position, value in enumerate(ideal, start=1)
        if value > 0.0
    )
    return dcg / idcg if idcg > 0.0 else 0.0


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
    """Compute mean hit@k, recall@k, precision@k, nDCG@k and MRR over a query set."""
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
        report["ndcg@{}".format(k)] = mean(
            [ndcg_at_k(r, g, k) for r, g in zip(rankings, relevancies)]
        )
    report["mrr"] = mean(
        [reciprocal_rank(r, g) for r, g in zip(rankings, relevancies)]
    )
    return report
