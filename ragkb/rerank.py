"""Maximal Marginal Relevance re-ranking.

The chunker in :mod:`ragkb.text` deliberately repeats trailing sentences at the
head of the next chunk, and a document usually discusses one topic throughout.
Both facts push the same way: the top ``k`` passages for a query are often
several near-copies of one passage. That is wasted context. It hurts an LLM
answerer (the same evidence three times crowds out a second supporting fact)
and it hurts the extractive answerer for exactly the same reason.

Maximal Marginal Relevance (Carbonell & Goldstein, "The Use of MMR,
Diversity-Based Reranking for Reordering Documents and Producing Summaries",
SIGIR 1998) selects results greedily, at each step maximising::

    lambda * relevance(d) - (1 - lambda) * max_{s in selected} similarity(d, s)

``lambda = 1.0`` reproduces the input ranking exactly; ``lambda = 0.0`` ignores
relevance and picks the most mutually dissimilar set. The interesting values sit
in between.

Relevance and similarity have to be on a comparable scale for the subtraction to
mean anything. Similarity here is a cosine in [0, 1], so the incoming relevance
scores are min-max normalised over the candidate pool before they are used. That
is what makes this work for RRF scores (which live near 1/60) and raw BM25
scores (which do not) without special-casing either.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

Candidate = Tuple[int, float]


def normalize_scores(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Min-max scale candidate scores into [0, 1], preserving order.

    A pool whose scores are all identical carries no relevance signal, so every
    item is given 1.0 and MMR degenerates to pure diversity selection.
    """
    if not candidates:
        return []
    scores = [score for _, score in candidates]
    lowest, highest = min(scores), max(scores)
    spread = highest - lowest
    if spread <= 0.0:
        return [(index, 1.0) for index, _ in candidates]
    return [(index, (score - lowest) / spread) for index, score in candidates]


def mmr_rerank(
    candidates: Sequence[Candidate],
    similarity: Callable[[int, int], float],
    k: int,
    lambda_: float = 0.7,
) -> List[Candidate]:
    """Re-order ``candidates`` by MMR and return the top ``k``.

    Args:
        candidates: ``(item_id, relevance)`` pairs, best first.
        similarity: symmetric similarity between two item ids, in [0, 1].
        k: how many items to select.
        lambda_: relevance/diversity trade-off in [0, 1].

    Returns:
        ``(item_id, relevance)`` pairs in selection order. The relevance value
        is the original score, not the MMR objective, so downstream code can
        still report a meaningful retrieval score.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    if k <= 0 or not candidates:
        return []

    original = {index: score for index, score in candidates}
    pool = normalize_scores(candidates)
    selected: List[Candidate] = []
    selected_ids: List[int] = []
    remaining = list(pool)

    while remaining and len(selected) < k:
        best_position = 0
        best_objective = None
        for position, (index, relevance) in enumerate(remaining):
            if selected_ids:
                redundancy = max(similarity(index, other) for other in selected_ids)
            else:
                redundancy = 0.0
            objective = lambda_ * relevance - (1.0 - lambda_) * redundancy
            # Strict ">" keeps the original ordering as the tie-break, so
            # lambda_ == 1.0 provably reproduces the input ranking.
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_position = position
        index, _ = remaining.pop(best_position)
        selected_ids.append(index)
        selected.append((index, original[index]))

    return selected


def mean_pairwise_similarity(
    ids: Sequence[int], similarity: Callable[[int, int], float]
) -> float:
    """Mean similarity over every unordered pair - a redundancy read-out.

    Returns 0.0 for fewer than two items, where redundancy is undefined.
    """
    if len(ids) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for position, left in enumerate(ids):
        for right in ids[position + 1 :]:
            total += similarity(left, right)
            pairs += 1
    return total / pairs if pairs else 0.0
