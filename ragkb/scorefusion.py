"""Score-based fusion: CombSUM and CombMNZ over normalised list scores.

Reciprocal Rank Fusion throws every score away and keeps only the rank. That is
its selling point - BM25, TF-IDF cosine and character n-gram cosine live on
unrelated scales, and RRF never has to reconcile them - and it is also the
obvious suspect for the finding `evals/evaluate.py` keeps reporting: the
character n-gram list on its own beats every fusion that contains it. If RRF is
diluting the strongest list because it cannot see that the n-gram list was
*confident* about its top hit while BM25 was not, then a fusion that does read
the scores should close some of that gap.

This module is the classic alternative from Fox & Shaw ("Combination of
Multiple Searches", TREC-2, 1994), with the normalisation step Lee
("Analyses of Multiple Evidence Combination", SIGIR 1997) showed it needs:

1. normalise each list's scores onto a comparable scale (min-max or z-score);
2. **CombSUM**: sum each candidate's normalised scores across lists;
3. **CombMNZ**: CombSUM multiplied by the number of lists that returned the
   candidate at all, which rewards agreement between lists.

A candidate a list did not return gets that list's *lowest* normalised score,
not zero. For min-max the two are the same thing; for z-scores, where the
bottom of a list is negative, substituting zero would rank "not retrieved"
above "retrieved, but badly", which is backwards.

It is deliberately *not* wired into :class:`ragkb.retriever.HybridRetriever`.
The abstention threshold, the attribution arithmetic in ``ragkb.cli explain``
and the persisted config are all calibrated against RRF's score scale, and a
second fusion mode in the main path would have to earn the cost of keeping all
three honest under both. `evals/score_fusion.py` measures whether it does; the
README records the answer. :func:`combine` also implements ``"rrf"`` so the
harness can prove it reproduces the retriever's own fusion bit-for-bit before
comparing anything against it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

from .text import tokenize

Hits = Sequence[Tuple[int, float]]

NORMALISATIONS = ("minmax", "zscore")
COMBINATIONS = ("rrf", "combsum", "combmnz")


def normalise(hits: Hits, method: str = "minmax") -> Dict[int, float]:
    """Rescale one list's ``(index, score)`` pairs onto a common scale.

    ``minmax`` maps the list onto [0, 1]; ``zscore`` centres it on its own mean
    in units of its own (population) standard deviation. A list whose scores
    are all equal carries no ordering information, so every member gets the
    same value (1.0 for min-max, 0.0 for z-score) rather than a division by zero.
    """
    if method not in NORMALISATIONS:
        raise ValueError(
            "normalisation must be one of {}; got {!r}".format(NORMALISATIONS, method)
        )
    if not hits:
        return {}
    scores = [score for _, score in hits]
    if method == "minmax":
        low, high = min(scores), max(scores)
        span = high - low
        if span == 0.0:
            return {index: 1.0 for index, _ in hits}
        return {index: (score - low) / span for index, score in hits}
    centre = sum(scores) / len(scores)
    spread = math.sqrt(sum((score - centre) ** 2 for score in scores) / len(scores))
    if spread == 0.0:
        return {index: 0.0 for index, _ in hits}
    return {index: (score - centre) / spread for index, score in hits}


def combine(
    lists: Mapping[str, Hits],
    combination: str = "combsum",
    normalisation: str = "minmax",
    rrf_k: int = 60,
) -> List[Tuple[int, float]]:
    """Fuse several ranked lists into one, best first, ties broken on index.

    ``lists`` maps a list name to its ``(index, score)`` pairs in rank order -
    exactly what each index's ``search`` returns. ``normalisation`` is ignored
    for ``"rrf"``, which by construction never looks at a score.
    """
    if combination not in COMBINATIONS:
        raise ValueError(
            "combination must be one of {}; got {!r}".format(COMBINATIONS, combination)
        )
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    fused: Dict[int, float] = {}
    if combination == "rrf":
        for hits in lists.values():
            for rank, (index, _) in enumerate(hits, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (rrf_k + rank)
    else:
        normalised = {name: normalise(hits, normalisation) for name, hits in lists.items()}
        candidates = set()
        for values in normalised.values():
            candidates.update(values)
        floors = {
            name: min(values.values()) for name, values in normalised.items() if values
        }
        for index in candidates:
            total = 0.0
            present = 0
            for name, values in normalised.items():
                if index in values:
                    total += values[index]
                    present += 1
                elif name in floors:
                    total += floors[name]
            fused[index] = total * present if combination == "combmnz" else total
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


def fused_pool(
    retriever,
    query: str,
    lists: Sequence[str],
    combination: str = "combsum",
    normalisation: str = "minmax",
    candidates: int = 25,
    k: int = 5,
) -> List[Tuple[int, float]]:
    """Run ``query`` against the named lists of a fitted retriever and fuse them.

    Each list is searched to the same depth the retriever itself uses -
    ``max(candidates, k)`` - so the comparison against the retriever's own RRF
    changes the arithmetic and nothing else.
    """
    available = {name: index for name, index in retriever._lists()}
    unknown = [name for name in lists if name not in available]
    if unknown:
        raise ValueError(
            "unknown list(s) {}; this retriever has {}".format(
                unknown, sorted(available)
            )
        )
    tokens = tokenize(query)
    if not tokens:
        return []
    depth = max(candidates, k)
    hits = {name: available[name].search(list(tokens), k=depth) for name in lists}
    return combine(hits, combination, normalisation, rrf_k=retriever.rrf_k)


def ranked_doc_ids(
    retriever,
    query: str,
    lists: Sequence[str],
    combination: str = "combsum",
    normalisation: str = "minmax",
    candidates: int = 25,
    k: int = 5,
) -> List[str]:
    """Deduplicated document ids in rank order, mirroring ``RAGEngine.ranked_doc_ids``.

    The engine asks the retriever for ``4 * k`` chunks and keeps the first
    ``k`` distinct documents among them; this does the same, so a document
    that only surfaces past chunk ``4 * k`` is missed here exactly as it would
    be there.
    """
    chunk_k = max(k * 4, k)
    pool = fused_pool(
        retriever, query, lists, combination, normalisation, candidates, chunk_k
    )
    ordered: List[str] = []
    for index, _ in pool[:chunk_k]:
        doc_id = retriever.chunks[index].doc_id
        if doc_id not in ordered:
            ordered.append(doc_id)
        if len(ordered) >= k:
            break
    return ordered
