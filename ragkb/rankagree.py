"""How much do two ranked lists agree - measured in the unit fusion consumes?

`evals/evaluate.py` already answers a version of this question. Its
``agreement()`` reports the mean pairwise *set* overlap of the top-k, and the
README leans on that number hard: "the two lists agree on 0.838 of their top-5
by chunk, and a fusion constant has no room to arbitrate lists that already
concur."

Set overlap cannot support that sentence, because set overlap is blind to
order and RRF is not. ``1 / (rrf_k + rank)`` reads nothing but rank. Two lists
holding the same five chunks in reversed order have a set overlap of 1.000 and
produce different fused scores, different hit@1, different MRR and different
nDCG@k. Every accuracy column this project reports except recall@k is
order-sensitive; the one statistic used to explain them is not.

The gap is not hypothetical on this corpus. BM25 and TF-IDF share 0.838 of
their top-5 and return the *same five chunks in the same order* on 9 of 132
questions - 0.068. A reader given only the first number would conclude the two
lists are near-duplicates. A reader given only the second would conclude they
are unrelated. Both numbers are correct; neither is the measurement the
argument needs.

So this module adds the order-aware half, and keeps the set-aware half beside
it so the two can be read against each other rather than in isolation.

Three measures, chosen because they fail in different places:

:func:`top_k_set_overlap`
    The existing unit, reimplemented here so all three come from one place.
    Ignores order entirely. Upper bound on how much any order-aware measure
    can find, since two lists cannot agree on the rank of an item only one of
    them retrieved.

:func:`kendall_tau_b`
    The textbook rank correlation, computed over the items *both* lists
    contain. Conjoint-only is the standard treatment for truncated lists and
    it carries a sharp caveat: with a top-5 cut, the conjoint set here
    averages 4.2 items for BM25-against-TF-IDF and 2.7 for the cross-
    representation pairs, and tau over two items is either +1 or -1. That is
    why the conjoint size is returned alongside the coefficient rather than
    buried - a tau computed on 2.7 items is not a number to quote on its own,
    and the pairs with the lowest tau are exactly the pairs where it is least
    trustworthy. Reporting it without its n would invert the finding.

:func:`rank_biased_overlap`
    Webber, Moffat & Zobel, "A Similarity Measure for Indefinite Rankings"
    (TOIS 2010). Built for precisely the case tau handles badly: top-weighted,
    truncated, non-conjoint lists. It sums set overlap at every depth with a
    geometrically decaying weight, so disagreement at rank 1 costs more than
    disagreement at rank 5, and items present in only one list are handled by
    construction instead of by exclusion. This is the measure that matches
    what a fused ranking is actually sensitive to.

None of the three is "the right one". The finding this module exists to
support is what happens when a claim stated in one of them is restated in the
others - see `evals/rank_agreement.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional, Sequence, Tuple

#: Default persistence for :func:`rank_biased_overlap`.
#:
#: RBO's ``p`` sets how quickly attention decays down the ranking, and Webber
#: et al. give it a reading: the expected depth a user examines is ``1/(1-p)``.
#: At ``p = 0.8`` that is 5, which is this project's ``k``. Picking ``p`` to
#: match the evaluation depth is the only choice here that is not arbitrary -
#: the common ``p = 0.9`` implies a user reading to depth 10 and would spend
#: most of its weight below the window every table in the README reports.
DEFAULT_RBO_P = 0.8


def _check_depth(depth: Optional[int], *lists: Sequence[Hashable]) -> int:
    if depth is None:
        depth = max((len(item) for item in lists), default=0)
    if depth < 0:
        raise ValueError("depth cannot be negative")
    return depth


def top_k_set_overlap(
    left: Sequence[Hashable], right: Sequence[Hashable], depth: Optional[int] = None
) -> float:
    """Share of the top-``depth`` the two lists hold in common, ignoring order.

    Normalised by ``depth`` rather than by the size of the union, so a pair of
    full-length lists that share everything scores 1.0 and the value is
    directly comparable across questions where one list came back short.

    Returns 0.0 at ``depth = 0``: no window, nothing to agree about.
    """
    depth = _check_depth(depth, left, right)
    if depth == 0:
        return 0.0
    shared = set(left[:depth]) & set(right[:depth])
    return len(shared) / float(depth)


@dataclass(frozen=True)
class TauResult:
    """A rank correlation and the evidence it rests on.

    ``tau`` is ``None`` when fewer than two items are shared, because a
    correlation over one point is undefined rather than zero. ``conjoint`` is
    the count that decides whether the coefficient means anything, and
    ``pairs`` is the number of item pairs it was computed over.
    """

    tau: Optional[float]
    conjoint: int
    pairs: int


def kendall_tau_b(
    left: Sequence[Hashable], right: Sequence[Hashable], depth: Optional[int] = None
) -> TauResult:
    """Kendall's tau-b over the items both truncated lists contain.

    Every pair of shared items is concordant (both lists order them the same
    way) or discordant (they disagree), and tau is the normalised difference.
    With no ties possible - a ranking gives each item one position - tau-b
    reduces to tau-a over the conjoint subset, so the ``b`` here is about the
    treatment of the *truncation*, not about tied scores.

    The restriction to shared items is the standard workaround for lists that
    do not cover the same universe, and it is a real limitation rather than a
    technicality: it silently discards the strongest evidence of disagreement,
    which is one list retrieving something the other never saw at all. Two
    lists sharing a single item score ``None``; two lists sharing two items
    score exactly +1 or -1. Read :attr:`TauResult.conjoint` before reading
    :attr:`TauResult.tau`, and prefer :func:`rank_biased_overlap` when the
    conjoint set is small.
    """
    depth = _check_depth(depth, left, right)
    a = list(left[:depth])
    b = list(right[:depth])
    rank_b = {item: index for index, item in enumerate(b)}
    shared = [item for item in a if item in rank_b]
    n = len(shared)
    if n < 2:
        return TauResult(tau=None, conjoint=n, pairs=0)

    rank_a = {item: index for index, item in enumerate(a)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = shared[i], shared[j]
            direction = (rank_a[x] - rank_a[y]) * (rank_b[x] - rank_b[y])
            if direction > 0:
                concordant += 1
            elif direction < 0:
                discordant += 1
    pairs = n * (n - 1) // 2
    return TauResult(
        tau=(concordant - discordant) / float(pairs), conjoint=n, pairs=pairs
    )


def rank_biased_overlap(
    left: Sequence[Hashable],
    right: Sequence[Hashable],
    p: float = DEFAULT_RBO_P,
    depth: Optional[int] = None,
) -> Tuple[float, float]:
    """Top-weighted agreement between two truncated rankings, plus its residual.

    Returns ``(base, residual)``.

    ``base`` is RBO evaluated over the visible prefix::

        base = (1 - p) * sum_{d=1..depth} p^(d-1) * |L_:d & R_:d| / d

    which is set overlap computed at *every* depth and averaged under a
    geometric weight. That is what makes it order-aware without needing the
    lists to be conjoint: an item ranked 1 by one list and 5 by the other
    contributes to the overlap at depth 5 but not at depths 1 through 4, so it
    is partially credited rather than counted or discarded.

    ``residual`` is the honest part. RBO is defined over unbounded rankings and
    this one is cut at ``depth``, so the true value lies somewhere in
    ``[base, base + residual]`` with ``residual = p ** depth`` - the entire
    remaining weight, which the unseen tail could contribute in full if the
    lists agreed perfectly below the cut. At ``p = 0.8`` and ``depth = 5`` that
    is 0.328, and it is large because comparing five-item prefixes genuinely
    does leave that much unknown. The extrapolated variant in the paper
    replaces this interval with a point estimate by assuming the agreement
    seen so far continues; that assumption is unverifiable here, so the
    interval is returned instead and the caller decides.

    Both returned values are 0.0 at ``depth = 0``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    depth = _check_depth(depth, left, right)
    if depth == 0:
        return 0.0, 0.0

    a = list(left[:depth])
    b = list(right[:depth])
    seen_a: set = set()
    seen_b: set = set()
    total = 0.0
    for d in range(1, depth + 1):
        # Grow both prefixes by one, then intersect. An incremental counter
        # would save a set build per depth and cost the reader the one line
        # they most need to trust; at depth 5 that is not a trade worth making.
        # A list shorter than ``depth`` simply stops contributing, which is the
        # correct behaviour - the denominator stays ``d``, so a short list is
        # penalised for the ranks it never filled.
        if d <= len(a):
            seen_a.add(a[d - 1])
        if d <= len(b):
            seen_b.add(b[d - 1])
        total += (p ** (d - 1)) * (len(seen_a & seen_b) / float(d))
    return (1.0 - p) * total, p**depth


@dataclass(frozen=True)
class Agreement:
    """All three views of one pair of ranked lists, measured at one depth."""

    depth: int
    set_overlap: float
    rbo: float
    rbo_residual: float
    tau: Optional[float]
    conjoint: int
    identical: bool
    same_head: bool


def compare_rankings(
    left: Sequence[Hashable],
    right: Sequence[Hashable],
    depth: Optional[int] = None,
    p: float = DEFAULT_RBO_P,
) -> Agreement:
    """Measure one pair of rankings every way this module knows how.

    ``identical`` and ``same_head`` are the two endpoints the continuous
    measures sit between, and they are here because they are the two readings
    that cannot be argued with. ``identical`` is the strictest possible notion
    of agreement - same items, same order, no partial credit - and
    ``same_head`` is the loosest one that still concerns order, since rank 1 is
    where every early-precision metric in this project is decided.
    """
    depth = _check_depth(depth, left, right)
    a = list(left[:depth])
    b = list(right[:depth])
    tau = kendall_tau_b(a, b, depth=depth)
    base, residual = rank_biased_overlap(a, b, p=p, depth=depth)
    return Agreement(
        depth=depth,
        set_overlap=top_k_set_overlap(a, b, depth=depth),
        rbo=base,
        rbo_residual=residual,
        tau=tau.tau,
        conjoint=tau.conjoint,
        identical=a == b,
        same_head=bool(a) and bool(b) and a[0] == b[0],
    )
