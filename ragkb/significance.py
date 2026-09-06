"""Paired significance testing for retrieval metrics.

Every table in this project reports a mean over the same 78 gold questions, and
several of the differences it prints are one question wide. On a 78-question set
one question is worth 0.013 of any per-query metric, so a 0.02 gap in hit@1 is
not a result - it is the measurement floor. Nothing in the repo could tell those
two cases apart, which made the sweep tables easy to over-read.

This module supplies the missing half of those tables. Two procedures, both
*paired* - every method is scored on exactly the same questions, so the
comparison should use each question as its own control rather than treating the
two means as independent samples:

* :func:`paired_bootstrap` resamples the questions with replacement and reports
  a confidence interval for the mean difference. It answers "how much would this
  delta move if I had drawn a different gold set of the same size?"
* :func:`permutation_test` (paired approximate randomization, the standard test
  in the IR literature) shuffles which side of each pair a difference falls on
  and reports how often chance alone produces a gap this large. It answers "is
  this delta distinguishable from noise?"

They are kept separate because they answer different questions and are
routinely confused; :func:`compare` runs both and packages the result together
with the discordant-pair counts, which is usually the most honest single line in
the output - "won 3, lost 4, tied 49" says more than any p-value.

Standard library only, like the rest of the engine: the resampling uses
:class:`random.Random` seeded explicitly so that a reported interval can be
reproduced exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


def _check_pair(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError(
            "paired tests need equal-length score vectors: {} != {}".format(
                len(a), len(b)
            )
        )
    if not a:
        raise ValueError("cannot compare empty score vectors")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def resolution(n: int) -> float:
    """The smallest non-zero delta a per-query metric can produce over ``n`` queries.

    hit@k and MRR are per-query scores averaged over the set, so a difference
    between two methods is always some multiple of ``1 / n`` for binary metrics.
    Printing this next to a table is the cheapest possible guard against reading
    a rounding artefact as an improvement.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 / n


def smallest_achievable_p(discordant: int) -> float:
    """Lowest two-sided p-value :func:`permutation_test` can return.

    Only the questions where the two methods disagree carry any evidence; the
    ties are invariant under the sign flips. With ``d`` discordant pairs there
    are ``2 ** d`` equally likely sign assignments, and at most two of them are
    as extreme as an all-one-direction observation, so the floor is
    ``2 ** (1 - d)``.

    The consequence is worth stating plainly, because it is not obvious from
    the p-value column alone: **one discordant pair can never be significant**
    (the floor is 1.0), and fewer than six can never clear 0.05 - no matter how
    many questions the gold set contains. A table row reading ``0/1/55`` is not
    a weak result, it is *no* result, and the fix is a bigger gold set rather
    than more resamples.
    """
    if discordant < 0:
        raise ValueError("discordant count cannot be negative")
    if discordant == 0:
        return 1.0
    return min(1.0, 2.0 ** (1 - discordant))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Used on its own this puts an error bar on a single method's score; used on
    the per-query *differences* it becomes :func:`paired_bootstrap`.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return _percentile(means, tail), _percentile(means, 1.0 - tail)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """``(delta, ci_low, ci_high)`` for ``mean(a) - mean(b)``.

    Queries - not scores - are resampled, so each draw keeps the two methods'
    results for a question together. That pairing is the whole point: the
    question-to-question variance is enormous next to the difference between two
    methods on the same question, and an unpaired interval would drown the
    signal in it.
    """
    _check_pair(a, b)
    differences = [x - y for x, y in zip(a, b)]
    low, high = bootstrap_ci(differences, resamples, confidence, seed)
    return mean(differences), low, high


#: Discordant-pair count below which :func:`permutation_test` enumerates every
#: sign assignment instead of sampling them. ``2 ** 14`` assignments is a few
#: milliseconds, and this project lives almost entirely in that regime - two
#: retrievers on the same 64-question set typically disagree on four to ten
#: questions - so the common case gets an exact p-value rather than a noisy
#: estimate of one.
EXACT_LIMIT = 14


def _exact_permutation(differences: Sequence[float], observed: float) -> float:
    """Enumerate all ``2 ** d`` sign assignments over the discordant pairs."""
    n = len(differences)
    nonzero = [value for value in differences if value != 0.0]
    total_assignments = 1 << len(nonzero)
    extreme = 0
    for mask in range(total_assignments):
        total = 0.0
        for index, value in enumerate(nonzero):
            total += value if (mask >> index) & 1 else -value
        if abs(total / n) >= observed - 1e-12:
            extreme += 1
    return extreme / float(total_assignments)


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 10000,
    seed: int = 0,
    exact_limit: int = EXACT_LIMIT,
) -> float:
    """Two-sided paired randomization p-value.

    Under the null hypothesis the two systems are interchangeable, so for any
    question it is equally likely that the observed difference was ``a - b`` or
    ``b - a``. The p-value is the share of sign assignments whose mean
    difference reaches the observed magnitude. Questions where the two methods
    tie contribute nothing either way, which is exactly right - they carry no
    evidence.

    Two regimes, and which one ran matters:

    * When the methods disagree on at most ``exact_limit`` questions the test
      enumerates every assignment and returns the *exact* p-value. Sampling in
      that regime is not merely wasteful, it is wrong in a way that is easy to
      miss: a Monte-Carlo estimate scatters around the true value and can land
      below the floor set by :func:`smallest_achievable_p`, which invites the
      reader to over-claim from four discordant questions.
    * Above that, it samples ``resamples`` assignments. The ``+1`` on both sides
      of the ratio counts the observed assignment itself, so a sampled result is
      never 0.0: the honest statement is "below ``1/(resamples + 1)``", not
      "impossible".
    """
    _check_pair(a, b)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if exact_limit < 0:
        raise ValueError("exact_limit cannot be negative")
    differences = [x - y for x, y in zip(a, b)]
    # >= rather than > throughout, so that an all-ties comparison returns 1.0
    # instead of claiming significance from a degenerate distribution at zero.
    observed = abs(mean(differences))
    discordant = sum(1 for value in differences if value != 0.0)
    if discordant <= exact_limit:
        return _exact_permutation(differences, observed)
    rng = random.Random(seed)
    n = len(differences)
    extreme = 0
    for _ in range(resamples):
        total = 0.0
        for value in differences:
            total += value if rng.random() < 0.5 else -value
        if abs(total / n) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / float(resamples + 1)


@dataclass(frozen=True)
class Comparison:
    """One method-versus-method result on one metric."""

    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    wins: int
    losses: int
    ties: int
    confidence: float

    @property
    def discordant(self) -> int:
        """Questions on which the two methods actually disagree.

        The real sample size of the comparison. ``wins + losses`` is usually a
        single-digit number even when ``n`` is in the dozens, which is the
        clearest explanation of why so few rows in the output are significant.
        """
        return self.wins + self.losses

    @property
    def p_floor(self) -> float:
        """The lowest p-value this comparison *could* have produced."""
        return smallest_achievable_p(self.discordant)

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero.

        Deliberately not thresholded on the p-value: with a 64-question set the
        interval is the more informative object, and reporting "the CI includes
        zero" keeps the reader's attention on the size of the uncertainty rather
        than on which side of 0.05 a number landed.
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def describe(self) -> str:
        return (
            "{:+.3f} [{:+.3f}, {:+.3f}] p={:.3f} ({}W/{}L/{}T)".format(
                self.delta,
                self.ci_low,
                self.ci_high,
                self.p_value,
                self.wins,
                self.losses,
                self.ties,
            )
        )


def compare(
    a: Sequence[float],
    b: Sequence[float],
    bootstrap_resamples: int = 2000,
    permutations: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
    exact_limit: int = EXACT_LIMIT,
) -> Comparison:
    """Run both procedures over one pair of per-query score vectors."""
    _check_pair(a, b)
    delta, low, high = paired_bootstrap(
        a, b, resamples=bootstrap_resamples, confidence=confidence, seed=seed
    )
    p_value = permutation_test(
        a, b, resamples=permutations, seed=seed, exact_limit=exact_limit
    )
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    return Comparison(
        n=len(a),
        mean_a=mean(a),
        mean_b=mean(b),
        delta=delta,
        ci_low=low,
        ci_high=high,
        p_value=p_value,
        wins=wins,
        losses=losses,
        ties=len(a) - wins - losses,
        confidence=confidence,
    )


def holm_bonferroni(
    p_values: Sequence[float], alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """Step-down familywise correction over a family of p-values.

    A single test at ``alpha = 0.05`` is wrong 5% of the time by construction.
    A *table* of tests is wrong far more often: the chance that at least one of
    ``m`` independent true nulls produces a p below 0.05 is ``1 - 0.95 ** m``,
    which is 54% at ``m = 16`` and 99.9% at ``m = 150``. Any script that prints
    a grid of comparisons and invites the reader to notice the small numbers is
    running exactly that experiment, so the grid needs a correction or it needs
    a disclaimer.

    Holm's procedure is the one to reach for here rather than plain Bonferroni.
    Both control the familywise error rate under arbitrary dependence - which
    matters, because these comparisons are anything but independent: the same
    questions and the same three retrievers appear in every row. But Bonferroni
    tests every hypothesis at ``alpha / m``, while Holm sorts the p-values
    ascending and tests the ``i``-th against ``alpha / (m - i)``, so it is
    uniformly more powerful and never rejects less. There is no cost to
    preferring it.

    Returns one ``(adjusted_p, reject)`` pair per input, in the input order.
    ``adjusted_p`` is the raw p scaled by its step-down multiplier, clamped to
    1.0 and forced non-decreasing along the sorted order - without that
    monotonicity fix a later, larger raw p could report a smaller adjusted
    value, which is incoherent. Comparing ``adjusted_p`` against the original
    ``alpha`` gives the same decisions as the sequential procedure.

    The expected result on this project's gold set is that *nothing* survives,
    and that is the honest headline rather than a disappointment: it says the
    gold set is too small and too easy to separate these methods, which is a
    statement about the benchmark and a concrete argument for growing it.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    for value in p_values:
        if not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be in [0, 1], got {}".format(value))
    total = len(p_values)
    if total == 0:
        return []
    order = sorted(range(total), key=lambda index: p_values[index])
    adjusted = [0.0] * total
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (total - rank) * p_values[index]
        running = max(running, scaled)
        adjusted[index] = min(1.0, running)
    return [(adjusted[index], adjusted[index] <= alpha) for index in range(total)]
