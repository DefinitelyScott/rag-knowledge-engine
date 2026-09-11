"""How big would the gold set have to be? Sample-size analysis for the sign test.

``ragkb/significance.py`` ends every negative result with the same sentence, and
``evals/significance.py`` prints it in as many words: *the fix is a bigger and
more discriminative gold set rather than a different test.* That sentence has
been the project's standing excuse for a year of NOT SHOWN verdicts, and until
now it was unfalsifiable, because nothing here could say how much bigger. This
module turns it into a number.

**This is a design tool, not a post-hoc one, and the distinction is the whole
point.** Computing "the power this test had to detect the effect I just
observed" is a well-known statistical anti-pattern: observed power is a
monotone, one-to-one function of the p-value, so it is the same number twice
and adds no information. A non-significant result *always* comes back with low
observed power, which is why reporting it as if it were a diagnosis is
circular. The legitimate question is prospective and conditional: *if the true
effect were this size, how many questions would it take to see it?* Every
function here answers that form. When :mod:`evals.power` feeds it the observed
win/loss counts it is supplying a **hypothesis about the true effect**, not an
estimate whose power is then reported as evidence, and the script says so on
the line where it does it.

The test being modelled is the exact two-sided paired sign test. That choice
lines up with the rest of the repo more tightly than it might look:

* For a **binary** per-query metric - hit@1, hit@5, anything valued in {0, 1} -
  every discordant question contributes a difference of exactly +1 or -1. The
  approximate-randomization test in :func:`ragkb.significance.permutation_test`
  is then *identical* to the sign test, because flipping signs on equal
  magnitudes only ever counts how many landed each way. For those metrics these
  sample sizes are exact, not an analogy.
* For a **graded** metric - MRR, nDCG@5 - the magnitudes differ, and the
  randomization test uses information the sign test throws away. Discarding
  informative magnitudes usually costs power, so the question counts here should
  be read as a conservative upper bound on those rows rather than a prediction.

The parameterisation deliberately splits the two things a gold set contributes,
because they are not interchangeable and conflating them is how "just add more
questions" becomes bad advice:

``discordance_rate``
    P(the two methods disagree on a question). A property of how
    *discriminative* the questions are.
``win_share``
    P(method A wins | they disagree). The effect size. 0.5 is the null.

Only discordant questions carry evidence - ties are invariant under the sign
flips and contribute nothing. So a question both methods already get right adds
to ``n`` and nothing to the test. Doubling the gold set with easy questions
barely moves the required-sample-size table; doubling the *discordance rate*
moves it roughly as much as doubling ``n``. That is the quantitative form of
the corpus-growth strategy the README describes - each new document placed
beside one it shares vocabulary with, so that some question gets a lexically
plausible wrong neighbour - and it is why that strategy is the right one rather
than simply writing more questions.

Standard library only, and exact rather than simulated: the binomial tails are
summed in log space with early termination, so a printed sample size is
reproducible to the digit and does not depend on a seed.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

#: Relative size at which a binomial term stops being worth adding. Terms below
#: this contribute less than float64 can represent next to the running total.
_NEGLIGIBLE = 1e-18


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    """``log P(X = k)`` for ``X ~ Binomial(n, p)``, via lgamma.

    Computed in log space because the factorials overflow long before the
    probabilities underflow: ``C(20000, 10000)`` has some six thousand digits,
    while the pmf it belongs to is a perfectly ordinary small float.
    """
    if k < 0 or k > n:
        return -math.inf
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binomial_lower_tail(c: int, n: int, p: float) -> float:
    """``P(X <= c)`` for ``X ~ Binomial(n, p)``.

    Summed downward from the largest term rather than upward from ``k = 0``.
    Anchoring on ``pmf(c)`` and accumulating ratios keeps every intermediate in
    a sane range: the naive version starts at ``(1 - p) ** n``, which underflows
    to exactly zero for ``n`` past about a thousand and silently returns 0.0 for
    a tail that is merely small.

    Early termination is gated on having passed the mode, since below it the
    terms shrink monotonically but above it they do not.
    """
    if n < 0:
        raise ValueError("n cannot be negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1], got {}".format(p))
    if c < 0:
        return 0.0
    if c >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_top = _log_binom_pmf(c, n, p)
    if log_top == -math.inf:
        return 0.0
    mode = int((n + 1) * p)
    odds = (1.0 - p) / p
    total = 1.0
    term = 1.0
    for index in range(c, 0, -1):
        term *= (index / float(n - index + 1)) * odds
        total += term
        if index - 1 < mode and term < _NEGLIGIBLE * total:
            break
    return min(1.0, math.exp(log_top) * total)


def binomial_upper_tail(m: int, n: int, p: float) -> float:
    """``P(X >= m)``, obtained by reflecting the lower tail.

    ``X >= m`` is the same event as ``n - X <= n - m``, and ``n - X`` is
    ``Binomial(n, 1 - p)``, so this reuses the stable summation above instead of
    duplicating it with the inequalities reversed.
    """
    return binomial_lower_tail(n - m, n, 1.0 - p)


def _binomial_window(
    n: int, p: float, threshold: float = 1e-13
) -> Tuple[int, List[float]]:
    """The pmf of ``Binomial(n, p)`` over the range that carries the mass.

    Returns ``(first_k, values)``. Walking outward from the mode and stopping
    once terms fall below ``threshold`` relative to the peak turns an ``O(n)``
    sum into an ``O(sqrt(n))`` one, which is what makes the outer integral in
    :func:`sign_test_power` affordable at ``n`` in the tens of thousands. The
    retained values are renormalised, so the truncated tails cost accuracy in
    the far decimals rather than biasing the total low.
    """
    if n < 0:
        raise ValueError("n cannot be negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1], got {}".format(p))
    if n == 0:
        return 0, [1.0]
    if p <= 0.0:
        return 0, [1.0]
    if p >= 1.0:
        return n, [1.0]

    mode = min(n, max(0, int((n + 1) * p)))
    odds = p / (1.0 - p)

    upper: List[float] = []
    term = 1.0
    for k in range(mode, n):
        term *= ((n - k) / float(k + 1)) * odds
        if term < threshold:
            break
        upper.append(term)

    lower: List[float] = []
    term = 1.0
    for k in range(mode, 0, -1):
        term *= (k / float(n - k + 1)) / odds
        if term < threshold:
            break
        lower.append(term)

    values = list(reversed(lower)) + [1.0] + upper
    start = mode - len(lower)
    total = math.fsum(values)
    return start, [value / total for value in values]


@lru_cache(maxsize=None)
def sign_test_critical_value(discordant: int, alpha: float = 0.05) -> int:
    """Largest ``c`` for which the exact two-sided sign test rejects at ``w <= c``.

    The rejection region is ``w <= c or w >= d - c``. Returns ``-1`` when no
    such ``c`` exists, which means the test *cannot* reject at this ``alpha`` no
    matter how lopsided the outcome - the discrete counterpart of
    :func:`ragkb.significance.smallest_achievable_p` returning a floor above
    ``alpha``.

    Cached because :func:`sign_test_power` evaluates it once per discordant
    count in the window and the same counts recur across every step of the
    searches below.
    """
    if discordant < 0:
        raise ValueError("discordant count cannot be negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    best = -1
    for candidate in range((discordant - 1) // 2 + 1):
        if 2.0 * binomial_lower_tail(candidate, discordant, 0.5) <= alpha:
            best = candidate
        else:
            break
    return best


def minimum_discordant_pairs(alpha: float = 0.05) -> int:
    """Fewest disagreements that could ever produce a significant result.

    Six at ``alpha = 0.05``, matching the floor documented on
    :func:`ragkb.significance.smallest_achievable_p`: with ``d`` discordant
    pairs the smallest reachable p-value is ``2 ** (1 - d)``, and that first
    drops to or below 0.05 at ``d = 6``. Below it a comparison is untestable by
    construction, which is a fact about the gold set rather than the methods.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    discordant = 1
    while discordant < 10000:
        if sign_test_critical_value(discordant, alpha) >= 0:
            return discordant
        discordant += 1
    raise ValueError("no attainable rejection region for alpha={}".format(alpha))


def power_given_discordant(
    discordant: int, win_share: float, alpha: float = 0.05
) -> float:
    """P(reject) conditional on the two methods disagreeing exactly ``discordant`` times.

    Under the alternative the wins among those disagreements are
    ``Binomial(discordant, win_share)``; the test rejects in the two tails
    delimited by :func:`sign_test_critical_value`. Symmetric about
    ``win_share = 0.5``, as a two-sided test should be - a method that loses
    consistently is just as detectable as one that wins consistently.
    """
    if not 0.0 <= win_share <= 1.0:
        raise ValueError("win_share must be in [0, 1], got {}".format(win_share))
    critical = sign_test_critical_value(discordant, alpha)
    if critical < 0:
        return 0.0
    return binomial_lower_tail(
        critical, discordant, win_share
    ) + binomial_upper_tail(discordant - critical, discordant, win_share)


def sign_test_power(
    n: int,
    discordance_rate: float,
    win_share: float,
    alpha: float = 0.05,
) -> float:
    """Power of the exact sign test over ``n`` questions.

    The number of discordant pairs is itself random - the gold set does not come
    with a guaranteed count of disagreements - so the power is the expectation
    of :func:`power_given_discordant` over ``D ~ Binomial(n, discordance_rate)``.
    Treating ``D`` as fixed at its mean overstates power at these sample sizes,
    because the conditional power is concave in ``D`` over the range that
    matters.
    """
    if n < 0:
        raise ValueError("n cannot be negative")
    if not 0.0 <= discordance_rate <= 1.0:
        raise ValueError(
            "discordance_rate must be in [0, 1], got {}".format(discordance_rate)
        )
    if n == 0 or discordance_rate == 0.0:
        return 0.0
    start, weights = _binomial_window(n, discordance_rate)
    return math.fsum(
        weight * power_given_discordant(start + offset, win_share, alpha)
        for offset, weight in enumerate(weights)
        if weight > 0.0
    )


#: Ceiling for the searches below. Twenty thousand questions is roughly two
#: hundred times the current gold set; if an effect needs more than that, the
#: useful output is "unreachable" rather than a number nobody will act on.
MAX_QUESTIONS = 20000


def required_questions(
    discordance_rate: float,
    win_share: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    max_questions: int = MAX_QUESTIONS,
) -> Optional[int]:
    """Smallest gold set reaching ``target_power`` against the stated alternative.

    ``None`` means the target is out of reach within ``max_questions``.

    Found by doubling to bracket, then bisecting. Bisection assumes power rises
    with ``n``, which is true of the expectation computed here but *not* of the
    conditional power at fixed ``D``: the exact test's rejection region moves in
    integer steps, so conditional power sawtooths. Averaging over the binomial
    distribution of ``D`` smooths that out, and ``tests/test_power.py`` pins the
    monotonicity on a grid rather than leaving it asserted here.
    """
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if max_questions < 1:
        raise ValueError("max_questions must be positive")
    if discordance_rate <= 0.0 or win_share == 0.5:
        return None

    bound = 1
    while bound <= max_questions:
        if sign_test_power(bound, discordance_rate, win_share, alpha) >= target_power:
            break
        bound *= 2
    else:
        if sign_test_power(max_questions, discordance_rate, win_share, alpha) < target_power:
            return None
        bound = max_questions

    low, high = bound // 2, min(bound, max_questions)
    while low < high:
        middle = (low + high) // 2
        if sign_test_power(middle, discordance_rate, win_share, alpha) >= target_power:
            high = middle
        else:
            low = middle + 1
    return low


def smallest_detectable_win_share(
    n: int,
    discordance_rate: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    tolerance: float = 1e-4,
) -> Optional[float]:
    """The mildest effect ``n`` questions could catch, as a win share above 0.5.

    This is the question the README actually needs answered - not "was this
    difference significant" but "what would this gold set have been *able* to
    show?" A returned 0.85 means: unless the better method wins 85% of the
    disagreements, this many questions will usually miss it.

    ``None`` when even a clean sweep of every disagreement would not reach
    ``target_power``, which is the honest verdict for a small or undiscriminating
    gold set.
    """
    if n < 0:
        raise ValueError("n cannot be negative")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if discordance_rate <= 0.0:
        return None
    if sign_test_power(n, discordance_rate, 1.0, alpha) < target_power:
        return None
    low, high = 0.5, 1.0
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if sign_test_power(n, discordance_rate, middle, alpha) >= target_power:
            high = middle
        else:
            low = middle
    return high


def rates_from_counts(wins: int, losses: int, n: int) -> Tuple[float, float]:
    """``(discordance_rate, win_share)`` from an observed ``W/L/T`` line.

    The plug-in estimates. Feeding these straight back into
    :func:`required_questions` is the supported use, but only under the reading
    set out in this module's docstring: the result is "how many questions would
    an effect *of this size* need", a conditional statement about a hypothesised
    truth. It is not the retrospective power of the test that produced the
    counts, and it must not be reported as though the observed split were known
    to be correct - on eight disagreements the win share is estimated to about
    plus or minus a third.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    discordant = wins + losses
    if discordant > n:
        raise ValueError(
            "wins + losses cannot exceed n: {} + {} > {}".format(wins, losses, n)
        )
    if wins < 0 or losses < 0:
        raise ValueError("counts cannot be negative")
    if discordant == 0:
        return 0.0, 0.5
    return discordant / float(n), wins / float(discordant)


def clopper_pearson(
    successes: int, trials: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Exact two-sided confidence interval for a binomial proportion.

    Both rates this module trades in - ``discordance_rate`` and ``win_share`` -
    are proportions estimated from counts that are routinely in the single
    digits, and :func:`rates_from_counts` hands them over as bare point
    estimates. Two disagreements out of fourteen questions is 14.3%, and quoting
    that figure beside a 14.1% measured on ninety-two questions invites a
    comparison the smaller sample cannot support. This function is what makes
    that visible: the same 2/14 carries an interval of roughly 1.8% to 42.8%,
    which is most of the range the rate could plausibly take.

    The interval is the Clopper-Pearson one, defined by inverting the exact
    binomial test rather than by a normal approximation. The lower bound is the
    ``p`` at which ``P(X >= successes) = (1 - confidence) / 2``, and the upper
    bound the ``p`` at which ``P(X <= successes) = (1 - confidence) / 2``; each
    tail is monotone in ``p``, so bisection finds them without needing a beta
    quantile from outside the standard library. The result is conservative -
    actual coverage is at least ``confidence``, usually a little above - which
    is the right direction to err for a number whose job is to stop a small
    sample being over-read.

    ``successes = 0`` and ``successes = trials`` are handled as the degenerate
    cases they are: the interval runs to the boundary on that side exactly,
    because no evidence contradicts a rate of zero (or one).
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError(
            "successes must be in [0, {}], got {}".format(trials, successes)
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1), got {}".format(confidence))

    tail = (1.0 - confidence) / 2.0

    def _bisect(predicate) -> float:
        low, high = 0.0, 1.0
        for _ in range(200):
            mid = (low + high) / 2.0
            if predicate(mid):
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    if successes == 0:
        lower = 0.0
    else:
        # P(X >= successes) rises with p; the bound is where it reaches `tail`.
        lower = _bisect(lambda p: binomial_upper_tail(successes, trials, p) < tail)
    if successes == trials:
        upper = 1.0
    else:
        # P(X <= successes) falls with p; the bound is where it drops to `tail`.
        upper = _bisect(lambda p: binomial_lower_tail(successes, trials, p) > tail)
    return lower, upper


def mcnemar_exact(gained: int, lost: int) -> float:
    """Exact two-sided p-value for a change in a paired binary classification.

    Written for the one comparison :mod:`evals.corpus_lever` makes that two
    independent confidence intervals cannot make: the discordance rate before
    and after a corpus change is measured on **the same questions**, so the two
    rates are paired, and a pair of marginal intervals throws that pairing away
    and reports far more uncertainty than the design actually carries.

    Only the questions that *changed* status carry information, exactly as only
    discordant questions carry information in the sign test this module is
    otherwise about. A question discordant both before and after, or concordant
    both times, is uninformative about whether anything moved. Under the null
    that the change was symmetric, each of the ``gained + lost`` switchers is an
    independent coin flip, so this is the same exact binomial as everywhere else
    in this module, applied one level up.

    Returns 1.0 when nothing switched, which is the correct reading of "no
    evidence of a change" rather than a degenerate case needing a guard.
    """
    if gained < 0 or lost < 0:
        raise ValueError("counts cannot be negative")
    switchers = gained + lost
    if switchers == 0:
        return 1.0
    smaller = min(gained, lost)
    return min(1.0, 2.0 * binomial_lower_tail(smaller, switchers, 0.5))


def requirement_grid(
    discordance_rates: Sequence[float],
    win_shares: Sequence[float],
    alpha: float = 0.05,
    target_power: float = 0.80,
    max_questions: int = MAX_QUESTIONS,
) -> Dict[Tuple[float, float], Optional[int]]:
    """Required gold-set size across a grid of assumed truths.

    The grid is the part of this module that owes nothing to the observed data,
    and therefore the part worth reading first: it is a statement about what
    this *kind* of benchmark can resolve, fixed before any result is looked at.
    """
    return {
        (rate, share): required_questions(
            rate, share, alpha=alpha, target_power=target_power,
            max_questions=max_questions,
        )
        for rate in discordance_rates
        for share in win_shares
    }
