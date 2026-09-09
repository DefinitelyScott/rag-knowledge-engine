"""Tests for the sample-size machinery.

A power calculation is an instrument whose output nobody can eyeball for
plausibility - "you need 358 questions" looks exactly as convincing whether or
not the binomial underneath it is right. So these tests pin it against things
that are known independently: exact rational binomial sums, the published
identity that the smallest reachable p-value is ``2 ** (1 - d)``, the
requirement that a test at ``alpha`` reject at most ``alpha`` of the time when
nothing is going on, and the symmetry a two-sided test must have.

The two structural assumptions the searches rely on - monotonicity of power in
``n`` and in the effect size - are pinned on a grid here rather than asserted in
a docstring, because bisection over a non-monotone function returns a wrong
answer quietly.
"""

from __future__ import annotations

import importlib.util
import math
import os

import pytest

from ragkb.power import (
    binomial_lower_tail,
    binomial_upper_tail,
    minimum_discordant_pairs,
    power_given_discordant,
    rates_from_counts,
    required_questions,
    requirement_grid,
    sign_test_critical_value,
    sign_test_power,
    smallest_detectable_win_share,
)
from ragkb.significance import smallest_achievable_p

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def exact_lower_tail(c, n, p):
    """Direct sum, used only as an oracle for the stable implementation."""
    return math.fsum(
        math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(c + 1)
    )


# --------------------------------------------------------------------------
# binomial tails
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "c,n,p",
    [(0, 6, 0.5), (2, 10, 0.5), (3, 20, 0.3), (7, 15, 0.6), (0, 1, 0.5), (12, 40, 0.25)],
)
def test_lower_tail_matches_direct_summation(c, n, p):
    assert binomial_lower_tail(c, n, p) == pytest.approx(
        exact_lower_tail(c, n, p), rel=1e-12
    )


def test_tails_are_complementary():
    for n, p in [(10, 0.5), (37, 0.2), (200, 0.45)]:
        for m in (1, n // 2, n):
            assert binomial_lower_tail(m - 1, n, p) + binomial_upper_tail(
                m, n, p
            ) == pytest.approx(1.0, rel=1e-12)


def test_lower_tail_boundaries():
    assert binomial_lower_tail(-1, 10, 0.5) == 0.0
    assert binomial_lower_tail(10, 10, 0.5) == 1.0
    assert binomial_lower_tail(99, 10, 0.5) == 1.0
    with pytest.raises(ValueError):
        binomial_lower_tail(1, 10, 1.5)


def test_lower_tail_survives_sizes_that_underflow_the_naive_form():
    # (1 - p) ** n is exactly 0.0 in float64 here, so a version that starts the
    # recurrence at k = 0 returns 0.0 for a tail that is merely small.
    assert (1.0 - 0.5) ** 5000 == 0.0
    value = binomial_lower_tail(2400, 5000, 0.5)
    assert 0.0 < value < 0.03


# --------------------------------------------------------------------------
# critical values
# --------------------------------------------------------------------------


def test_critical_value_matches_exact_two_sided_test():
    for discordant in range(1, 40):
        critical = sign_test_critical_value(discordant, 0.05)
        if critical >= 0:
            assert 2.0 * exact_lower_tail(critical, discordant, 0.5) <= 0.05
        # one step further must break the budget
        assert 2.0 * exact_lower_tail(critical + 1, discordant, 0.5) > 0.05


def test_six_disagreements_is_the_floor_at_five_percent():
    # Agrees with the independently derived 2 ** (1 - d) floor in significance.py.
    assert minimum_discordant_pairs(0.05) == 6
    assert sign_test_critical_value(5, 0.05) == -1
    assert sign_test_critical_value(6, 0.05) == 0
    assert smallest_achievable_p(5) > 0.05
    assert smallest_achievable_p(6) <= 0.05


def test_minimum_discordant_pairs_grows_as_alpha_shrinks():
    assert minimum_discordant_pairs(0.10) <= minimum_discordant_pairs(0.05)
    assert minimum_discordant_pairs(0.05) < minimum_discordant_pairs(0.001)


def test_critical_value_rejects_bad_alpha():
    with pytest.raises(ValueError):
        sign_test_critical_value(10, 0.0)
    with pytest.raises(ValueError):
        sign_test_critical_value(-1, 0.05)


# --------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------


def test_untestable_comparisons_have_zero_power():
    # Fewer than six disagreements: no rejection region exists at all.
    for discordant in range(6):
        assert power_given_discordant(discordant, 1.0, 0.05) == 0.0


def test_power_given_discordant_is_the_exact_tail_mass():
    # d = 6, alpha = 0.05 rejects only on 0 or 6 wins.
    assert power_given_discordant(6, 1.0, 0.05) == pytest.approx(1.0)
    assert power_given_discordant(6, 0.5, 0.05) == pytest.approx(2.0 / 64.0)
    assert power_given_discordant(6, 0.9, 0.05) == pytest.approx(
        0.9 ** 6 + 0.1 ** 6
    )


def test_type_one_error_never_exceeds_alpha():
    """The null must be rejected at most alpha of the time - the core guarantee."""
    for n in (30, 92, 400, 1500):
        for alpha in (0.01, 0.05, 0.10):
            assert sign_test_power(n, 0.12, 0.5, alpha) <= alpha + 1e-12


def test_power_is_symmetric_about_the_null():
    for share in (0.6, 0.75, 0.95):
        assert sign_test_power(300, 0.1, share) == pytest.approx(
            sign_test_power(300, 0.1, 1.0 - share), rel=1e-9
        )


def test_power_rises_with_n():
    previous = -1.0
    for n in range(10, 1500, 10):
        value = sign_test_power(n, 0.12, 0.72)
        assert value >= previous - 1e-12, "power fell at n={}".format(n)
        previous = value


def test_power_rises_with_effect_size_and_with_discordance():
    shares = [0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    values = [sign_test_power(300, 0.1, share) for share in shares]
    assert values == sorted(values)
    rates = [0.05, 0.1, 0.2, 0.4]
    by_rate = [sign_test_power(300, rate, 0.7) for rate in rates]
    assert by_rate == sorted(by_rate)


def test_power_of_an_empty_or_tie_only_set_is_zero():
    assert sign_test_power(0, 0.5, 0.9) == 0.0
    assert sign_test_power(100, 0.0, 0.9) == 0.0


def test_power_rejects_bad_input():
    with pytest.raises(ValueError):
        sign_test_power(-1, 0.1, 0.7)
    with pytest.raises(ValueError):
        sign_test_power(100, 1.4, 0.7)
    with pytest.raises(ValueError):
        power_given_discordant(10, 1.7)


# --------------------------------------------------------------------------
# required sample size
# --------------------------------------------------------------------------


def test_required_questions_is_the_smallest_sufficient_n():
    for rate, share in [(0.12, 0.72), (0.10, 0.65), (0.30, 0.60)]:
        needed = required_questions(rate, share)
        assert needed is not None
        assert sign_test_power(needed, rate, share) >= 0.80
        assert sign_test_power(needed - 1, rate, share) < 0.80


def test_required_questions_grows_as_the_effect_shrinks():
    sizes = [required_questions(0.15, share) for share in (0.9, 0.8, 0.7, 0.6)]
    assert all(size is not None for size in sizes)
    assert sizes == sorted(sizes)


def test_required_questions_grows_as_questions_get_less_discriminative():
    sizes = [required_questions(rate, 0.7) for rate in (0.4, 0.2, 0.1, 0.05)]
    assert all(size is not None for size in sizes)
    assert sizes == sorted(sizes)


def test_a_null_effect_needs_infinitely_many_questions():
    assert required_questions(0.2, 0.5) is None
    assert required_questions(0.0, 0.9) is None


def test_unreachable_targets_return_none_rather_than_the_cap():
    assert required_questions(0.01, 0.51, max_questions=500) is None


def test_higher_target_power_costs_more_questions():
    modest = required_questions(0.15, 0.7, target_power=0.5)
    strict = required_questions(0.15, 0.7, target_power=0.95)
    assert modest is not None and strict is not None
    assert modest < strict


def test_required_questions_rejects_bad_input():
    with pytest.raises(ValueError):
        required_questions(0.1, 0.7, target_power=1.0)
    with pytest.raises(ValueError):
        required_questions(0.1, 0.7, max_questions=0)


# --------------------------------------------------------------------------
# smallest detectable effect, and the grid
# --------------------------------------------------------------------------


def test_smallest_detectable_effect_is_consistent_with_power():
    share = smallest_detectable_win_share(400, 0.15)
    assert share is not None
    assert sign_test_power(400, 0.15, share) >= 0.80
    assert sign_test_power(400, 0.15, share - 0.01) < 0.80


def test_a_bigger_set_detects_a_smaller_effect():
    small = smallest_detectable_win_share(200, 0.15)
    large = smallest_detectable_win_share(2000, 0.15)
    assert small is not None and large is not None
    assert large < small


def test_the_current_gold_set_can_only_see_a_rout():
    """The finding the README reports, pinned so it cannot drift silently."""
    share = smallest_detectable_win_share(92, 8.0 / 92.0)
    assert share is not None
    assert share > 0.95


def test_tiny_sets_cannot_detect_anything():
    assert smallest_detectable_win_share(5, 0.1) is None
    assert smallest_detectable_win_share(100, 0.0) is None


def test_requirement_grid_shape_and_ordering():
    grid = requirement_grid((0.1, 0.2), (0.7, 0.9))
    assert set(grid) == {(0.1, 0.7), (0.1, 0.9), (0.2, 0.7), (0.2, 0.9)}
    assert grid[(0.1, 0.7)] > grid[(0.2, 0.7)]
    assert grid[(0.1, 0.7)] > grid[(0.1, 0.9)]


# --------------------------------------------------------------------------
# plug-in rates
# --------------------------------------------------------------------------


def test_rates_from_counts():
    assert rates_from_counts(3, 5, 92) == (8 / 92.0, 3 / 8.0)
    assert rates_from_counts(0, 0, 50) == (0.0, 0.5)
    with pytest.raises(ValueError):
        rates_from_counts(10, 10, 5)
    with pytest.raises(ValueError):
        rates_from_counts(1, 1, 0)
    with pytest.raises(ValueError):
        rates_from_counts(-1, 1, 10)


def test_plugin_rates_feed_the_sizing_functions():
    rate, share = rates_from_counts(30, 10, 100)
    needed = required_questions(rate, share)
    assert needed is not None
    assert sign_test_power(needed, rate, share) >= 0.80


#: The three claims from ``evals/significance.py``, with the win/loss/tie counts
#: they produce on ndcg@5 over the whole 92-question gold set, and the verdict
#: that harness reaches independently.
OBSERVED_CLAIMS = (
    ("ngram vs hybrid3", 22, 12, 58, False),
    ("hybrid3 vs hybrid2", 16, 2, 74, True),
    ("hybrid2 vs bm25", 3, 5, 84, False),
)


@pytest.mark.parametrize("label,wins,losses,ties,was_shown", OBSERVED_CLAIMS)
def test_sizing_predicts_which_claims_this_gold_set_could_establish(
    label, wins, losses, ties, was_shown
):
    """The cross-check that makes this module more than plausible arithmetic.

    Required-sample-size is derived here from the binomial alone, with no
    knowledge of what the significance harness concluded. Comparing the two
    is therefore a real test: a comparison needing fewer questions than the
    gold set holds should be the one that came back SUPPORTED, and the ones
    needing more should be the ones that came back NOT SHOWN. All three agree,
    and a future change that breaks the agreement is either a bug here or a
    genuine shift worth noticing.
    """
    n = wins + losses + ties
    assert n == 92
    rate, share = rates_from_counts(wins, losses, n)
    needed = required_questions(rate, share)
    assert needed is not None
    if was_shown:
        assert needed <= n, "{}: needs {} questions but was shown on {}".format(
            label, needed, n
        )
    else:
        assert needed > n, "{}: needs only {} questions yet was not shown".format(
            label, needed
        )


# --------------------------------------------------------------------------
# the eval script
# --------------------------------------------------------------------------


def test_power_eval_script_runs():
    spec = importlib.util.spec_from_file_location(
        "power_eval", os.path.join(ROOT, "evals", "power.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--max-questions", "2000"]) == 0
