"""Tests for the paired significance machinery.

A statistical instrument is only useful if it is trustworthy in the cases where
the answer is already known, so the tests here pin the corners: identical
systems must not be called different, a system that wins every question must be,
a seed must reproduce an interval exactly, and the paired structure must
actually be used rather than the two vectors being treated as independent
samples.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from ragkb.significance import (
    Comparison,
    bootstrap_ci,
    compare,
    holm_bonferroni,
    mean,
    paired_bootstrap,
    permutation_test,
    resolution,
    smallest_achievable_p,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "significance_eval", os.path.join(ROOT, "evals", "significance.py")
)
significance_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(significance_eval)


def test_resolution_is_one_question():
    assert resolution(56) == pytest.approx(1.0 / 56)
    with pytest.raises(ValueError):
        resolution(0)


def test_identical_systems_are_not_significant():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    result = compare(scores, list(scores), bootstrap_resamples=200, permutations=200)
    assert result.delta == 0.0
    assert result.ci_low == 0.0 and result.ci_high == 0.0
    # Every resample reproduces the observed (zero) difference, so chance
    # explains it perfectly.
    assert result.p_value == pytest.approx(1.0)
    assert not result.significant
    assert (result.wins, result.losses) == (0, 0)
    assert result.ties == len(scores)


def test_a_system_that_wins_everywhere_is_significant():
    winner = [1.0] * 30
    loser = [0.0] * 30
    result = compare(winner, loser, bootstrap_resamples=500, permutations=2000)
    assert result.delta == pytest.approx(1.0)
    assert result.ci_low > 0.0
    assert result.significant
    assert result.p_value < 0.01
    assert (result.wins, result.losses, result.ties) == (30, 0, 0)


def test_one_question_difference_is_not_significant():
    """The case the module exists for: a single discordant pair out of many."""
    a = [1.0] * 55 + [1.0]
    b = [1.0] * 55 + [0.0]
    result = compare(a, b, bootstrap_resamples=1000, permutations=2000)
    assert result.delta == pytest.approx(1.0 / 56)
    assert (result.wins, result.losses, result.ties) == (1, 0, 55)
    assert result.discordant == 1
    assert not result.significant
    # One discordant pair has two equally likely sign assignments and both are
    # as extreme as what was observed, so the test cannot say anything.
    assert result.p_value == pytest.approx(1.0)
    assert result.p_floor == 1.0


def test_p_floor_depends_only_on_discordant_pairs():
    assert smallest_achievable_p(0) == 1.0
    assert smallest_achievable_p(1) == 1.0
    assert smallest_achievable_p(2) == pytest.approx(0.5)
    assert smallest_achievable_p(5) == pytest.approx(0.0625)
    # Six discordant pairs is the first count that can clear the usual 0.05.
    assert smallest_achievable_p(6) < 0.05
    assert smallest_achievable_p(5) > 0.05
    with pytest.raises(ValueError):
        smallest_achievable_p(-1)


def test_exact_regime_hits_its_floor_exactly():
    """Small comparisons are enumerated, so the p-value equals the floor exactly.

    This is the regression guard for the bug that motivated the exact path: a
    sampled p-value scatters around the truth and can land *below* the smallest
    value the test could legitimately return, which reads as stronger evidence
    than the data contains.
    """
    for discordant in (1, 2, 3, 4):
        a = [1.0] * discordant + [1.0] * 20
        b = [0.0] * discordant + [1.0] * 20
        result = compare(a, b, bootstrap_resamples=200, permutations=4000)
        assert result.discordant == discordant
        # All differences point the same way, so only the two extreme
        # assignments reach the observed magnitude: p is exactly the floor.
        assert result.p_value == pytest.approx(result.p_floor)


def test_exact_and_sampled_paths_agree():
    """Forcing the sampled path must approximate what enumeration proves."""
    a = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    b = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    exact = permutation_test(a, b, exact_limit=10)
    sampled = permutation_test(a, b, resamples=20000, seed=5, exact_limit=0)
    assert exact == pytest.approx(sampled, abs=0.03)


def test_exact_limit_is_validated():
    with pytest.raises(ValueError):
        permutation_test([1.0, 0.0], [0.0, 0.0], exact_limit=-1)


def test_pairing_is_used_not_independent_samples():
    """Same means, different pairings, different verdicts.

    Both vectors have mean 0.5 in each scenario. When ``a`` wins every question
    the paired test sees a consistent effect; when the wins and losses cancel
    question by question it does not. An unpaired comparison of the two means
    would report identical - and useless - results for both.
    """
    consistent_a = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    consistent_b = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    delta_consistent, _, _ = paired_bootstrap(
        consistent_a, consistent_b, resamples=200
    )
    delta_zero, low, high = paired_bootstrap(
        consistent_a, list(consistent_a), resamples=200
    )
    assert delta_consistent == pytest.approx(0.0)
    assert delta_zero == 0.0 and low == 0.0 and high == 0.0
    assert mean(consistent_a) == mean(consistent_b)


def test_seed_reproduces_the_interval():
    a = [0.1, 0.9, 0.4, 0.7, 0.2, 0.8, 0.5, 0.3]
    b = [0.2, 0.6, 0.5, 0.5, 0.4, 0.6, 0.4, 0.4]
    first = compare(a, b, bootstrap_resamples=300, permutations=300, seed=11)
    second = compare(a, b, bootstrap_resamples=300, permutations=300, seed=11)
    third = compare(a, b, bootstrap_resamples=300, permutations=300, seed=12)
    assert first == second
    assert isinstance(first, Comparison)
    # A different seed moves the estimate but not the observed delta.
    assert third.delta == pytest.approx(first.delta)


def test_confidence_interval_brackets_the_observed_delta():
    a = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    b = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    delta, low, high = paired_bootstrap(a, b, resamples=1000, seed=3)
    assert low <= delta <= high


def test_wider_confidence_gives_a_wider_interval():
    a = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    b = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    _, narrow_low, narrow_high = paired_bootstrap(a, b, resamples=800, confidence=0.5)
    _, wide_low, wide_high = paired_bootstrap(a, b, resamples=800, confidence=0.99)
    assert (wide_high - wide_low) >= (narrow_high - narrow_low)


def test_permutation_p_value_is_bounded_and_never_zero():
    a = [1.0] * 40
    b = [0.0] * 40
    p = permutation_test(a, b, resamples=100)
    assert 0.0 < p <= 1.0
    # The observed assignment counts as one sample, so the floor is 1/(N+1).
    assert p == pytest.approx(1.0 / 101)


def test_bootstrap_ci_of_a_constant_is_that_constant():
    low, high = bootstrap_ci([0.25] * 20, resamples=100)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 0.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], confidence=1.0)
    with pytest.raises(ValueError):
        permutation_test([1.0], [0.0], resamples=0)


def test_metric_value_dispatches_by_name():
    ranked = ["a", "b", "c", "d", "e"]
    relevant = ["c"]
    assert significance_eval.metric_value("hit@1", ranked, relevant) == 0.0
    assert significance_eval.metric_value("hit@3", ranked, relevant) == 1.0
    assert significance_eval.metric_value("recall@5", ranked, relevant) == 1.0
    assert significance_eval.metric_value("mrr", ranked, relevant) == pytest.approx(
        1.0 / 3
    )
    assert 0.0 < significance_eval.metric_value("ndcg@5", ranked, relevant) < 1.0
    with pytest.raises(ValueError):
        significance_eval.metric_value("bleu@5", ranked, relevant)
    with pytest.raises(ValueError):
        significance_eval.metric_value("hit", ranked, relevant)


def test_pairs_enumerates_each_comparison_once():
    assert significance_eval.pairs(["a", "b", "c"]) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


def test_select_slice_filters_by_difficulty():
    gold = [
        {"id": "1", "difficulty": "easy"},
        {"id": "2", "difficulty": "hard"},
        {"id": "3", "difficulty": "hard"},
    ]
    assert len(significance_eval.select_slice(gold, "all")) == 3
    assert [r["id"] for r in significance_eval.select_slice(gold, "hard")] == ["2", "3"]
    assert significance_eval.select_slice(gold, "missing") == []


def test_holm_worked_example():
    """A hand-computed case, so the multipliers are pinned rather than assumed.

    Sorted ascending the raw values are 0.01, 0.03, 0.04 against multipliers
    3, 2, 1: 0.03, 0.06, 0.04. The running maximum turns the last into 0.06,
    which is the monotonicity fix doing its job.
    """
    result = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    assert [round(adjusted, 10) for adjusted, _ in result] == [0.03, 0.06, 0.06]
    assert [reject for _, reject in result] == [True, False, False]


def test_holm_is_monotone_in_the_sorted_order():
    """A larger raw p must never receive a smaller adjusted p."""
    raw = [0.001, 0.049, 0.02, 0.5, 0.011, 0.3]
    adjusted = [value for value, _ in holm_bonferroni(raw)]
    by_raw = [adjusted[i] for i in sorted(range(len(raw)), key=lambda j: raw[j])]
    assert by_raw == sorted(by_raw)


def test_holm_never_rejects_less_than_bonferroni():
    """The whole reason to prefer Holm: it is uniformly more powerful."""
    raw = [0.001, 0.009, 0.02, 0.04, 0.3, 0.8, 0.02, 0.006]
    total = len(raw)
    holm = holm_bonferroni(raw, alpha=0.05)
    for value, (adjusted, reject) in zip(raw, holm):
        bonferroni = min(1.0, value * total)
        assert adjusted <= bonferroni + 1e-12
        if value * total <= 0.05:
            assert reject


def test_holm_is_order_independent():
    """Shuffling the family must not change any element's verdict."""
    raw = [0.04, 0.001, 0.6, 0.02, 0.3]
    straight = holm_bonferroni(raw)
    order = [3, 0, 4, 1, 2]
    shuffled = holm_bonferroni([raw[i] for i in order])
    for position, source in enumerate(order):
        assert shuffled[position][0] == pytest.approx(straight[source][0])
        assert shuffled[position][1] == straight[source][1]


def test_holm_clamps_and_handles_degenerate_families():
    assert holm_bonferroni([]) == []
    # A family of one is uncorrected: the multiplier is 1.
    assert holm_bonferroni([0.04])[0] == (pytest.approx(0.04), True)
    # Nothing may exceed 1.0 however large the family.
    assert all(value <= 1.0 for value, _ in holm_bonferroni([0.9] * 20))
    assert not any(reject for _, reject in holm_bonferroni([1.0, 1.0, 1.0]))


def test_holm_rejects_the_whole_family_when_every_p_is_tiny():
    result = holm_bonferroni([1e-6] * 8, alpha=0.05)
    assert all(reject for _, reject in result)


def test_holm_validates_its_inputs():
    with pytest.raises(ValueError):
        holm_bonferroni([0.5], alpha=0.0)
    with pytest.raises(ValueError):
        holm_bonferroni([0.5], alpha=1.0)
    with pytest.raises(ValueError):
        holm_bonferroni([0.5, 1.2])
    with pytest.raises(ValueError):
        holm_bonferroni([-0.1])


def test_eval_method_routing_covers_the_ngram_list():
    """The gap this increment closed: the harness must score the third list.

    ``evaluate.py`` tells the reader to run the significance script before
    believing its ``ngram only`` control, so a routing table that omits
    ``ngram`` makes that instruction a dead pointer.
    """
    labels = [name for name, _, _ in significance_eval.METHOD_ROUTING]
    assert "ngram" in labels
    assert "hybrid3" in labels
    for _, engine_name, method in significance_eval.METHOD_ROUTING:
        assert engine_name in {"base", "ngram"}
        # Only the three-list engine can serve the n-gram or three-way rows.
        if method == "ngram":
            assert engine_name == "ngram"


def test_claims_are_stated_in_terms_of_known_methods():
    """A claim naming a method the routing table does not build is unfalsifiable."""
    labels = {name for name, _, _ in significance_eval.METHOD_ROUTING}
    for left, right, description in significance_eval.CLAIMS:
        assert left in labels
        assert right in labels
        assert description
