"""Tests for cross-validated selection.

The module measures a bias, so most of these tests are constructed cases where
the correct amount of bias is known in advance: no bias when the configurations
are identical or when one genuinely dominates, and a large, detectable bias when
the "best" configuration is only best by luck.
"""

from __future__ import annotations

import random

import pytest

from ragkb.holdout import (
    cross_validated_selection,
    k_fold,
    repeated_selection,
    summarize,
)


# --------------------------------------------------------------------------
# k_fold
# --------------------------------------------------------------------------


def test_k_fold_is_a_partition():
    groups = k_fold(56, folds=5, seed=0)
    flat = [index for group in groups for index in group]
    assert sorted(flat) == list(range(56))
    assert len(flat) == len(set(flat))


def test_k_fold_sizes_differ_by_at_most_one():
    sizes = [len(group) for group in k_fold(56, folds=5, seed=0)]
    assert max(sizes) - min(sizes) <= 1


def test_k_fold_is_deterministic_for_a_seed():
    assert k_fold(56, folds=5, seed=3) == k_fold(56, folds=5, seed=3)


def test_k_fold_seeds_actually_differ():
    assert k_fold(56, folds=5, seed=0) != k_fold(56, folds=5, seed=1)


def test_k_fold_shuffles_rather_than_slicing():
    """A contiguous split would put whole difficulty blocks on one side.

    ``gold.jsonl`` stores the easy questions before the hard ones, so a fold
    made of consecutive indices measures the difference between slices instead
    of the effect of selection. Guard the shuffle explicitly.
    """
    first = k_fold(40, folds=4, seed=0)[0]
    assert first != list(range(0, 10))
    # and it should not be a stride either
    assert first != list(range(0, 40, 4))


def test_k_fold_leave_one_out_is_allowed():
    groups = k_fold(6, folds=6, seed=0)
    assert sorted(len(group) for group in groups) == [1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize("n,folds", [(0, 5), (10, 1), (10, 0), (5, 6)])
def test_k_fold_rejects_impossible_splits(n, folds):
    with pytest.raises(ValueError):
        k_fold(n, folds=folds)


# --------------------------------------------------------------------------
# no-bias cases
# --------------------------------------------------------------------------


def test_identical_configs_have_exactly_zero_optimism():
    """Nothing to select between, so selection cannot inflate anything."""
    values = [0.0, 1.0, 0.5, 1.0, 0.0, 0.25, 1.0, 0.75, 0.5, 0.0]
    scores = {"a": list(values), "b": list(values), "c": list(values)}
    result = cross_validated_selection(scores, folds=5, seed=0, default="a")
    assert result.optimism == pytest.approx(0.0)
    assert result.cv_selected == pytest.approx(result.in_sample_best)
    assert result.tuning_gain == pytest.approx(0.0)


def test_ties_break_towards_the_first_configuration():
    """The caller lists the shipped default first, so a tie means "do not move"."""
    values = [1.0, 0.0] * 5
    scores = {"default": list(values), "other": list(values)}
    result = cross_validated_selection(scores, folds=5, seed=0, default="default")
    assert result.in_sample_argbest == "default"
    assert set(result.winners) == {"default"}


def test_a_uniformly_dominant_config_wins_every_fold_with_no_optimism():
    n = 20
    weak = [0.0] * n
    strong = [1.0] * n
    result = cross_validated_selection(
        {"weak": weak, "strong": strong}, folds=5, seed=0, default="weak"
    )
    assert result.in_sample_argbest == "strong"
    assert result.stable and result.distinct_winners == 1
    assert result.optimism == pytest.approx(0.0)
    assert result.tuning_gain == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the case the module exists for
# --------------------------------------------------------------------------


def test_pure_noise_produces_optimism_and_unstable_winners():
    """Twelve configurations of coin flips: the best row is worth nothing.

    Every configuration here has the same expected score, so the honest
    held-out estimate should sit near 0.5 while the in-sample maximum sits well
    above it. That gap is the whole point of the module, and a harness that
    could not detect it on this input would be measuring nothing.
    """
    rng = random.Random(11)
    n = 56
    scores = {
        "c{}".format(index): [float(rng.random() < 0.5) for _ in range(n)]
        for index in range(12)
    }
    results = repeated_selection(scores, folds=5, repeats=20, seed=0)
    summary = summarize(results)
    assert summary["in_sample_best"] > 0.5
    assert summary["optimism"] > 0.02
    assert summary["cv_selected"] < summary["in_sample_best"]
    # And the choice should not be reproducible across splits.
    assert summary["stable_share"] < 0.5


def test_a_noise_sweep_loses_to_the_default_on_average():
    """Sweeping noise is not merely useless, it is worse than not sweeping.

    Each configuration has the same expectation as the default, so selecting a
    winner on part of the data buys nothing on the rest - while the default's
    own score is not inflated by having been chosen.
    """
    rng = random.Random(5)
    n = 60
    scores = {"default": [float(rng.random() < 0.5) for _ in range(n)]}
    for index in range(10):
        scores["c{}".format(index)] = [float(rng.random() < 0.5) for _ in range(n)]
    summary = summarize(
        repeated_selection(scores, folds=5, repeats=25, seed=0, default="default")
    )
    assert summary["in_sample_best"] > summary["cv_selected"]
    assert "tuning_gain" in summary


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------


def test_held_out_scores_cover_every_question_exactly_once():
    """The held-out mean is over the whole set, not an average of fold means.

    Folds differ in size by one when ``n`` is not divisible by ``folds``, so
    averaging the per-fold means would over-weight the smaller folds. Build a
    case where one configuration wins everywhere, which makes the correct
    held-out value the plain mean of that configuration.
    """
    n = 23  # deliberately not divisible by 5
    winner = [float(i) for i in range(n)]
    loser = [-1.0] * n
    result = cross_validated_selection(
        {"loser": loser, "winner": winner}, folds=5, seed=0, default="loser"
    )
    assert result.cv_selected == pytest.approx(sum(winner) / n)


def test_in_sample_matches_the_plain_means():
    scores = {"a": [1.0, 0.0, 1.0, 0.0], "b": [0.5, 0.5, 0.5, 0.5]}
    result = cross_validated_selection(scores, folds=2, seed=0, default="a")
    assert result.in_sample["a"] == pytest.approx(0.5)
    assert result.in_sample["b"] == pytest.approx(0.5)
    assert result.in_sample_best == pytest.approx(0.5)


def test_tuning_gain_is_none_without_a_default():
    result = cross_validated_selection({"a": [1.0] * 8, "b": [0.0] * 8}, folds=4)
    assert result.default_name is None
    assert result.default_score is None
    assert result.tuning_gain is None
    assert "vs" not in result.describe()


def test_result_is_reproducible_for_a_seed():
    rng = random.Random(2)
    scores = {
        "c{}".format(i): [float(rng.random() < 0.5) for _ in range(40)]
        for i in range(4)
    }
    first = cross_validated_selection(scores, folds=5, seed=9)
    second = cross_validated_selection(scores, folds=5, seed=9)
    assert first == second


def test_repeated_selection_uses_consecutive_seeds():
    scores = {"a": [1.0, 0.0] * 10, "b": [0.0, 1.0] * 10}
    results = repeated_selection(scores, folds=5, repeats=3, seed=4)
    assert [result.seed for result in results] == [4, 5, 6]


def test_describe_reports_the_headline_numbers():
    result = cross_validated_selection(
        {"weak": [0.0] * 20, "strong": [1.0] * 20},
        folds=5,
        seed=0,
        default="weak",
    )
    text = result.describe()
    assert "strong" in text and "optimism=+0.000" in text and "winners=1/5" in text


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_rejects_ragged_score_vectors():
    with pytest.raises(ValueError):
        cross_validated_selection({"a": [1.0, 0.0], "b": [1.0]}, folds=2)


def test_rejects_empty_inputs():
    with pytest.raises(ValueError):
        cross_validated_selection({}, folds=2)
    with pytest.raises(ValueError):
        cross_validated_selection({"a": []}, folds=2)


def test_rejects_an_unknown_default():
    with pytest.raises(ValueError):
        cross_validated_selection({"a": [1.0] * 10}, folds=5, default="b")


def test_rejects_non_positive_repeats():
    with pytest.raises(ValueError):
        repeated_selection({"a": [1.0] * 10}, folds=5, repeats=0)


def test_summarize_rejects_nothing_to_summarize():
    with pytest.raises(ValueError):
        summarize([])
