"""Tests for the corpus-versus-questions lever measurement.

Two instruments are being pinned here and they fail in different ways.

The **interval and the paired test** are arithmetic with known answers, so they
are checked against values that can be derived by hand: a Clopper-Pearson bound
at ``successes = 0`` is exactly ``1 - (alpha/2) ** (1/n)``, and an exact McNemar
p-value on ``d`` one-sided switchers is exactly ``2 ** (1 - d)``. Both are the
kind of number nobody can eyeball - "the rate is 14.3% [1.8%, 42.8%]" looks
equally convincing whether or not the tails underneath it are right.

The **partition** fails more quietly. Its only job is to decide which questions
predate an increment, and getting that wrong does not raise: it silently scores
a question against a corpus that is missing part of its answer key, which shows
up as a retrieval "miss" attributed to the method rather than to the setup. The
multi-document case is the one that matters, so it is tested explicitly.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from ragkb.power import binomial_lower_tail, clopper_pearson, mcnemar_exact

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_lever():
    spec = importlib.util.spec_from_file_location(
        "corpus_lever_eval", os.path.join(ROOT, "evals", "corpus_lever.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lever():
    return load_lever()


# --------------------------------------------------------------------------
# clopper_pearson
# --------------------------------------------------------------------------


def test_zero_successes_pins_the_lower_bound_at_zero():
    low, high = clopper_pearson(0, 10)
    assert low == 0.0
    # The closed form for the upper bound when nothing was observed.
    assert high == pytest.approx(1.0 - 0.025 ** (1.0 / 10), abs=1e-9)


def test_all_successes_pins_the_upper_bound_at_one():
    low, high = clopper_pearson(10, 10)
    assert high == 1.0
    assert low == pytest.approx(0.025 ** (1.0 / 10), abs=1e-9)


def test_interval_matches_the_textbook_value():
    # 3 of 10 is the worked example in every treatment of this interval.
    low, high = clopper_pearson(3, 10)
    assert low == pytest.approx(0.0667, abs=5e-4)
    assert high == pytest.approx(0.6525, abs=5e-4)


def test_interval_reproduces_the_readme_figure():
    """The README quotes 2-of-14 as "about 2% and 43%" in prose."""
    low, high = clopper_pearson(2, 14)
    assert low == pytest.approx(0.018, abs=2e-3)
    assert high == pytest.approx(0.428, abs=2e-3)


def test_interval_contains_the_point_estimate():
    for trials in (5, 14, 106):
        for successes in range(trials + 1):
            low, high = clopper_pearson(successes, trials)
            assert low <= successes / float(trials) <= high


def test_interval_is_symmetric_under_relabelling():
    """Counting failures instead of successes must mirror the interval."""
    for successes in range(0, 15):
        low, high = clopper_pearson(successes, 14)
        mirror_low, mirror_high = clopper_pearson(14 - successes, 14)
        assert low == pytest.approx(1.0 - mirror_high, abs=1e-9)
        assert high == pytest.approx(1.0 - mirror_low, abs=1e-9)


def test_interval_narrows_as_evidence_accumulates():
    widths = []
    for trials in (10, 40, 160, 640):
        low, high = clopper_pearson(trials // 4, trials)
        widths.append(high - low)
    assert widths == sorted(widths, reverse=True)


def test_bounds_invert_the_exact_binomial_test():
    """The defining property, checked directly rather than assumed."""
    successes, trials = 4, 20
    low, high = clopper_pearson(successes, trials, confidence=0.95)
    assert binomial_lower_tail(successes, trials, high) == pytest.approx(0.025, abs=1e-6)
    # P(X >= successes) at the lower bound, via the reflection the module uses.
    assert binomial_lower_tail(
        trials - successes, trials, 1.0 - low
    ) == pytest.approx(0.025, abs=1e-6)


def test_higher_confidence_gives_a_wider_interval():
    narrow = clopper_pearson(5, 20, confidence=0.80)
    wide = clopper_pearson(5, 20, confidence=0.99)
    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


@pytest.mark.parametrize(
    "successes,trials,confidence",
    [(-1, 10, 0.95), (11, 10, 0.95), (1, 0, 0.95), (1, 10, 0.0), (1, 10, 1.0)],
)
def test_interval_rejects_impossible_arguments(successes, trials, confidence):
    with pytest.raises(ValueError):
        clopper_pearson(successes, trials, confidence)


# --------------------------------------------------------------------------
# mcnemar_exact
# --------------------------------------------------------------------------


def test_no_switchers_is_no_evidence():
    assert mcnemar_exact(0, 0) == 1.0


def test_a_perfectly_one_sided_result_hits_the_floor():
    """With every switcher going one way the p-value is exactly 2 ** (1 - d)."""
    for switchers in range(1, 12):
        assert mcnemar_exact(switchers, 0) == pytest.approx(2.0 ** (1 - switchers))
        assert mcnemar_exact(0, switchers) == pytest.approx(2.0 ** (1 - switchers))


def test_an_even_split_is_maximally_unconvincing():
    for half in (1, 3, 10):
        assert mcnemar_exact(half, half) == 1.0


def test_direction_does_not_change_the_p_value():
    assert mcnemar_exact(8, 2) == pytest.approx(mcnemar_exact(2, 8))


def test_matches_a_hand_computed_tail():
    # 2 * P(X <= 2) for X ~ Binomial(10, 0.5) = 2 * (1 + 10 + 45) / 1024.
    assert mcnemar_exact(8, 2) == pytest.approx(2.0 * 56 / 1024)


def test_p_value_falls_as_the_imbalance_grows():
    values = [mcnemar_exact(gained, 12 - gained) for gained in (6, 7, 8, 9, 10)]
    assert values == sorted(values, reverse=True)


def test_p_values_stay_in_range():
    for gained in range(0, 9):
        for lost in range(0, 9):
            assert 0.0 <= mcnemar_exact(gained, lost) <= 1.0


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


# --------------------------------------------------------------------------
# the partition and the flags
# --------------------------------------------------------------------------


def _record(ident, relevant):
    return {"id": ident, "question": ident, "relevant": relevant, "difficulty": "easy"}


def test_partition_splits_on_the_added_documents(lever):
    gold = [_record("a", ["old"]), _record("b", ["new"])]
    carried, fresh = lever.partition_gold(gold, ["new"])
    assert [r["id"] for r in carried] == ["a"]
    assert [r["id"] for r in fresh] == ["b"]


def test_a_question_touching_one_new_document_is_not_carried(lever):
    """The case that would otherwise be scored against a missing answer key."""
    gold = [_record("mixed", ["old", "new"])]
    carried, fresh = lever.partition_gold(gold, ["new"])
    assert carried == []
    assert [r["id"] for r in fresh] == ["mixed"]


def test_partition_preserves_every_question_exactly_once(lever):
    gold = [_record(str(i), ["old" if i % 2 else "new"]) for i in range(10)]
    carried, fresh = lever.partition_gold(gold, ["new"])
    assert len(carried) + len(fresh) == len(gold)
    assert {r["id"] for r in carried} | {r["id"] for r in fresh} == {
        r["id"] for r in gold
    }


def test_partition_with_nothing_added_carries_everything(lever):
    gold = [_record("a", ["old"]), _record("b", ["older"])]
    carried, fresh = lever.partition_gold(gold, [])
    assert len(carried) == 2
    assert fresh == []


def test_discordance_flags_mark_only_disagreements(lever):
    assert lever.discordance_flags([1.0, 0.0, 0.5], [1.0, 1.0, 0.25]) == [
        False,
        True,
        True,
    ]


def test_discordance_flag_count_matches_the_comparison(lever):
    """The flags and the W/L/T summary must agree on the sample size."""
    from ragkb.significance import compare

    left = [1.0, 0.0, 1.0, 0.0, 1.0]
    right = [0.0, 0.0, 1.0, 1.0, 1.0]
    result = compare(left, right, bootstrap_resamples=50, permutations=50)
    assert sum(lever.discordance_flags(left, right)) == result.discordant


# --------------------------------------------------------------------------
# the script
# --------------------------------------------------------------------------


def test_lever_script_runs_end_to_end(lever):
    assert (
        lever.main(
            [
                "--pair",
                "hybrid2,bm25",
                "--bootstrap-resamples",
                "50",
                "--permutations",
                "50",
            ]
        )
        == 0
    )


def test_exclude_rolls_the_corpus_back(lever):
    """Re-measuring an earlier increment is the point of ``--exclude``."""
    assert (
        lever.main(
            [
                "--exclude",
                "relevance-feedback,score-normalisation",
                "--added",
                "query-performance-prediction,passage-aggregation",
                "--bootstrap-resamples",
                "50",
                "--permutations",
                "50",
            ]
        )
        == 0
    )


def test_unknown_method_is_rejected(lever):
    with pytest.raises(SystemExit):
        lever.main(["--pair", "hybrid2,nosuchmethod"])


def test_unknown_document_is_rejected(lever):
    with pytest.raises(SystemExit):
        lever.main(["--added", "not-a-document"])


def test_a_document_cannot_be_added_and_excluded(lever):
    with pytest.raises(SystemExit):
        lever.main(["--added", "bm25", "--exclude", "bm25"])
