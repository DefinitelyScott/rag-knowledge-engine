"""Tests for the leave-one-document-out jackknife.

The instrument's arithmetic has closed forms that can be checked by hand, and
its one structural decision (which questions to score once a document is gone)
fails silently if wrong: a question scored against a missing answer key looks
like a retrieval miss, not a setup error.
"""

from __future__ import annotations

import importlib.util
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@pytest.fixture(scope="module")
def jk():
    spec = importlib.util.spec_from_file_location(
        "jackknife_eval", os.path.join(ROOT, "evals", "jackknife.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOLD = [
    {"id": "a", "relevant": ["x"]},
    {"id": "b", "relevant": ["x", "y"]},
    {"id": "c", "relevant": ["y"]},
    {"id": "d", "relevant": ["z"]},
]


def test_questions_without_drops_every_question_naming_the_document(jk):
    # "b" names x as one of two relevant documents; it must go too, because
    # half its answer key disappears with x.
    assert jk.questions_without(GOLD, "x") == [2, 3]
    assert jk.questions_without(GOLD, "z") == [0, 1, 2]
    assert jk.questions_without(GOLD, "absent") == [0, 1, 2, 3]


def test_influence_is_the_mean_difference_and_counts_changed_questions(jk):
    value, changed = jk.influence([0.5, 0.0, 0.25], [0.5, 0.0, -0.25])
    assert value == pytest.approx(0.5 / 3)
    assert changed == 1


def test_influence_of_an_identical_pair_is_exactly_zero(jk):
    assert jk.influence([0.1, -0.2], [0.1, -0.2]) == (0.0, 0)


def test_influence_rejects_mismatched_or_empty_vectors(jk):
    with pytest.raises(ValueError):
        jk.influence([0.1], [0.1, 0.2])
    with pytest.raises(ValueError):
        jk.influence([], [])


def test_jackknife_se_is_zero_when_no_document_matters(jk):
    assert jk.jackknife_se(0.02, [0.0] * 10) == pytest.approx(0.0, abs=1e-12)


def test_jackknife_se_matches_the_hand_computed_value(jk):
    # leave-one-out values: 0.1-0.01=0.09 and 0.1+0.01=0.11, mean 0.10,
    # sum of squares 2e-4, times (n-1)/n = 1/2 -> 1e-4, root 0.01.
    assert jk.jackknife_se(0.1, [0.01, -0.01]) == pytest.approx(0.01)


def test_jackknife_se_equals_sd_times_n_minus_one_over_root_n(jk):
    values = [0.001 * i for i in range(-5, 6)]
    n = len(values)
    expected = jk.sample_sd(values) * (n - 1) / math.sqrt(n)
    assert jk.jackknife_se(0.02, values) == pytest.approx(expected)


def test_jackknife_needs_at_least_two_documents(jk):
    with pytest.raises(ValueError):
        jk.jackknife_se(0.1, [0.0])


def test_sign_flips_only_when_removal_crosses_zero(jk):
    # delta 0.02: removing a document whose influence is 0.03 leaves -0.01.
    assert jk.sign_flips(0.02, [0.01, 0.03, -0.05, 0.02]) == [1]
    assert jk.sign_flips(0.0, [0.5]) == []
    assert jk.sign_flips(-0.02, [-0.03, 0.01]) == [0]


def test_gap_scores_are_left_minus_right_in_question_order(jk, monkeypatch):
    rankings = {"L": [["x"], ["q"]], "R": [["q"], ["q"]]}
    monkeypatch.setattr(
        jk.significance_eval, "rankings_by_method", lambda *a, **kw: rankings
    )
    gold = [{"relevant": ["x"]}, {"relevant": ["x"]}]
    assert jk.gap_scores(None, None, gold, "L", "R", "hit@1", 1) == [1.0, 0.0]
