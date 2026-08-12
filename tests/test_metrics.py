import math

import pytest

from ragkb.metrics import (
    aggregate,
    hit_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d"]


def test_hit_at_k():
    assert hit_at_k(RANKED, ["c"], 3) == 1.0
    assert hit_at_k(RANKED, ["c"], 2) == 0.0


def test_recall_at_k():
    assert recall_at_k(RANKED, ["a", "c"], 3) == 1.0
    assert recall_at_k(RANKED, ["a", "c"], 2) == 0.5
    assert recall_at_k(RANKED, [], 3) == 0.0


def test_precision_at_k():
    assert precision_at_k(RANKED, ["a", "b"], 2) == 1.0
    assert precision_at_k(RANKED, ["a"], 4) == 0.25
    assert precision_at_k(RANKED, ["a"], 0) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "z", "z2"], ["a", "b"], 4) == pytest.approx(1.0)


def test_ndcg_discounts_lower_ranks():
    # relevant item at rank 2 instead of rank 1
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(expected)


def test_ndcg_sees_the_second_relevant_document():
    # MRR is identical for both rankings; nDCG must not be.
    close = ndcg_at_k(["a", "b", "x", "y"], ["a", "b"], 4)
    far = ndcg_at_k(["a", "x", "y", "b"], ["a", "b"], 4)
    assert reciprocal_rank(["a", "b", "x", "y"], ["a", "b"]) == reciprocal_rank(
        ["a", "x", "y", "b"], ["a", "b"]
    )
    assert close > far > 0.0


def test_ndcg_credits_duplicates_once():
    # Three chunks of the same relevant document must not beat two distinct
    # relevant documents.
    duplicated = ndcg_at_k(["a", "a", "a"], ["a", "b"], 3)
    distinct = ndcg_at_k(["a", "b", "x"], ["a", "b"], 3)
    assert distinct > duplicated
    # ...and the duplicate itself adds nothing beyond the first occurrence.
    assert duplicated == ndcg_at_k(["a", "x", "y"], ["a", "b"], 3)


def test_ndcg_ideal_is_capped_at_k():
    # With more relevant items than slots, retrieving k of them perfectly
    # is still a perfect score.
    assert ndcg_at_k(["a", "b"], ["a", "b", "c"], 2) == pytest.approx(1.0)


def test_ndcg_graded_gains_prefer_the_higher_gain_first():
    gains = {"a": 3.0, "b": 1.0}
    best_first = ndcg_at_k(["a", "b"], ["a", "b"], 2, gains=gains)
    worst_first = ndcg_at_k(["b", "a"], ["a", "b"], 2, gains=gains)
    assert best_first == pytest.approx(1.0)
    assert worst_first < best_first


def test_ndcg_edge_cases():
    assert ndcg_at_k(RANKED, [], 3) == 0.0
    assert ndcg_at_k(RANKED, ["z"], 3) == 0.0
    assert ndcg_at_k(RANKED, ["a"], 0) == 0.0
    assert ndcg_at_k([], ["a"], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, ["b"]) == 0.5
    assert reciprocal_rank(RANKED, ["z"]) == 0.0


def test_aggregate_reports_every_metric():
    report = aggregate([RANKED, ["z", "a"]], [["a"], ["a"]], ks=(1, 3))
    assert report["hit@1"] == 0.5
    assert report["hit@3"] == 1.0
    assert report["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert report["ndcg@1"] == 0.5
    assert report["ndcg@3"] == pytest.approx((1.0 + 1.0 / math.log2(3)) / 2)


def test_aggregate_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        aggregate([RANKED], [["a"], ["b"]])


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
