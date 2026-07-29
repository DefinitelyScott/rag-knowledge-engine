import pytest

from ragkb.metrics import (
    aggregate,
    hit_at_k,
    mean,
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


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, ["b"]) == 0.5
    assert reciprocal_rank(RANKED, ["z"]) == 0.0


def test_aggregate_reports_every_metric():
    report = aggregate([RANKED, ["z", "a"]], [["a"], ["a"]], ks=(1, 3))
    assert report["hit@1"] == 0.5
    assert report["hit@3"] == 1.0
    assert report["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_aggregate_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        aggregate([RANKED], [["a"], ["b"]])


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
