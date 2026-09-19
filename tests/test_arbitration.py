"""Tests for the fusion-arbitration audit.

Two jobs. The first is ordinary: the reordering classifier, the per-query
scorer and the decomposition have to be right, and the scorer in particular has
to agree with ``ragkb.metrics.aggregate`` rather than merely look like it,
because a private reimplementation that drifts would make every delta in the
report wrong in a way no reader could see.

The second is the one that matters for the README. The whole point of the
script is to distinguish "the fusion returned its input" from "the fusion
reordered and the reorderings cancelled", and the README now asserts the second
for the two-list fusion. That assertion is a measurement, so it is tested here
the same way the other standing claims in this repo are: if the corpus grows
and the two-list fusion really does go inert, this fails and the prose gets
corrected rather than quietly becoming false.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

from ragkb.engine import EngineConfig, RAGEngine
from ragkb.metrics import aggregate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
GOLD = os.path.join(ROOT, "evals", "gold.jsonl")


def _load_module():
    path = os.path.join(ROOT, "evals", "arbitration.py")
    spec = importlib.util.spec_from_file_location("arbitration_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arbitration = _load_module()


@pytest.fixture(scope="module")
def gold():
    with open(GOLD, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def engines():
    shared = dict(target_tokens=120, overlap_tokens=30, rrf_k=60)
    return {
        "base": RAGEngine.from_corpus(CORPUS, config=EngineConfig(**shared)),
        "ngram": RAGEngine.from_corpus(
            CORPUS, config=EngineConfig(ngram=4, **shared)
        ),
    }


# --------------------------------------------------------------------------
# the reordering classifier
# --------------------------------------------------------------------------


def test_identical_rankings_are_same_order():
    assert arbitration.relation(["a", "b"], ["a", "b"]) == arbitration.SAME_ORDER


def test_permuted_rankings_are_same_set():
    assert arbitration.relation(["a", "b"], ["b", "a"]) == arbitration.SAME_SET


def test_differing_membership_is_different_set():
    assert arbitration.relation(["a", "b"], ["a", "c"]) == arbitration.DIFFERENT_SET


def test_a_shorter_ranking_is_a_different_set():
    """Truncation is a membership change, not a reordering."""
    assert arbitration.relation(["a", "b"], ["a"]) == arbitration.DIFFERENT_SET


def test_relation_partitions_every_pair():
    """The three relations are exhaustive and mutually exclusive."""
    cases = [
        (["a"], ["a"]),
        (["a", "b"], ["b", "a"]),
        (["a"], ["b"]),
        ([], []),
    ]
    seen = [arbitration.relation(left, right) for left, right in cases]
    valid = {
        arbitration.SAME_ORDER,
        arbitration.SAME_SET,
        arbitration.DIFFERENT_SET,
    }
    assert set(seen) <= valid
    assert seen[0] == arbitration.SAME_ORDER
    assert seen[3] == arbitration.SAME_ORDER


def test_relation_counts_sum_to_the_question_count():
    fused = [["a", "b"], ["a", "b"], ["c"]]
    other = [["a", "b"], ["b", "a"], ["d"]]
    counts = arbitration.relation_counts(fused, other)
    assert sum(counts.values()) == len(fused)
    assert counts[arbitration.SAME_ORDER] == 1
    assert counts[arbitration.SAME_SET] == 1
    assert counts[arbitration.DIFFERENT_SET] == 1


# --------------------------------------------------------------------------
# the per-query scorer, against the shipped aggregator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["hit@1", "hit@5", "recall@5", "ndcg@5", "mrr"])
def test_score_matches_aggregate_on_one_question(metric, gold, engines):
    """The private scorer must reproduce ``aggregate`` exactly, not roughly.

    ``evals/evaluate.py`` reports means from ``aggregate``; this script reports
    per-query values it computes itself. If the two ever disagree the two
    halves of the README would be describing different systems.
    """
    engine = engines["base"]
    for record in gold[:25]:
        ranking = engine.ranked_doc_ids(record["question"], k=5, method="hybrid")
        expected = aggregate([ranking], [record["relevant"]], ks=(1, 3, 5))[metric]
        assert arbitration.score(metric, ranking, record["relevant"], 5) == pytest.approx(
            expected
        )


def test_score_rejects_an_unknown_metric():
    with pytest.raises(ValueError):
        arbitration.score("map@5", ["a"], ["a"], 5)


# --------------------------------------------------------------------------
# the decomposition
# --------------------------------------------------------------------------


def test_decompose_splits_the_aggregate_into_rate_and_effect():
    fused = [1.0, 0.0, 1.0, 1.0]
    other = [1.0, 0.0, 0.0, 0.0]
    acted = [False, False, True, True]
    aggregate_delta, subset_delta, count, wins, losses = arbitration.decompose(
        fused, other, acted
    )
    assert count == 2
    assert wins == 2 and losses == 0
    assert subset_delta == pytest.approx(1.0)
    assert aggregate_delta == pytest.approx(0.5)
    assert aggregate_delta == pytest.approx((count / len(fused)) * subset_delta)


def test_decompose_handles_an_empty_acted_subset():
    aggregate_delta, subset_delta, count, wins, losses = arbitration.decompose(
        [1.0, 1.0], [1.0, 1.0], [False, False]
    )
    assert (count, wins, losses) == (0, 0, 0)
    assert subset_delta == 0.0
    assert aggregate_delta == pytest.approx(0.0)


def test_check_identity_accepts_a_consistent_decomposition():
    arbitration.check_identity(0.5, 1.0, 2, 4)


def test_check_identity_rejects_an_inconsistent_one():
    """The guard has to actually fire, or it is decoration.

    This is the failure it exists to catch: a question classified as unchanged
    that nonetheless scored differently, which would mean `relation` and the
    scorer disagree about what "the same ranking" is.
    """
    with pytest.raises(AssertionError, match="decomposition broke"):
        arbitration.check_identity(0.9, 1.0, 2, 4)


def test_the_decomposition_identity_holds_on_the_real_gold_set(gold, engines):
    """End to end, on every fusion/input pair the report prints.

    A same-order question contributes exactly zero to the aggregate delta. That
    is the premise of the whole report, and it is cheap enough to verify over
    the shipped gold set rather than argue for.
    """
    for label, engine_name, inputs in arbitration.FUSIONS:
        engine = engines[engine_name]
        fused = arbitration.rankings(engine, gold, "hybrid", 5)
        for name in inputs:
            other = arbitration.rankings(engine, gold, name, 5)
            acted = [
                arbitration.relation(f, o) != arbitration.SAME_ORDER
                for f, o in zip(fused, other)
            ]
            for metric, _ in arbitration.METRICS:
                left = [
                    arbitration.score(metric, r, g["relevant"], 5)
                    for r, g in zip(fused, gold)
                ]
                right = [
                    arbitration.score(metric, r, g["relevant"], 5)
                    for r, g in zip(other, gold)
                ]
                total, subset, count, _, _ = arbitration.decompose(left, right, acted)
                arbitration.check_identity(total, subset, count, len(gold))


def test_same_order_questions_score_identically(gold, engines):
    """The mechanism behind the identity, stated directly."""
    engine = engines["base"]
    fused = arbitration.rankings(engine, gold, "hybrid", 5)
    other = arbitration.rankings(engine, gold, "vector", 5)
    checked = 0
    for ranking, baseline, record in zip(fused, other, gold):
        if arbitration.relation(ranking, baseline) != arbitration.SAME_ORDER:
            continue
        checked += 1
        for metric, _ in arbitration.METRICS:
            assert arbitration.score(
                metric, ranking, record["relevant"], 5
            ) == arbitration.score(metric, baseline, record["relevant"], 5)
    assert checked, "no same-order questions to check - the premise is untested"


# --------------------------------------------------------------------------
# promotion: the one thing no single input list could have done
# --------------------------------------------------------------------------


def test_promoted_counts_documents_no_input_surfaced():
    fused = [["a", "z"], ["a", "b"]]
    inputs = {"x": [["a", "b"], ["a", "b"]], "y": [["a", "c"], ["b", "a"]]}
    assert arbitration.promoted(fused, inputs) == 1


def test_promoted_is_zero_when_the_fusion_only_permutes():
    fused = [["b", "a"]]
    inputs = {"x": [["a", "b"]]}
    assert arbitration.promoted(fused, inputs) == 0


# --------------------------------------------------------------------------
# the standing claim
# --------------------------------------------------------------------------


def test_the_two_list_fusion_is_not_inert_against_its_inputs(gold, engines):
    """The README's cancellation claim, tested rather than asserted.

    The ablation table shows the two-list fusion tying TF-IDF on five of six
    columns. The README says that is cancellation - the fusion reorders most
    questions and the wins and losses balance - and explicitly rejects the
    reading that RRF simply returned the vector list. If that ever stops being
    true the prose is wrong, so this fails first.

    The threshold is deliberately loose. The claim is "most questions", not a
    specific rate, and pinning the exact count would turn every corpus
    increment into a test failure that says nothing.
    """
    engine = engines["base"]
    fused = arbitration.rankings(engine, gold, "hybrid", 5)
    for name in ("bm25", "vector"):
        other = arbitration.rankings(engine, gold, name, 5)
        counts = arbitration.relation_counts(fused, other)
        acted = counts[arbitration.SAME_SET] + counts[arbitration.DIFFERENT_SET]
        assert acted > len(gold) / 2, (
            "the two-list fusion returned the {} ranking unchanged on {} of {} "
            "questions, so the README's 'cancellation, not inertness' reading "
            "no longer holds".format(name, counts[arbitration.SAME_ORDER], len(gold))
        )


def test_verdict_names_inertness_when_the_fusion_really_does_nothing():
    """The other branch, which the shipped corpus does not exercise."""
    gold_rows = [{"question": "q", "relevant": ["a"]} for _ in range(20)]
    same = [["a", "b"] for _ in range(20)]
    line = arbitration.verdict("hybrid2", "vector", same, same, gold_rows, 5)
    assert "INERT" in line


def test_verdict_names_cancellation_on_a_balanced_split():
    gold_rows = [{"question": "q", "relevant": ["a"]} for _ in range(20)]
    fused = [["a", "b"] if i % 2 else ["b", "a"] for i in range(20)]
    other = [["b", "a"] if i % 2 else ["a", "b"] for i in range(20)]
    line = arbitration.verdict("hybrid2", "vector", fused, other, gold_rows, 5)
    assert "CANCELLATION" in line


# --------------------------------------------------------------------------
# the script itself
# --------------------------------------------------------------------------


def test_main_runs_and_reports_every_pair(capsys):
    assert arbitration.main(["--bootstrap-resamples", "200"]) == 0
    out = capsys.readouterr().out
    for label, _, inputs in arbitration.FUSIONS:
        for name in inputs:
            assert "{} vs {}".format(label, name) in out
    assert "Reading, on nDCG@5:" in out
    assert "no input's top-k" in out


def test_main_accepts_a_slice(capsys):
    assert arbitration.main(["--slice", "hard", "--bootstrap-resamples", "200"]) == 0
    assert "hard slice" in capsys.readouterr().out
