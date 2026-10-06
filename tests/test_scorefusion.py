import importlib.util
import math
import os

import pytest

from ragkb.engine import EngineConfig, RAGEngine
from ragkb.scorefusion import combine, normalise, ranked_doc_ids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")


def test_minmax_maps_the_list_onto_the_unit_interval():
    values = normalise([(3, 9.0), (1, 5.0), (7, 1.0)], "minmax")
    assert values == {3: 1.0, 1: 0.5, 7: 0.0}


def test_zscore_has_zero_mean_and_unit_spread():
    values = normalise([(0, 4.0), (1, 2.0), (2, 0.0)], "zscore")
    scores = list(values.values())
    assert abs(sum(scores)) < 1e-12
    assert math.isclose(sum(s * s for s in scores) / len(scores), 1.0)
    assert values[0] > values[1] > values[2]


def test_a_flat_list_does_not_divide_by_zero():
    assert normalise([(0, 2.0), (1, 2.0)], "minmax") == {0: 1.0, 1: 1.0}
    assert normalise([(0, 2.0), (1, 2.0)], "zscore") == {0: 0.0, 1: 0.0}
    assert normalise([], "minmax") == {}


def test_unknown_options_are_rejected():
    with pytest.raises(ValueError):
        normalise([(0, 1.0)], "rank")
    with pytest.raises(ValueError):
        combine({"a": [(0, 1.0)]}, combination="borda")
    with pytest.raises(ValueError):
        combine({"a": [(0, 1.0)]}, combination="rrf", rrf_k=0)


def test_combsum_reads_scores_where_rrf_only_reads_ranks():
    # List "a" is far more confident about item 1 than about item 2; list "b"
    # barely prefers item 2. Ranks are a perfect tie (1st + 2nd each), so RRF
    # falls back to the index tie-break, while CombSUM follows the margin.
    lists = {"a": [(1, 10.0), (2, 1.0), (3, 0.0)], "b": [(2, 1.0), (1, 0.99), (3, 0.0)]}
    assert [i for i, _ in combine(lists, "rrf")][:2] == [1, 2]
    assert combine(lists, "rrf")[0][1] == combine(lists, "rrf")[1][1]
    fused = combine(lists, "combsum", "minmax")
    assert fused[0][0] == 1
    assert fused[0][1] > fused[1][1]


def test_combmnz_rewards_agreement_between_lists():
    # Item 5 is top of one list only; item 6 is second in both.
    lists = {"a": [(5, 1.0), (6, 0.6), (8, 0.0)], "b": [(9, 1.0), (6, 0.6), (8, 0.0)]}
    summed = dict(combine(lists, "combsum", "minmax"))
    mnz = dict(combine(lists, "combmnz", "minmax"))
    assert summed[5] == pytest.approx(1.0)
    assert summed[6] == pytest.approx(1.2)
    assert mnz[5] == pytest.approx(1.0)  # one list
    assert mnz[6] == pytest.approx(2.4)  # two lists


def test_a_missing_item_gets_the_list_floor_not_zero():
    # Under z-scores the bottom of a list is negative, so "not retrieved" must
    # not outrank "retrieved, badly" by being handed 0.0.
    lists = {"a": [(1, 3.0), (2, 2.0), (3, 1.0)], "b": [(1, 3.0), (3, 2.0), (4, 1.0)]}
    fused = dict(combine(lists, "combsum", "zscore"))
    floor_a = min(normalise(lists["a"], "zscore").values())
    floor_b = min(normalise(lists["b"], "zscore").values())
    assert fused[4] == pytest.approx(floor_a + floor_b)
    assert fused[2] == pytest.approx(0.0 + floor_b)


@pytest.fixture(scope="module")
def engines():
    return (
        RAGEngine.from_corpus(CORPUS),
        RAGEngine.from_corpus(CORPUS, EngineConfig(ngram=4)),
    )


QUESTIONS = (
    "How does BM25 saturate term frequency?",
    "Why does reciprocal rank fusion ignore the raw scores?",
    "what is a dangling node in pagerank",
    "retreival augmented genration",
)


@pytest.mark.parametrize("question", QUESTIONS)
def test_module_rrf_reproduces_the_engine_bit_for_bit(engines, question):
    base, three = engines
    candidates = base.config.candidates
    assert ranked_doc_ids(
        base.retriever, question, ("bm25", "vector"), "rrf", candidates=candidates
    ) == base.ranked_doc_ids(question, k=5, method="hybrid")
    assert ranked_doc_ids(
        three.retriever, question, ("bm25", "vector", "ngram"), "rrf", candidates=candidates
    ) == three.ranked_doc_ids(question, k=5, method="hybrid")


def test_unknown_list_name_is_rejected(engines):
    base, _ = engines
    with pytest.raises(ValueError):
        ranked_doc_ids(base.retriever, "bm25", ("bm25", "ngram"))


def test_empty_query_returns_nothing(engines):
    base, _ = engines
    assert ranked_doc_ids(base.retriever, "   ", ("bm25", "vector")) == []


def test_harness_identity_check_passes_on_the_bundled_gold_set():
    path = os.path.join(ROOT, "evals", "score_fusion.py")
    spec = importlib.util.spec_from_file_location("score_fusion_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gold = module.load_gold(os.path.join(ROOT, "evals", "gold.jsonl"))
    _, failures = module.run_all(gold, k=5)
    assert failures == []
