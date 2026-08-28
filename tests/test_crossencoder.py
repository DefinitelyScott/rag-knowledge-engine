"""Tests for the second-stage cross-encoder re-ranker."""

from __future__ import annotations

import pytest

from ragkb.crossencoder import (
    CrossEncoder,
    CrossEncoderWeights,
    LexicalCrossEncoder,
    RerankConfig,
    _minimal_window,
    _term_positions,
)
from ragkb.engine import EngineConfig, RAGEngine
from ragkb.text import tokenize


def tokens(text: str):
    return tokenize(text)


class ConstantEncoder(CrossEncoder):
    """Scores every pair the same - isolates the blending from the features."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def score(self, query_tokens, passage_tokens) -> float:
        return self.value


class LastTokenEncoder(CrossEncoder):
    """Scores 1.0 only when the passage ends with the query's last token."""

    def score(self, query_tokens, passage_tokens) -> float:
        if not query_tokens or not passage_tokens:
            return 0.0
        return 1.0 if passage_tokens[-1] == query_tokens[-1] else 0.0


# --------------------------------------------------------------------------
# window helper
# --------------------------------------------------------------------------


def test_minimal_window_finds_the_tightest_span():
    positions = {"a": [0, 10], "b": [11], "c": [12]}
    # The tightest span covering a, b and c is 10..12.
    assert _minimal_window(positions) == 3


def test_minimal_window_of_a_single_term_is_one():
    assert _minimal_window({"a": [4, 9]}) == 1


def test_minimal_window_of_nothing_is_none():
    assert _minimal_window({}) is None


def test_term_positions_only_reports_matched_terms():
    positions = _term_positions(["b25", "missing"], tokens("bm25 b25 scores b25"))
    assert positions == {"b25": [1, 3]}


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def test_coverage_prefers_a_passage_containing_every_query_term():
    encoder = LexicalCrossEncoder(idf={"length": 2.0, "normalisation": 2.0})
    query = tokens("length normalisation")
    both = encoder.explain(query, tokens("length normalisation matters"))
    one = encoder.explain(query, tokens("length length length length"))
    assert both["coverage"] == pytest.approx(1.0)
    assert one["coverage"] == pytest.approx(0.5)
    assert both["score"] > one["score"]


def test_coverage_is_idf_weighted():
    """Matching the rare term should count for more than matching the common one."""
    encoder = LexicalCrossEncoder(idf={"the": 0.1, "saturation": 3.0})
    query = ["the", "saturation"]
    rare = encoder.coverage(query, {"saturation": [0]})
    common = encoder.coverage(query, {"the": [0]})
    assert rare > common
    assert rare == pytest.approx(3.0 / 3.1)


def test_coverage_falls_back_to_counting_without_idf():
    encoder = LexicalCrossEncoder()
    assert encoder.coverage(["a", "b"], {"a": [0]}) == pytest.approx(0.5)


def test_proximity_rewards_terms_that_sit_close_together():
    encoder = LexicalCrossEncoder(idf={"b": 1.0, "parameter": 1.0})
    query = tokens("b parameter")
    near = encoder.explain(query, tokens("the b parameter controls it"))
    far = encoder.explain(
        query, tokens("the b value " + "filler " * 20 + "is a parameter")
    )
    assert near["proximity"] > far["proximity"]
    assert near["coverage"] == pytest.approx(far["coverage"])
    assert near["score"] > far["score"]


def test_proximity_is_the_signal_bag_of_words_cannot_see():
    """Same terms, same counts, different arrangement - only proximity moves."""
    encoder = LexicalCrossEncoder(idf={"alpha": 1.0, "beta": 1.0})
    query = ["alpha", "beta"]
    together = encoder.explain(query, ["x", "alpha", "beta", "y"])
    apart = encoder.explain(query, ["alpha", "x", "y", "beta"])
    assert together["coverage"] == apart["coverage"]
    assert together["proximity"] > apart["proximity"]


def test_phrase_matches_adjacent_query_bigrams():
    encoder = LexicalCrossEncoder()
    query = tokens("rank fusion")
    assert encoder.phrase(query, tokens("reciprocal rank fusion works")) == 1.0
    assert encoder.phrase(query, tokens("fusion of every rank")) == 0.0


def test_phrase_is_zero_for_a_one_word_query():
    assert LexicalCrossEncoder().phrase(["bm25"], tokens("bm25 bm25")) == 0.0


def test_score_stays_in_the_unit_interval():
    encoder = LexicalCrossEncoder(idf={"rank": 1.0, "fusion": 1.0})
    perfect = encoder.score(tokens("rank fusion"), tokens("rank fusion"))
    nothing = encoder.score(tokens("rank fusion"), tokens("unrelated words here"))
    assert 0.0 <= nothing <= perfect <= 1.0


def test_weights_are_normalised_on_construction():
    encoder = LexicalCrossEncoder(
        weights=CrossEncoderWeights(coverage=2.0, proximity=1.0, phrase=1.0)
    )
    assert encoder.weights.coverage == pytest.approx(0.5)
    assert encoder.weights.proximity == pytest.approx(0.25)


def test_weights_must_be_positive():
    with pytest.raises(ValueError):
        LexicalCrossEncoder(
            weights=CrossEncoderWeights(coverage=0.0, proximity=0.0, phrase=0.0)
        )


def test_an_all_stopword_query_still_scores():
    """Stripping stop words must not leave the encoder with nothing to match."""
    encoder = LexicalCrossEncoder(idf={"what": 1.0, "is": 1.0})
    assert encoder.score(tokens("what is it"), tokens("what is it")) > 0.0


# --------------------------------------------------------------------------
# blending
# --------------------------------------------------------------------------


def test_alpha_one_reproduces_the_first_stage_ordering():
    candidates = [(0, 0.9), (1, 0.5), (2, 0.1)]
    documents = [["a"], ["b"], ["c"]]
    out = LastTokenEncoder().rerank(["c"], candidates, documents, alpha=1.0)
    assert out == candidates


def test_alpha_zero_ranks_purely_by_the_second_stage():
    candidates = [(0, 0.9), (1, 0.5), (2, 0.1)]
    documents = [["a"], ["b"], ["c"]]
    out = LastTokenEncoder().rerank(["c"], candidates, documents, alpha=0.0)
    assert [index for index, _ in out] == [2, 0, 1]


def test_blending_can_promote_a_lower_ranked_candidate():
    candidates = [(0, 1.0), (1, 0.95)]
    documents = [["a"], ["c"]]
    out = LastTokenEncoder().rerank(["c"], candidates, documents, alpha=0.4)
    assert out[0][0] == 1


def test_ties_keep_the_incoming_order():
    candidates = [(0, 0.6), (1, 0.6), (2, 0.6)]
    documents = [["a"], ["a"], ["a"]]
    out = ConstantEncoder().rerank(["a"], candidates, documents, alpha=0.5)
    assert [index for index, _ in out] == [0, 1, 2]


def test_rerank_respects_k_and_defaults_to_the_whole_pool():
    candidates = [(0, 0.9), (1, 0.5), (2, 0.1)]
    documents = [["a"], ["b"], ["c"]]
    encoder = ConstantEncoder()
    assert len(encoder.rerank(["a"], candidates, documents, k=2, alpha=0.5)) == 2
    assert len(encoder.rerank(["a"], candidates, documents, alpha=0.5)) == 3


def test_rerank_of_an_empty_pool_is_empty():
    assert ConstantEncoder().rerank(["a"], [], [], alpha=0.5) == []


def test_alpha_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        ConstantEncoder().rerank(["a"], [(0, 1.0)], [["a"]], alpha=1.5)


def test_the_base_interface_requires_a_score_implementation():
    with pytest.raises(NotImplementedError):
        CrossEncoder().score(["a"], ["a"])


def test_config_builds_the_default_encoder_over_the_supplied_idf():
    encoder = RerankConfig().build({"bm25": 2.0})
    assert isinstance(encoder, LexicalCrossEncoder)
    assert encoder.idf == {"bm25": 2.0}


def test_config_prefers_an_explicit_encoder():
    explicit = ConstantEncoder()
    assert RerankConfig(encoder=explicit).build({}) is explicit


# --------------------------------------------------------------------------
# integration with the retriever
# --------------------------------------------------------------------------


DOCUMENTS = {
    "near": "The b parameter controls document length normalisation directly.",
    "far": (
        "The b value appears here. Many unrelated sentences follow about "
        "indexing, storage and evaluation. Much later the text mentions a "
        "parameter in passing."
    ),
    "other": "Reciprocal rank fusion combines two ranked lists by rank alone.",
}


def build_engine(**kwargs) -> RAGEngine:
    return RAGEngine(DOCUMENTS, config=EngineConfig(**kwargs))


def test_disabled_reranking_is_the_default():
    assert build_engine().config.rerank is None


def test_alpha_one_leaves_engine_results_unchanged():
    engine = build_engine()
    query = "what does the b parameter control"
    baseline = [r.chunk.chunk_id for r in engine.search(query, k=3)]
    unchanged = [
        r.chunk.chunk_id
        for r in engine.search(query, k=3, rerank=RerankConfig(alpha=1.0))
    ]
    assert unchanged == baseline


def test_reranking_records_its_own_rank_for_provenance():
    engine = build_engine(rerank=RerankConfig(alpha=0.5))
    results = engine.search("what does the b parameter control", k=3)
    assert all(result.rerank_rank is not None for result in results)
    assert [result.rerank_rank for result in results] == sorted(
        result.rerank_rank for result in results
    )


def test_reranking_promotes_the_passage_that_states_it_in_one_clause():
    engine = build_engine(rerank=RerankConfig(alpha=0.3))
    ranked = engine.ranked_doc_ids(
        "what does the b parameter control", k=3, rerank=RerankConfig(alpha=0.3)
    )
    assert ranked[0] == "near"


def test_per_call_rerank_overrides_the_engine_setting():
    """A per-call alpha of 1.0 must undo an engine configured to rerank hard."""
    query = "what does the b parameter control"
    plain = [r.chunk.chunk_id for r in build_engine().search(query, k=3)]
    engine = build_engine(rerank=RerankConfig(alpha=0.0))
    overridden = [
        r.chunk.chunk_id
        for r in engine.search(query, k=3, rerank=RerankConfig(alpha=1.0))
    ]
    assert overridden == plain


def test_reranking_composes_with_mmr():
    engine = build_engine(rerank=RerankConfig(alpha=0.5), mmr_lambda=0.7)
    results = engine.search("what does the b parameter control", k=2)
    assert len(results) == 2
    assert all(result.rerank_rank is not None for result in results)
    assert all(result.mmr_rank is not None for result in results)
