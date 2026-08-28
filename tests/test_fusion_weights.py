"""Tests for weighted Reciprocal Rank Fusion.

The properties worth pinning down are the ones a future change could silently
break: that uniform weights are *exactly* plain RRF (so the feature is opt-in
in the strongest sense), that only the ratios matter, that a zero weight really
removes a list, and that the reported components still add up to the score once
weighting is on.
"""

import math

import pytest

from ragkb.engine import EngineConfig, RAGEngine
from ragkb.retriever import HybridRetriever, resolve_fusion_weights
from ragkb.text import chunk_document

DOCS = {
    "bm25": (
        "BM25 saturates term frequency so a term repeated ten times does not "
        "score ten times higher. The b parameter controls length "
        "normalisation. The k1 parameter controls how quickly saturation "
        "sets in."
    ),
    "vectors": (
        "TF-IDF cosine similarity compares direction rather than magnitude, "
        "so a long document is not penalised simply for being long. Inverse "
        "document frequency down-weights terms that appear everywhere."
    ),
    "fusion": (
        "Reciprocal rank fusion combines ranked lists without needing their "
        "scores to share a scale. Each list contributes one over a constant "
        "plus the rank it assigned."
    ),
}


def build(fusion_weights=None, ngram=4):
    chunks = []
    for doc_id in sorted(DOCS):
        chunks.extend(chunk_document(doc_id, DOCS[doc_id], target_tokens=25, overlap_tokens=5))
    return HybridRetriever(chunks, ngram=ngram, fusion_weights=fusion_weights)


# --- resolve_fusion_weights -------------------------------------------------


def test_none_resolves_to_uniform():
    assert resolve_fusion_weights(None, ["bm25", "vector"]) == {
        "bm25": 1.0,
        "vector": 1.0,
    }


def test_uniform_input_resolves_to_uniform_output():
    """Any constant mapping is the identity, whatever constant it uses."""
    for constant in (0.5, 1.0, 7.0):
        resolved = resolve_fusion_weights(
            {"bm25": constant, "vector": constant}, ["bm25", "vector"]
        )
        assert resolved == pytest.approx({"bm25": 1.0, "vector": 1.0})


def test_weights_are_rescaled_to_mean_one():
    resolved = resolve_fusion_weights({"bm25": 1.0, "vector": 3.0}, ["bm25", "vector"])
    assert sum(resolved.values()) == pytest.approx(len(resolved))
    assert resolved["vector"] / resolved["bm25"] == pytest.approx(3.0)


def test_proportional_weights_are_the_same_weights():
    a = resolve_fusion_weights({"bm25": 1.0, "vector": 2.0}, ["bm25", "vector"])
    b = resolve_fusion_weights({"bm25": 5.0, "vector": 10.0}, ["bm25", "vector"])
    assert a == pytest.approx(b)


def test_omitted_list_defaults_to_one_not_zero():
    resolved = resolve_fusion_weights({"bm25": 1.0}, ["bm25", "vector"])
    assert resolved == pytest.approx({"bm25": 1.0, "vector": 1.0})


def test_unknown_list_is_rejected():
    with pytest.raises(ValueError, match="unknown fusion list"):
        resolve_fusion_weights({"colbert": 1.0}, ["bm25", "vector"])


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        resolve_fusion_weights({"bm25": -1.0}, ["bm25", "vector"])


def test_all_zero_weights_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        resolve_fusion_weights({"bm25": 0.0, "vector": 0.0}, ["bm25", "vector"])


def test_empty_name_list_is_rejected():
    with pytest.raises(ValueError, match="at least one list"):
        resolve_fusion_weights(None, [])


# --- retriever integration --------------------------------------------------


def test_retriever_defaults_to_uniform_weights():
    assert build().fusion_weights == {"bm25": 1.0, "vector": 1.0, "ngram": 1.0}


def test_ngram_weight_rejected_without_an_ngram_index():
    """Which names are legal depends on what was actually fitted."""
    with pytest.raises(ValueError, match="unknown fusion list"):
        build(fusion_weights={"ngram": 2.0}, ngram=None)


def test_uniform_weights_reproduce_plain_rrf_exactly():
    plain = build()
    weighted = build(fusion_weights={"bm25": 1.0, "vector": 1.0, "ngram": 1.0})
    query = "what does the b parameter control"
    left = plain.search(query, k=5)
    right = weighted.search(query, k=5)
    assert [r.chunk_index for r in left] == [r.chunk_index for r in right]
    for a, b in zip(left, right):
        assert a.score == pytest.approx(b.score)


def test_components_sum_to_the_fused_score_when_weighted():
    """`ragkb.cli explain` prints these components as an explanation of the
    score; they have to still be one."""
    retriever = build(fusion_weights={"ngram": 4.0})
    for result in retriever.search("length normalisation", k=3):
        assert sum(result.components.values()) == pytest.approx(result.score)


def test_zero_weight_drops_a_list_from_the_fusion():
    retriever = build(fusion_weights={"bm25": 0.0, "vector": 0.0, "ngram": 1.0})
    query = "how does saturation work"
    fused = retriever.search(query, k=5)
    ngram_only = retriever.search(query, k=5, method="ngram")
    assert [r.chunk_index for r in fused] == [r.chunk_index for r in ngram_only]


def test_zero_weighted_list_contributes_nothing():
    retriever = build(fusion_weights={"bm25": 0.0})
    for result in retriever.search("inverse document frequency", k=3):
        assert result.components["bm25_rrf"] == 0.0
        # ...but the rank itself is still reported, because the list still ran
        # and "ranked #2 and was ignored" is the diagnosis someone debugging a
        # weight setting actually needs.
        assert result.bm25_rank is None or result.bm25_rank > 0


def test_heavier_weight_moves_a_list_toward_its_own_ordering():
    """The whole point: as one list's weight grows, the fusion converges on it.

    "Converges on it" is a statement about *order*, not about membership. A
    dominant list can only place the chunks it actually ranked, and here the
    vector list ranks fewer than k of them - cosine is zero for a chunk sharing
    no query term. The remaining slots stay with the other lists, so the
    property to assert is that the dominant list's ordering survives as a
    prefix, not that the two lists are equal.
    """
    retriever = build()
    query = "cosine similarity direction"
    target = [r.chunk_index for r in retriever.search(query, k=3, method="vector")]
    heavy = [
        r.chunk_index
        for r in retriever.search(query, k=3, fusion_weights={"vector": 1000.0})
    ]
    assert heavy[: len(target)] == target
    # A weight of 1.0 is no weight at all.
    light = [
        r.chunk_index
        for r in retriever.search(query, k=3, fusion_weights={"vector": 1.0})
    ]
    assert light == [r.chunk_index for r in retriever.search(query, k=3)]


def test_per_call_weights_do_not_mutate_the_retriever():
    retriever = build()
    retriever.search("saturation", k=3, fusion_weights={"ngram": 5.0})
    assert retriever.fusion_weights == {"bm25": 1.0, "vector": 1.0, "ngram": 1.0}


def test_per_call_weights_are_validated():
    with pytest.raises(ValueError, match="unknown fusion list"):
        build().search("saturation", k=3, fusion_weights={"nope": 2.0})


def test_scale_of_fused_scores_is_preserved():
    """Mean-1.0 rescaling is what keeps abstention thresholds comparable."""
    retriever = build()
    query = "term frequency saturation"
    plain = retriever.search(query, k=5)[0].score
    skewed = retriever.search(query, k=5, fusion_weights={"ngram": 2.0})[0].score
    # Same order of magnitude, not merely "some number".
    assert 0.5 < skewed / plain < 2.0
    assert not math.isnan(skewed)


# --- engine integration -----------------------------------------------------


def test_engine_config_threads_weights_to_the_retriever(tmp_path):
    for name, text in DOCS.items():
        (tmp_path / "{}.md".format(name)).write_text(text, encoding="utf-8")
    engine = RAGEngine.from_corpus(
        str(tmp_path), config=EngineConfig(ngram=4, fusion_weights={"ngram": 2.0})
    )
    assert engine.retriever.fusion_weights["ngram"] > engine.retriever.fusion_weights["bm25"]
    assert engine.search("length normalisation", k=2)


def test_explain_trace_still_adds_up_when_weighted(tmp_path):
    """`explain` prints per-list arithmetic and a total; weighting must not
    turn that total into a number that disagrees with the score it explains."""
    from ragkb.attribution import attribute

    for name, text in DOCS.items():
        (tmp_path / "{}.md".format(name)).write_text(text, encoding="utf-8")
    engine = RAGEngine.from_corpus(
        str(tmp_path), config=EngineConfig(ngram=4, fusion_weights={"ngram": 4.0})
    )
    top = engine.search("length normalisation", k=1)[0]
    trace = attribute(engine.retriever, "length normalisation", top.chunk_index)
    total = sum(trace.fusion_contribution(entry.name) for entry in trace.lists)
    assert total == pytest.approx(trace.score)
    # The printed numerator is the *resolved* weight, so the trace also shows
    # the mean-1.0 rescaling instead of echoing back what the user typed:
    # {bm25: 1, vector: 1, ngram: 4} resolves to {0.5, 0.5, 2}.
    rendered = trace.format()
    assert "2/(60+" in rendered and "0.5/(60+" in rendered


def test_explain_trace_is_unchanged_without_weights(tmp_path):
    from ragkb.attribution import attribute

    for name, text in DOCS.items():
        (tmp_path / "{}.md".format(name)).write_text(text, encoding="utf-8")
    engine = RAGEngine.from_corpus(str(tmp_path), config=EngineConfig(ngram=4))
    top = engine.search("length normalisation", k=1)[0]
    trace = attribute(engine.retriever, "length normalisation", top.chunk_index)
    # The unweighted trace keeps its original "1/(k+rank)" shape rather than
    # gaining a redundant "1.0" numerator.
    assert "1/(60+" in trace.format()
