"""Tests for the character n-gram index and its place in the fusion.

The tests that matter most here are the two at the ends of the file. One shows
the capability a word-level index provably does not have - a morphological
variant scoring above zero where TF-IDF scores exactly zero - because that is
the entire justification for a third list. The other shows that leaving the
list out reproduces the two-list ranking exactly, which is the same identity
check ``mmr_lambda = 1.0`` and ``alpha = 1.0`` provide elsewhere.
"""

from __future__ import annotations

import pytest

from ragkb.engine import EngineConfig, RAGEngine
from ragkb.ngram import BOUNDARY, CharNgramIndex, char_ngrams
from ragkb.retriever import METHODS, NGRAM_METHOD, HybridRetriever
from ragkb.text import chunk_document, tokenize
from ragkb.vector import TfidfIndex


def test_ngrams_carry_word_boundaries():
    grams = char_ngrams(["rank"], n=4)
    assert grams == ["_ran", "rank", "ank_"]
    assert all(len(gram) == 4 for gram in grams)


def test_short_tokens_are_emitted_whole_not_dropped():
    # "_a_" is shorter than n. Dropping it would silently change what the
    # index can match; the IDF weighting is what handles it being common.
    assert char_ngrams(["a"], n=4) == [BOUNDARY + "a" + BOUNDARY]
    assert char_ngrams(["of"], n=4) == ["_of_"]


def test_boundary_marker_distinguishes_prefix_from_interior():
    # "_ran" (ranking) and "eran" (reranking) are different features, which is
    # what stops a word and a word containing it collapsing together.
    assert "_ran" in char_ngrams(["ranking"], n=4)
    assert "_ran" not in char_ngrams(["reranking"], n=4)


def test_n_below_two_is_rejected():
    with pytest.raises(ValueError):
        char_ngrams(["rank"], n=1)
    with pytest.raises(ValueError):
        CharNgramIndex(n=1)


def _docs():
    return [
        tokenize("product quantisation compresses the stored vectors"),
        tokenize("an inverted index maps terms to posting lists"),
    ]


def test_morphological_variant_scores_where_tfidf_scores_zero():
    """The whole reason the third list exists, stated as an assertion.

    "quantised" appears nowhere in the corpus. A word index has no feature in
    common with it and must score zero; the n-gram index shares _qua, quan,
    uant, anti and ntis with "quantisation" and ranks the right document.
    """
    documents = _docs()
    query = tokenize("quantised")

    words = TfidfIndex().fit(documents)
    assert words.search(query, k=2) == []

    grams = CharNgramIndex(n=4).fit(documents)
    hits = grams.search(query, k=2)
    assert hits, "n-gram index should match a morphological variant"
    assert hits[0][0] == 0
    assert hits[0][1] > 0.0


def test_typo_still_retrieves():
    grams = CharNgramIndex(n=4).fit(_docs())
    hits = grams.search(tokenize("inverrted index"), k=2)
    assert hits[0][0] == 1


def test_weighted_query_accumulates_gram_weights():
    index = CharNgramIndex(n=4).fit(_docs())
    weighted = index.search_weighted({"quantisation": 1.0}, k=2)
    plain = index.search(tokenize("quantisation"), k=2)
    assert weighted[0][0] == plain[0][0] == 0
    # A zero or negative weight contributes nothing at all, matching the word
    # index's behaviour rather than quietly inverting the term.
    assert index.search_weighted({"quantisation": 0.0}, k=2) == []


def test_serialisation_round_trip_preserves_scores():
    index = CharNgramIndex(n=5).fit(_docs())
    restored = CharNgramIndex.from_dict(index.to_dict())
    assert isinstance(restored, CharNgramIndex)
    assert restored.n == 5
    query = tokenize("quantisation")
    assert restored.search(query, k=2) == index.search(query, k=2)


def _chunks():
    text = (
        "Product quantisation compresses stored vectors into codes.\n\n"
        "An inverted index maps each term to a posting list of documents.\n\n"
        "Reciprocal rank fusion merges ranked lists without shared scales.\n"
    )
    return chunk_document("doc", text, target_tokens=30, overlap_tokens=5)


def test_ngram_off_fuses_exactly_two_lists():
    """The identity check: off has to mean unchanged, bit for bit.

    Comparing the default against ``ngram=None`` would be circular - they are
    the same call. So this reconstructs the fused score from the BM25 and
    TF-IDF ranks the result itself reports, and asserts nothing else went into
    it: two components, and a total that is their exact sum.
    """
    retriever = HybridRetriever(_chunks(), ngram=None)
    assert retriever.ngram is None
    assert retriever.methods == METHODS
    for result in retriever.search("reciprocal rank fusion", k=3):
        assert result.ngram_rank is None
        assert set(result.components) == {"bm25_rrf", "vector_rrf"}
        expected = sum(
            1.0 / (retriever.rrf_k + rank)
            for rank in (result.bm25_rank, result.vector_rank)
            if rank is not None
        )
        assert result.score == pytest.approx(expected)


def test_ngram_list_joins_the_fusion_and_is_reported():
    retriever = HybridRetriever(_chunks(), ngram=4)
    assert retriever.methods == METHODS + (NGRAM_METHOD,)
    results = retriever.search("reciprocal rank fusion", k=2)
    assert results
    top = results[0]
    assert top.ngram_rank is not None
    assert "ngram_rrf" in top.components
    # Three lists contribute, so the fused score is the sum of three terms.
    assert top.score == pytest.approx(sum(top.components.values()))


def test_ngram_method_is_only_offered_when_it_was_fitted():
    plain = HybridRetriever(_chunks())
    with pytest.raises(ValueError):
        plain.search("rank fusion", method=NGRAM_METHOD)
    enabled = HybridRetriever(_chunks(), ngram=4)
    assert enabled.search("rank fusion", method=NGRAM_METHOD)


def test_engine_config_threads_the_flag_through_and_reports_the_cost():
    documents = {"doc": _chunks()[0].text}
    engine = RAGEngine(documents, config=EngineConfig(ngram=4))
    stats = engine.stats()
    assert stats["ngram_vocabulary"] > stats["vocabulary"]
    assert "ngram_vocabulary" not in RAGEngine(documents).stats()
