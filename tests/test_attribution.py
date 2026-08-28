"""Tests for per-term score attribution.

The load-bearing assertions here are the exact-sum identities. A saliency
story that does not add back up to the score is a story, not an explanation,
so every decomposition is checked against the scorer it claims to decompose
rather than eyeballed.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ragkb.attribution import (
    attribute,
    attribute_bm25,
    attribute_vector,
    best_chunk_for_doc,
    query_model,
)
from ragkb.bm25 import BM25Index
from ragkb.expansion import ExpansionConfig
from ragkb.retriever import HybridRetriever
from ragkb.text import Chunk, tokenize
from ragkb.vector import TfidfIndex

DOCUMENTS = [
    "Okapi BM25 saturates term frequency so the tenth occurrence of a word "
    "counts for much less than the first occurrence.",
    "The b parameter controls length normalisation in BM25, trading document "
    "length against raw term frequency.",
    "TF-IDF cosine similarity ranks documents by the angle between sparse "
    "normalised vectors of weighted terms.",
    "Reciprocal rank fusion combines two ranked lists by summing one over a "
    "constant plus the rank of each item.",
]


def make_chunks():
    return [
        Chunk(
            doc_id="doc{}.md".format(index),
            chunk_id="doc{}.md#0".format(index),
            text=text,
            token_count=len(tokenize(text)),
            sentence_start=0,
            sentence_end=1,
            metadata={},
        )
        for index, text in enumerate(DOCUMENTS)
    ]


@pytest.fixture
def retriever():
    return HybridRetriever(make_chunks())


@pytest.fixture
def tokenised():
    return [tokenize(text) for text in DOCUMENTS]


def test_bm25_contributions_sum_to_the_bm25_score(tokenised):
    index = BM25Index().fit(tokenised)
    weights = {term: 1.0 for term in tokenize("bm25 term frequency saturation")}
    for doc_index in range(index.doc_count):
        report = attribute_bm25(index, weights, doc_index)
        assert report.attributed == pytest.approx(
            index.score_document_weighted(weights, doc_index)
        )
        assert report.attributed == pytest.approx(report.score)


def test_bm25_attribution_honours_query_term_weights(tokenised):
    index = BM25Index().fit(tokenised)
    single = attribute_bm25(index, {"bm25": 1.0}, 1)
    double = attribute_bm25(index, {"bm25": 2.0}, 1)
    # Saturation applies to the document's term frequency, not the query's, so
    # doubling a query weight really does double that term's contribution.
    assert double.score == pytest.approx(2.0 * single.score)


def test_vector_contributions_sum_to_the_cosine(tokenised):
    index = TfidfIndex().fit(tokenised)
    tokens = tokenize("cosine similarity between normalised vectors")
    for doc_index in range(index.doc_count):
        report = attribute_vector(index, "vector", tokens, doc_index)
        expected = index.similarity(index.vectorize_query(tokens), doc_index)
        assert report.attributed == pytest.approx(expected)
        assert report.attributed == pytest.approx(report.score)


def test_contributions_are_sorted_heaviest_first(tokenised):
    index = BM25Index().fit(tokenised)
    weights = {term: 1.0 for term in tokenize("bm25 length normalisation parameter")}
    terms = attribute_bm25(index, weights, 1).terms
    assert terms
    assert terms == sorted(terms, key=lambda item: (-item.contribution, item.term))


def test_missing_and_unknown_are_different_diagnoses(tokenised):
    index = BM25Index().fit(tokenised)
    # "fusion" is in the collection but not in chunk 0; "kubernetes" is in
    # neither, and no amount of re-ranking could ever recover it.
    weights = {"fusion": 1.0, "kubernetes": 1.0, "bm25": 1.0}
    report = attribute_bm25(index, weights, 0)
    assert report.missing == ["fusion"]
    assert report.unknown == ["kubernetes"]
    assert [term.term for term in report.terms] == ["bm25"]


def test_attribution_rank_matches_what_search_returned(retriever):
    query = "how does reciprocal rank fusion combine lists"
    results = retriever.search(query, k=4)
    for position, result in enumerate(results, start=1):
        report = attribute(retriever, query, result.chunk_index)
        assert report.rank == position
        assert report.score == pytest.approx(result.score)
        assert report.doc_id == result.doc_id


def test_fusion_contributions_sum_to_the_fused_score(retriever):
    query = "what does the b parameter control"
    result = retriever.search(query, k=1)[0]
    report = attribute(retriever, query, result.chunk_index)
    total = sum(report.fusion_contribution(entry.name) for entry in report.lists)
    assert total == pytest.approx(report.score)


def test_fusion_contribution_matches_the_rrf_formula(retriever):
    query = "term frequency saturation"
    report = attribute(retriever, query, retriever.search(query, k=1)[0].chunk_index)
    for entry in report.lists:
        expected = 0.0 if entry.rank is None else 1.0 / (report.rrf_k + entry.rank)
        assert report.fusion_contribution(entry.name) == pytest.approx(expected)


def test_per_list_scores_match_the_single_method_scores(retriever):
    query = "sparse normalised vectors of weighted terms"
    chunk_index = retriever.search(query, k=1, method="vector")[0].chunk_index
    report = attribute(retriever, query, chunk_index, method="vector")
    entry = next(item for item in report.lists if item.name == "vector")
    tokens = tokenize(query)
    expected = retriever.vector.similarity(
        retriever.vector.vectorize_query(tokens), chunk_index
    )
    assert entry.score == pytest.approx(expected)
    assert report.score == pytest.approx(expected)


def test_an_unretrieved_chunk_still_reports_its_real_rank(retriever):
    query = "reciprocal rank fusion"
    top = retriever.search(query, k=1)[0].chunk_index
    other = next(index for index in range(len(retriever.chunks)) if index != top)
    report = attribute(retriever, query, other)
    assert report.rank is None or report.rank > 1


def test_a_chunk_matching_nothing_scores_zero_and_has_no_rank(retriever):
    report = attribute(retriever, "kubernetes sidecar autoscaling", 0)
    assert report.score == 0.0
    assert report.rank is None
    assert all(not entry.terms for entry in report.lists)


def test_query_model_is_unexpanded_when_expansion_is_off(retriever):
    weights, expanded = query_model(retriever, "bm25 bm25 length")
    assert expanded is False
    assert weights == {term: float(count) for term, count in Counter(
        tokenize("bm25 bm25 length")
    ).items()}


def test_query_model_reports_expansion_when_it_borrows_terms():
    retriever = HybridRetriever(
        make_chunks(), expansion=ExpansionConfig(feedback_docs=2, feedback_terms=5)
    )
    weights, expanded = query_model(retriever, "length normalisation")
    assert expanded is True
    assert set(weights) - set(tokenize("length normalisation"))


def test_ngram_list_decomposes_into_grams_not_words():
    retriever = HybridRetriever(make_chunks(), ngram=4)
    query = "quantisation"
    report = attribute(retriever, query, 0, method="hybrid")
    entry = next(item for item in report.lists if item.name == "ngram")
    # The n-gram list is a TF-IDF index over character n-grams, so its rows are
    # grams; nothing in it should ever be a whole query word.
    assert all(len(term.term) <= 4 for term in entry.terms)
    assert entry.attributed == pytest.approx(entry.score)


def test_best_chunk_for_doc_prefers_the_highest_ranked_chunk(retriever):
    index = best_chunk_for_doc(retriever, "length normalisation", "doc1.md")
    assert retriever.chunks[index].doc_id == "doc1.md"


def test_best_chunk_for_doc_falls_back_when_nothing_scores(retriever):
    index = best_chunk_for_doc(retriever, "kubernetes sidecar", "doc2.md")
    assert retriever.chunks[index].doc_id == "doc2.md"


def test_best_chunk_for_doc_rejects_an_unknown_document(retriever):
    with pytest.raises(KeyError):
        best_chunk_for_doc(retriever, "bm25", "nope.md")


def test_attribute_rejects_a_bad_method_and_a_bad_index(retriever):
    with pytest.raises(ValueError):
        attribute(retriever, "bm25", 0, method="ngram")
    with pytest.raises(IndexError):
        attribute(retriever, "bm25", len(retriever.chunks))


def test_format_shows_the_fusion_arithmetic_and_the_terms(retriever):
    query = "what does the b parameter control"
    report = attribute(retriever, query, retriever.search(query, k=1)[0].chunk_index)
    text = report.format()
    assert "fusion (rrf_k=60)" in text
    assert "bm25" in text and "vector" in text
    assert "contrib" in text
