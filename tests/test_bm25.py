import math

import pytest

from ragkb.bm25 import BM25Index
from ragkb.text import tokenize

DOCS = [
    tokenize("bm25 ranks documents using term frequency saturation"),
    tokenize("cosine similarity compares normalised tfidf vectors"),
    tokenize("reciprocal rank fusion merges ranked lists from several retrievers"),
    tokenize("bm25 and tfidf are both lexical retrieval methods"),
]


def build():
    return BM25Index().fit(DOCS)


def test_rare_term_outranks_common_term():
    index = build()
    hits = index.search(tokenize("saturation"), k=4)
    assert hits[0][0] == 0


def test_idf_is_lower_for_more_common_terms():
    index = build()
    assert index.idf["bm25"] < index.idf["saturation"]


def test_unknown_term_scores_zero_everywhere():
    index = build()
    assert index.search(tokenize("quantum"), k=4) == []


def test_term_frequency_saturates():
    index = BM25Index(k1=1.5, b=0.0).fit([tokenize("alpha " * 1 + "filler"), tokenize("alpha " * 20 + "filler")])
    once = index.score_document(tokenize("alpha"), 0)
    twenty = index.score_document(tokenize("alpha"), 1)
    assert twenty > once
    assert twenty < 20 * once


def test_length_normalisation_penalises_long_documents():
    short = tokenize("alpha beta")
    long = tokenize("alpha " + "filler " * 40)
    index = BM25Index(b=1.0).fit([short, long])
    assert index.score_document(tokenize("alpha"), 0) > index.score_document(
        tokenize("alpha"), 1
    )


def test_results_are_sorted_descending():
    index = build()
    scores = [score for _, score in index.search(tokenize("bm25 tfidf"), k=4)]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("k1,b", [(-1.0, 0.5), (1.5, 1.5), (1.5, -0.1)])
def test_invalid_parameters_raise(k1, b):
    with pytest.raises(ValueError):
        BM25Index(k1=k1, b=b)


def test_empty_index_returns_nothing():
    assert BM25Index().fit([]).search(tokenize("anything")) == []
