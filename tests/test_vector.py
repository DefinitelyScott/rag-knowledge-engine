import math

from ragkb.text import tokenize
from ragkb.vector import TfidfIndex

DOCS = [
    tokenize("cosine similarity between normalised sparse vectors"),
    tokenize("bm25 uses term frequency saturation and length normalisation"),
    tokenize("vectors are normalised so that document length does not dominate"),
]


def build():
    return TfidfIndex().fit(DOCS)


def test_vectors_are_unit_length():
    index = build()
    for vector in index.vectors:
        norm = math.sqrt(sum(w * w for w in vector.values()))
        assert abs(norm - 1.0) < 1e-9


def test_self_similarity_is_one():
    index = build()
    query = index.vectorize_query(DOCS[0])
    assert abs(index.similarity(query, 0) - 1.0) < 1e-9


def test_search_ranks_the_matching_document_first():
    index = build()
    hits = index.search(tokenize("term frequency saturation"), k=3)
    assert hits[0][0] == 1


def test_out_of_vocabulary_query_returns_nothing():
    index = build()
    assert index.search(tokenize("thermodynamics"), k=3) == []


def test_rarer_terms_get_higher_idf():
    index = build()
    assert index.idf["cosine"] > index.idf["normalised"]


def test_repeated_terms_are_dampened():
    index = TfidfIndex(sublinear_tf=True).fit([tokenize("alpha alpha alpha alpha beta")])
    vector = index.vectors[0]
    assert vector["alpha"] < 4 * vector["beta"]
