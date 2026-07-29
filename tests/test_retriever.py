import pytest

from ragkb.retriever import HybridRetriever
from ragkb.text import chunk_document

DOCS = {
    "lexical": "BM25 saturates term frequency and normalises by document length. "
    "It relies on exact term overlap between the query and the document.",
    "vectorspace": "TF-IDF cosine similarity compares normalised sparse vectors. "
    "Direction rather than magnitude decides the ranking.",
    "fusion": "Reciprocal rank fusion merges ranked lists by summing one over "
    "a constant plus the rank. It never looks at the raw relevance scores.",
}


def build(rrf_k=60):
    chunks = []
    for doc_id, text in sorted(DOCS.items()):
        chunks.extend(chunk_document(doc_id, text, target_tokens=40, overlap_tokens=10))
    return HybridRetriever(chunks, rrf_k=rrf_k)


def test_hybrid_returns_the_expected_document():
    results = build().search("how does reciprocal rank fusion merge lists", k=3)
    assert results[0].doc_id == "fusion"


def test_methods_all_return_results():
    retriever = build()
    for method in ("hybrid", "bm25", "vector"):
        results = retriever.search("term frequency saturation", k=2, method=method)
        assert results and results[0].doc_id == "lexical"


def test_hybrid_scores_are_rrf_shaped():
    retriever = build(rrf_k=60)
    top = retriever.search("cosine similarity of sparse vectors", k=1)[0]
    assert top.method == "hybrid"
    assert top.bm25_rank is not None or top.vector_rank is not None
    assert top.score == pytest.approx(
        sum(top.components.values()), rel=1e-9
    )
    assert 0.0 < top.score <= 2.0 / 61.0


def test_results_are_ordered_by_score():
    scores = [r.score for r in build().search("document length normalisation", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing():
    assert build().search("   ", k=3) == []


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        build().search("anything", method="magic")


def test_invalid_rrf_k_raises():
    with pytest.raises(ValueError):
        HybridRetriever([], rrf_k=0)
