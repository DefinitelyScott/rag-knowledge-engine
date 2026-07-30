"""Tests for MMR re-ranking and the redundancy diagnostic."""

import pytest

from ragkb.rerank import mean_pairwise_similarity, mmr_rerank, normalize_scores
from ragkb.retriever import HybridRetriever
from ragkb.text import chunk_document


def constant_similarity(value):
    def similarity(left, right):
        return 1.0 if left == right else value

    return similarity


def matrix_similarity(matrix):
    def similarity(left, right):
        return matrix[left][right]

    return similarity


def test_normalize_scores_maps_to_unit_interval():
    scaled = dict(normalize_scores([(0, 4.0), (1, 2.0), (2, 3.0)]))
    assert scaled[0] == pytest.approx(1.0)
    assert scaled[1] == pytest.approx(0.0)
    assert scaled[2] == pytest.approx(0.5)


def test_normalize_scores_handles_a_flat_pool():
    # No relevance signal to preserve, so every item is equally relevant and
    # MMR is left to decide purely on diversity.
    assert normalize_scores([(0, 2.0), (1, 2.0)]) == [(0, 1.0), (1, 1.0)]


def test_lambda_one_reproduces_the_input_ranking():
    candidates = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
    selected = mmr_rerank(candidates, constant_similarity(0.9), k=4, lambda_=1.0)
    assert [index for index, _ in selected] == [0, 1, 2, 3]


def test_original_scores_are_preserved_through_reranking():
    candidates = [(0, 0.9), (1, 0.8)]
    selected = mmr_rerank(candidates, constant_similarity(0.0), k=2, lambda_=0.5)
    assert dict(selected) == {0: 0.9, 1: 0.8}


def test_diversity_can_outrank_relevance():
    # 1 is nearly identical to 0; 2 is unrelated but slightly less relevant.
    matrix = [
        [1.0, 0.95, 0.05],
        [0.95, 1.0, 0.05],
        [0.05, 0.05, 1.0],
    ]
    candidates = [(0, 1.0), (1, 0.9), (2, 0.8)]
    relevance_first = mmr_rerank(candidates, matrix_similarity(matrix), k=2, lambda_=1.0)
    diversity_first = mmr_rerank(
        candidates, matrix_similarity(matrix), k=2, lambda_=0.5
    )
    assert [index for index, _ in relevance_first] == [0, 1]
    assert [index for index, _ in diversity_first] == [0, 2]


def test_selection_is_truncated_to_k():
    candidates = [(index, 1.0 - index / 10.0) for index in range(6)]
    selected = mmr_rerank(candidates, constant_similarity(0.1), k=3, lambda_=0.7)
    assert len(selected) == 3


def test_degenerate_inputs_return_empty():
    assert mmr_rerank([], constant_similarity(0.0), k=3) == []
    assert mmr_rerank([(0, 1.0)], constant_similarity(0.0), k=0) == []


def test_lambda_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        mmr_rerank([(0, 1.0)], constant_similarity(0.0), k=1, lambda_=1.5)


def test_mean_pairwise_similarity():
    matrix = [
        [1.0, 0.2, 0.4],
        [0.2, 1.0, 0.6],
        [0.4, 0.6, 1.0],
    ]
    assert mean_pairwise_similarity([0, 1, 2], matrix_similarity(matrix)) == pytest.approx(0.4)
    # Redundancy is undefined for a single result.
    assert mean_pairwise_similarity([0], matrix_similarity(matrix)) == 0.0


DOCUMENTS = {
    "bm25": (
        "BM25 saturates term frequency with a k1 parameter. "
        "The saturation curve flattens after a few occurrences. "
        "Length normalisation is controlled by the b parameter."
    ),
    "bm25-notes": (
        "BM25 saturates term frequency with a k1 parameter. "
        "The saturation curve flattens after a few occurrences. "
        "This restates the same saturation behaviour again."
    ),
    "fusion": (
        "Reciprocal rank fusion merges two ranked lists. "
        "Each list contributes one over k plus rank to every item."
    ),
}


def build_retriever(**kwargs):
    chunks = []
    for doc_id in sorted(DOCUMENTS):
        chunks.extend(
            chunk_document(doc_id, DOCUMENTS[doc_id], target_tokens=20, overlap_tokens=5)
        )
    return HybridRetriever(chunks, **kwargs)


def test_retriever_rejects_an_out_of_range_lambda():
    with pytest.raises(ValueError):
        build_retriever(mmr_lambda=-0.1)


def test_search_lambda_one_matches_unranked_search():
    retriever = build_retriever()
    query = "how does BM25 saturate term frequency"
    plain = retriever.search(query, k=3)
    reranked = retriever.search(query, k=3, mmr_lambda=1.0)
    assert [r.chunk_index for r in plain] == [r.chunk_index for r in reranked]


def test_mmr_does_not_increase_redundancy():
    retriever = build_retriever()
    query = "how does BM25 saturate term frequency"
    plain = retriever.search(query, k=3)
    reranked = retriever.search(query, k=3, mmr_lambda=0.4)
    assert retriever.redundancy(reranked) <= retriever.redundancy(plain) + 1e-9


def test_mmr_rank_is_recorded_only_when_reranking():
    retriever = build_retriever()
    query = "how does BM25 saturate term frequency"
    assert all(r.mmr_rank is None for r in retriever.search(query, k=2))
    reranked = retriever.search(query, k=2, mmr_lambda=0.5)
    assert [r.mmr_rank for r in reranked] == [1, 2]


def test_constructor_lambda_is_the_default_for_search():
    retriever = build_retriever(mmr_lambda=0.4)
    query = "how does BM25 saturate term frequency"
    default = retriever.search(query, k=3)
    explicit = retriever.search(query, k=3, mmr_lambda=0.4)
    assert [r.chunk_index for r in default] == [r.chunk_index for r in explicit]
    assert all(r.mmr_rank is not None for r in default)


def test_pairwise_similarity_is_symmetric_and_self_maximal():
    retriever = build_retriever()
    index = retriever.vector
    assert index.pairwise_similarity(0, 0) == 1.0
    assert index.pairwise_similarity(0, 1) == pytest.approx(
        index.pairwise_similarity(1, 0)
    )
    assert 0.0 <= index.pairwise_similarity(0, 1) <= 1.0
