"""Growing an index without rebuilding it.

The claim ``add_documents`` makes is not "close enough". It is that the index
afterwards is the one :meth:`fit` would have produced over the concatenated
input - same counters, same postings, same IDF, same rankings. These tests
check that against real from-scratch builds rather than against a
reimplementation of the same arithmetic, because a bug shared by both would
pass a self-consistency test happily.
"""

from __future__ import annotations

import os

import pytest

from ragkb.bm25 import BM25Index
from ragkb.engine import EngineConfig, RAGEngine, read_corpus
from ragkb.retriever import HybridRetriever
from ragkb.text import chunk_document

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")

DOCS = [
    "the quick brown fox jumps over the lazy dog".split(),
    "bm25 saturates term frequency and discounts common terms".split(),
    "cosine similarity ranks normalised sparse vectors".split(),
    "a fox and a dog share a vocabulary".split(),
]


def _assert_bm25_equal(left: BM25Index, right: BM25Index) -> None:
    assert left.doc_count == right.doc_count
    assert left.doc_lengths == right.doc_lengths
    assert left.avg_doc_length == pytest.approx(right.avg_doc_length)
    assert left.term_frequencies == right.term_frequencies
    assert left.document_frequency == right.document_frequency
    assert left.idf == right.idf
    # Postings are compared per term rather than as whole dicts: insertion
    # order differs (a term first seen in an added document lands last here and
    # mid-dict there) and that difference is not observable in any ranking.
    assert set(left.postings) == set(right.postings)
    for term in left.postings:
        assert left.postings[term] == right.postings[term]


def test_bm25_add_documents_matches_a_single_fit():
    incremental = BM25Index().fit(DOCS[:2])
    incremental.add_documents(DOCS[2:])
    _assert_bm25_equal(incremental, BM25Index().fit(DOCS))


def test_bm25_add_documents_one_at_a_time_matches_a_single_fit():
    incremental = BM25Index().fit(DOCS[:1])
    for document in DOCS[1:]:
        incremental.add_documents([document])
    _assert_bm25_equal(incremental, BM25Index().fit(DOCS))


def test_bm25_add_documents_refreshes_idf_for_untouched_terms():
    """A new document changes the IDF of terms it does not contain."""
    index = BM25Index().fit(DOCS[:2])
    before = index.idf["fox"]
    index.add_documents([["cosine", "similarity", "only"]])
    assert "fox" not in index.term_frequencies[-1]
    assert index.idf["fox"] != before


def test_bm25_add_documents_is_a_no_op_for_an_empty_batch():
    index = BM25Index().fit(DOCS)
    snapshot = index.to_dict()
    index.add_documents([])
    assert index.to_dict() == snapshot


def test_bm25_add_documents_scores_match_a_fresh_fit():
    incremental = BM25Index().fit(DOCS[:2])
    incremental.add_documents(DOCS[2:])
    fresh = BM25Index().fit(DOCS)
    for query in (["fox"], ["dog", "vocabulary"], ["bm25", "term"]):
        assert incremental.scores(query) == fresh.scores(query)


def _held_out_engines(config: EngineConfig, ngram=None):
    """Two engines over the same 14 documents: one built whole, one grown.

    The held-out document is the one that sorts last, so both engines end up
    with their chunks in the same order. That is the only condition under which
    the two are required to agree exactly - see ``HybridRetriever.add_chunks``.
    """
    documents = read_corpus(CORPUS)
    held_out = sorted(documents)[-1]
    rest = {key: value for key, value in documents.items() if key != held_out}
    grown = RAGEngine(rest, config=config)
    grown.add_document(held_out, documents[held_out])
    whole = RAGEngine(documents, config=config)
    return grown, whole, held_out


def test_engine_add_document_reproduces_a_full_build():
    grown, whole, _ = _held_out_engines(EngineConfig())
    assert [chunk.chunk_id for chunk in grown.chunks] == [
        chunk.chunk_id for chunk in whole.chunks
    ]
    assert grown.documents == whole.documents
    assert grown.stats() == whole.stats()
    _assert_bm25_equal(grown.retriever.bm25, whole.retriever.bm25)


def test_engine_add_document_reproduces_full_build_rankings():
    grown, whole, _ = _held_out_engines(EngineConfig())
    queries = [
        "What does the b parameter control in BM25?",
        "how are overlapping chunks produced",
        "why fuse two ranked lists instead of one",
        "what does a re-ranker read that the first stage did not",
    ]
    for query in queries:
        for method in ("hybrid", "bm25", "vector"):
            assert grown.ranked_doc_ids(query, k=5, method=method) == whole.ranked_doc_ids(
                query, k=5, method=method
            )


def test_engine_add_document_reproduces_full_build_with_the_ngram_list():
    """The third list is refitted too, so it has to agree as well."""
    config = EngineConfig(ngram=4)
    grown, whole, _ = _held_out_engines(config)
    assert grown.retriever.ngram.doc_count == whole.retriever.ngram.doc_count
    for query in ("reranking versus re-ranking", "quantised vectors"):
        assert grown.ranked_doc_ids(query, k=5, method="hybrid") == whole.ranked_doc_ids(
            query, k=5, method="hybrid"
        )


def test_engine_add_document_finds_the_new_document():
    documents = read_corpus(CORPUS)
    engine = RAGEngine(documents)
    engine.add_document(
        "zebra-striping",
        "Zebra striping alternates row shading in a table so that a reader's "
        "eye can follow one row across many columns without losing its place. "
        "It is a presentation choice, not a retrieval one, and it appears in "
        "no other document in this corpus.\n",
    )
    assert "zebra-striping" in engine.ranked_doc_ids(
        "what is zebra striping in a table", k=5
    )


def test_engine_add_document_rejects_a_duplicate_id():
    engine = RAGEngine({"a": "alpha beta gamma delta epsilon.\n"})
    with pytest.raises(ValueError):
        engine.add_document("a", "different text entirely.\n")


def test_engine_add_document_rejects_empty_text():
    engine = RAGEngine({"a": "alpha beta gamma delta epsilon.\n"})
    with pytest.raises(ValueError):
        engine.add_document("b", "   \n  ")


def test_engine_chunks_and_retriever_chunks_stay_in_step():
    engine = RAGEngine({"a": "alpha beta gamma delta epsilon.\n"})
    engine.add_document("b", "zeta eta theta iota kappa.\n")
    assert engine.chunks == engine.retriever.chunks
    assert len(engine.retriever) == len(engine.chunks)


def test_saved_index_round_trips_after_a_document_is_added(tmp_path):
    """The fingerprint covers the grown corpus, so the snapshot stays valid."""
    documents = read_corpus(CORPUS)
    held_out = sorted(documents)[-1]
    rest = {key: value for key, value in documents.items() if key != held_out}
    engine = RAGEngine(rest)
    engine.add_document(held_out, documents[held_out])

    path = str(tmp_path / "index.json")
    engine.save(path)
    restored = RAGEngine.load(path, verify_corpus=documents)

    query = "how does reciprocal rank fusion combine lists"
    assert restored.ranked_doc_ids(query, k=5) == engine.ranked_doc_ids(query, k=5)


def test_retriever_add_chunks_is_a_no_op_for_an_empty_batch():
    chunks = chunk_document("a", "alpha beta gamma delta epsilon zeta eta.\n")
    retriever = HybridRetriever(chunks)
    before = len(retriever)
    retriever.add_chunks([])
    assert len(retriever) == before
