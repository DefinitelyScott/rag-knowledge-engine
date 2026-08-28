"""Term-at-a-time ranking over postings must not change any ranking.

Inverting the scoring loop is a pure optimisation, and the only thing that
makes it safe to call one is evidence. So these tests do not check that the top
result still looks sensible - they reconstruct the document-at-a-time scan the
postings list replaced and compare against it on every index and every question
in the gold set.

Two different strengths of claim are asserted, because two different strengths
are true. BM25 accumulates in query-term order either way, so its scores are
compared with ``==``: bit-for-bit. The TF-IDF path's ``similarity`` iterates
whichever vector is shorter and can therefore sum the same products in another
order, so its *ranking* is compared exactly and its scores approximately. Both
are exactness claims; conflating them would overstate one of them.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import pytest

from ragkb.bm25 import BM25Index
from ragkb.engine import EngineConfig, RAGEngine, read_corpus
from ragkb.ngram import CharNgramIndex
from ragkb.text import tokenize
from ragkb.vector import TfidfIndex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
GOLD = os.path.join(ROOT, "evals", "gold.jsonl")

DOCS = [
    "the cache stores recently used embeddings",
    "embeddings are dense vectors of floating point numbers",
    "a cache eviction policy decides what to discard",
    "unrelated prose about gardening and tomatoes",
]
TOKENISED = [tokenize(text) for text in DOCS]


def brute_force_bm25(
    index: BM25Index, weights: Dict[str, float], k: int
) -> List[Tuple[int, float]]:
    """The ranking loop as it was before postings: score every document."""
    scored = [
        (i, index.score_document_weighted(weights, i)) for i in range(index.doc_count)
    ]
    scored = [(i, score) for i, score in scored if score > 0.0]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]


def brute_force_vector(
    index: TfidfIndex, query_vector: Dict[str, float], k: int
) -> List[Tuple[int, float]]:
    """The ranking loop as it was before postings: score every document."""
    if not query_vector:
        return []
    scored = [(i, index.similarity(query_vector, i)) for i in range(index.doc_count)]
    scored = [(i, score) for i, score in scored if score > 0.0]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]


def gold_questions() -> List[str]:
    questions = []
    with open(GOLD, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])
    return questions


# --- the postings list itself ------------------------------------------------


def test_bm25_postings_record_each_document_and_its_term_frequency():
    index = BM25Index().fit(TOKENISED)
    for term, entries in index.postings.items():
        expected = [
            (i, tokens.count(term))
            for i, tokens in enumerate(TOKENISED)
            if term in tokens
        ]
        assert entries == expected, term


def test_bm25_postings_length_is_document_frequency():
    index = BM25Index().fit(TOKENISED)
    assert index.document_frequency == {
        term: len(entries) for term, entries in index.postings.items()
    }


def test_vector_postings_carry_the_normalised_weights_scoring_uses():
    index = TfidfIndex().fit(TOKENISED)
    for term, entries in index.postings.items():
        expected = [
            (i, vector[term])
            for i, vector in enumerate(index.vectors)
            if term in vector
        ]
        assert entries == expected, term


def test_postings_are_in_ascending_document_order():
    index = BM25Index().fit(TOKENISED)
    for entries in index.postings.values():
        assert [doc for doc, _ in entries] == sorted(doc for doc, _ in entries)


def test_candidates_union_the_postings_of_every_query_term():
    index = BM25Index().fit(TOKENISED)
    assert index.candidates(["cache"]) == [0, 2]
    assert index.candidates(["cache", "embeddings"]) == [0, 1, 2]


def test_candidates_ignore_terms_that_are_not_in_the_vocabulary():
    index = BM25Index().fit(TOKENISED)
    assert index.candidates(["quantisation"]) == []
    assert index.candidates(["cache", "quantisation"]) == [0, 2]


def test_candidates_prune_something_on_this_corpus():
    """A guard against the pruning silently degrading to a full scan."""
    index = BM25Index().fit(TOKENISED)
    assert len(index.candidates(tokenize("cache eviction"))) < index.doc_count


# --- exactness against the scan the postings replaced -----------------------


def test_bm25_search_matches_a_full_scan_on_every_gold_question():
    engine = RAGEngine(read_corpus(CORPUS))
    index = engine.retriever.bm25
    for question in gold_questions():
        weights = Counter(tokenize(question))
        assert index.search_weighted(weights, k=10) == brute_force_bm25(
            index, weights, 10
        ), question


def assert_same_ranking(actual, expected, label: str) -> None:
    """Identical order and near-identical scores.

    The order is compared with ``==`` because a re-ordering would be a real
    change; the scores with ``approx`` because summation order is all that can
    differ, and demanding bit-equality of a sum whose order legitimately
    changed would be testing the wrong thing.
    """
    assert [doc for doc, _ in actual] == [doc for doc, _ in expected], label
    for (_, got), (_, want) in zip(actual, expected):
        assert got == pytest.approx(want), label


def test_tfidf_search_matches_a_full_scan_on_every_gold_question():
    engine = RAGEngine(read_corpus(CORPUS))
    index = engine.retriever.vector
    for question in gold_questions():
        tokens = tokenize(question)
        vector = index.vectorize_query(tokens)
        assert_same_ranking(
            index.search(tokens, k=10), brute_force_vector(index, vector, 10), question
        )


def test_ngram_search_matches_a_full_scan_on_every_gold_question():
    """The inherited ranking loop, over the densest index in the project."""
    engine = RAGEngine(read_corpus(CORPUS), config=EngineConfig(ngram=4))
    index = engine.retriever.ngram
    assert isinstance(index, CharNgramIndex)
    for question in gold_questions():
        tokens = tokenize(question)
        vector = index.vectorize_query(tokens)
        assert_same_ranking(
            index.search(tokens, k=10), brute_force_vector(index, vector, 10), question
        )


def test_bm25_scores_still_returns_one_entry_per_document():
    index = BM25Index().fit(TOKENISED)
    tokens = tokenize("cache eviction")
    values = index.scores(tokens)
    assert len(values) == index.doc_count
    expected = [
        index.score_document(tokens, i) for i in range(index.doc_count)
    ]
    assert values == expected


def test_weighted_search_matches_a_full_scan_for_an_expanded_query():
    """Expansion produces fractional weights; pruning must survive them."""
    index = BM25Index().fit(TOKENISED)
    weights = {"cache": 1.0, "embeddings": 0.35, "policy": 0.2}
    assert index.search_weighted(weights, k=10) == brute_force_bm25(index, weights, 10)


def test_zero_weight_terms_do_not_widen_the_candidate_set():
    index = BM25Index().fit(TOKENISED)
    weights = {"cache": 1.0, "gardening": 0.0}
    assert index.search_weighted(weights, k=10) == brute_force_bm25(index, weights, 10)
    assert all(doc != 3 for doc, _ in index.search_weighted(weights, k=10))


def test_query_of_only_unknown_terms_returns_nothing():
    index = BM25Index().fit(TOKENISED)
    assert index.search(tokenize("quantisation gpu kernel"), k=5) == []
    vector_index = TfidfIndex().fit(TOKENISED)
    assert vector_index.search(tokenize("quantisation gpu kernel"), k=5) == []


def test_empty_index_still_answers_without_scanning():
    assert BM25Index().fit([]).search(["cache"], k=5) == []
    assert TfidfIndex().fit([]).search(["cache"], k=5) == []


# --- persistence -------------------------------------------------------------


def test_bm25_round_trip_rebuilds_postings_and_ranks_identically():
    index = BM25Index().fit(TOKENISED)
    restored = BM25Index.from_dict(index.to_dict())
    assert restored.postings == index.postings
    tokens = tokenize("cache eviction policy")
    assert restored.search(tokens, k=5) == index.search(tokens, k=5)


def test_tfidf_round_trip_rebuilds_postings_and_ranks_identically():
    index = TfidfIndex().fit(TOKENISED)
    restored = TfidfIndex.from_dict(index.to_dict())
    assert restored.postings == index.postings
    tokens = tokenize("cache eviction policy")
    assert restored.search(tokens, k=5) == index.search(tokens, k=5)


def test_ngram_round_trip_rebuilds_postings_and_ranks_identically():
    index = CharNgramIndex(n=4).fit(TOKENISED)
    restored = CharNgramIndex.from_dict(index.to_dict())
    assert restored.postings == index.postings
    tokens = tokenize("cache eviction policy")
    assert restored.search(tokens, k=5) == index.search(tokens, k=5)


def test_snapshot_does_not_carry_the_postings_list():
    """Postings are derived on load; storing them would double the file."""
    snapshot = BM25Index().fit(TOKENISED).to_dict()
    assert "postings" not in snapshot


# --- the pipeline on top of it ----------------------------------------------


def test_engine_answers_are_unchanged_across_the_gold_set():
    """End-to-end guard: the fusion consumes these lists, so it inherits them."""
    engine = RAGEngine(read_corpus(CORPUS))
    for question in gold_questions():
        results = engine.search(question, k=5)
        assert results, question
        assert len(results) <= 5


def query_terms(index, question: str):
    """The terms this index will actually look up for ``question``."""
    tokens = tokenize(question)
    if hasattr(index, "vectorize_query"):
        return index.vectorize_query(tokens)
    return Counter(tokens)


def scan_work(index, questions: Sequence[str]) -> int:
    """Lookups a document-at-a-time scan performs: candidates x query terms."""
    total = 0
    for question in questions:
        terms = query_terms(index, question)
        total += len(index.candidates(terms)) * len(terms)
    return total


def postings_work(index, questions: Sequence[str]) -> int:
    """Accumulator updates term-at-a-time performs: sum of the terms' df."""
    total = 0
    for question in questions:
        for term in query_terms(index, question):
            total += len(index.postings.get(term, ()))
    return total


def test_membership_pruning_alone_barely_helps_on_real_questions():
    """The honest half of the result, pinned so it cannot quietly be reframed.

    A question containing one common word reaches almost every chunk through
    that word alone, so skipping documents that contain *no* query term is
    correct and nearly free of benefit. This is why the ranking loop was
    inverted rather than merely filtered.
    """
    engine = RAGEngine(read_corpus(CORPUS))
    questions = gold_questions()
    index = engine.retriever.bm25
    covered = sum(len(index.candidates(query_terms(index, q))) for q in questions)
    fraction = covered / (len(questions) * index.doc_count)
    assert fraction > 0.75


def test_term_at_a_time_does_less_work_than_the_scan_it_replaced():
    engine = RAGEngine(read_corpus(CORPUS), config=EngineConfig(ngram=4))
    questions = gold_questions()
    for index in (
        engine.retriever.bm25,
        engine.retriever.vector,
        engine.retriever.ngram,
    ):
        scan = scan_work(index, questions)
        postings = postings_work(index, questions)
        assert postings < scan, type(index).__name__
