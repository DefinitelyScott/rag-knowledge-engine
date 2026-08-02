"""Tests for pseudo-relevance feedback query expansion."""

import json
import os

import pytest

from ragkb.bm25 import BM25Index
from ragkb.engine import RAGEngine
from ragkb.expansion import ExpansionConfig, expand_query, relevance_model
from ragkb.retriever import HybridRetriever
from ragkb.text import Chunk, tokenize
from ragkb.vector import TfidfIndex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
GOLD = os.path.join(ROOT, "evals", "gold.jsonl")


def make_chunks(texts):
    return [
        Chunk(
            doc_id="d{}".format(i),
            chunk_id="d{}#0".format(i),
            text=text,
            token_count=len(tokenize(text)),
            sentence_start=0,
            sentence_end=1,
        )
        for i, text in enumerate(texts)
    ]


# --------------------------------------------------------------------------
# config validation


def test_config_rejects_out_of_range_weight():
    with pytest.raises(ValueError):
        ExpansionConfig(original_weight=1.5)


def test_config_rejects_non_positive_feedback_docs():
    with pytest.raises(ValueError):
        ExpansionConfig(feedback_docs=0)


def test_config_rejects_negative_feedback_terms():
    with pytest.raises(ValueError):
        ExpansionConfig(feedback_terms=-1)


# --------------------------------------------------------------------------
# the query model


def test_query_model_is_a_distribution():
    model = expand_query(
        ["alpha", "beta"], [], [], {}, ExpansionConfig(original_weight=1.0)
    )
    assert model == pytest.approx({"alpha": 0.5, "beta": 0.5})


def test_repeated_query_term_gets_more_weight():
    model = expand_query(
        ["alpha", "alpha", "beta"], [], [], {}, ExpansionConfig(original_weight=1.0)
    )
    assert model["alpha"] > model["beta"]


def test_original_weight_of_one_ignores_feedback():
    documents = [tokenize("gamma gamma delta")]
    model = expand_query(
        ["alpha"],
        [(0, 1.0)],
        documents,
        {"gamma": 2.0, "delta": 2.0},
        ExpansionConfig(feedback_terms=5, original_weight=1.0),
    )
    assert model == {"alpha": 1.0}


def test_zero_feedback_terms_ignores_feedback():
    documents = [tokenize("gamma delta")]
    model = expand_query(
        ["alpha"],
        [(0, 1.0)],
        documents,
        {"gamma": 2.0, "delta": 2.0},
        ExpansionConfig(feedback_terms=0, original_weight=0.5),
    )
    assert model == {"alpha": 1.0}


def test_blended_model_sums_to_one():
    documents = [tokenize("gamma delta epsilon")]
    model = expand_query(
        ["alpha", "beta"],
        [(0, 1.0)],
        documents,
        {"gamma": 2.0, "delta": 1.0, "epsilon": 1.0},
        ExpansionConfig(feedback_terms=3, original_weight=0.6),
    )
    assert sum(model.values()) == pytest.approx(1.0)
    assert model["alpha"] == pytest.approx(0.3)


def test_expansion_terms_are_added_and_query_terms_keep_the_larger_share():
    documents = [tokenize("gamma delta")]
    model = expand_query(
        ["alpha"],
        [(0, 1.0)],
        documents,
        {"gamma": 2.0, "delta": 2.0},
        ExpansionConfig(feedback_terms=2, original_weight=0.6),
    )
    assert set(model) == {"alpha", "gamma", "delta"}
    assert model["alpha"] > model["gamma"]


def test_empty_query_yields_empty_model():
    assert expand_query([], [(0, 1.0)], [["a"]], {"a": 1.0}, ExpansionConfig()) == {}


# --------------------------------------------------------------------------
# the relevance model


def test_relevance_model_prefers_discriminative_terms():
    # "common" appears everywhere so its IDF is low; "rare" should win even
    # though both occur once in the single feedback document.
    documents = [tokenize("common rare"), tokenize("common other")]
    model = relevance_model(
        [(0, 1.0)], documents, {"common": 0.1, "rare": 5.0}, top_terms=2
    )
    assert model["rare"] > model["common"]


def test_relevance_model_drops_stopwords_and_query_terms():
    documents = [tokenize("the alpha beta gamma")]
    model = relevance_model(
        [(0, 1.0)],
        documents,
        {"the": 1.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0},
        exclude=["alpha"],
        top_terms=5,
    )
    assert "the" not in model
    assert "alpha" not in model
    assert set(model) == {"beta", "gamma"}


def test_relevance_model_is_normalised():
    documents = [tokenize("alpha beta gamma")]
    model = relevance_model(
        [(0, 1.0)], documents, {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}, top_terms=2
    )
    assert len(model) == 2
    assert sum(model.values()) == pytest.approx(1.0)


def test_relevance_model_without_feedback_is_empty():
    assert relevance_model([], [["alpha"]], {"alpha": 1.0}) == {}


def test_relevance_model_weights_by_first_pass_score():
    documents = [tokenize("alpha alpha"), tokenize("beta beta")]
    strong_first = relevance_model(
        [(0, 10.0), (1, 1.0)], documents, {"alpha": 1.0, "beta": 1.0}, top_terms=2
    )
    assert strong_first["alpha"] > strong_first["beta"]


# --------------------------------------------------------------------------
# weighted query interfaces on the indexes


def test_bm25_weighted_matches_repeated_tokens():
    documents = [tokenize(text) for text in ("alpha beta", "beta gamma", "alpha alpha")]
    index = BM25Index().fit(documents)
    by_tokens = index.scores(["alpha", "alpha", "beta"])
    by_weights = [
        index.score_document_weighted({"alpha": 2.0, "beta": 1.0}, i)
        for i in range(len(documents))
    ]
    assert by_weights == pytest.approx(by_tokens)


def test_tfidf_weighted_ranking_matches_single_occurrence_tokens():
    documents = [tokenize(text) for text in ("alpha beta", "beta gamma", "alpha gamma")]
    index = TfidfIndex().fit(documents)
    by_tokens = index.search(["alpha", "beta"], k=3)
    by_weights = index.search_weighted({"alpha": 0.5, "beta": 0.5}, k=3)
    assert [i for i, _ in by_tokens] == [i for i, _ in by_weights]


def test_weighted_search_ignores_zero_weights():
    documents = [tokenize(text) for text in ("alpha", "beta")]
    index = TfidfIndex().fit(documents)
    assert index.search_weighted({"alpha": 1.0, "beta": 0.0}, k=2) == index.search(
        ["alpha"], k=2
    )


# --------------------------------------------------------------------------
# retriever integration


def test_expansion_off_is_the_default():
    chunks = make_chunks(["alpha beta gamma", "delta epsilon zeta"])
    assert HybridRetriever(chunks).expansion is None


def test_original_weight_one_reproduces_the_unexpanded_ranking():
    """The correctness check on the whole expansion path.

    With all the weight on the original query, the borrowed terms cannot
    contribute, so every ranking must be identical to the unexpanded one.
    """
    engine = RAGEngine.from_corpus(CORPUS)
    identity = ExpansionConfig(feedback_terms=10, original_weight=1.0)
    with open(GOLD, "r", encoding="utf-8") as handle:
        questions = [json.loads(line)["question"] for line in handle if line.strip()]
    for question in questions:
        assert engine.ranked_doc_ids(question, k=5) == engine.ranked_doc_ids(
            question, k=5, expansion=identity
        )


def test_expansion_borrows_terms_the_question_never_used():
    engine = RAGEngine.from_corpus(CORPUS)
    question = "If someone writes automobiles but the page only ever says cars, what is that?"
    model = engine.retriever.expansion_model(
        question, expansion=ExpansionConfig(feedback_terms=5, original_weight=0.8)
    )
    borrowed = set(model) - set(tokenize(question))
    assert borrowed
    assert all(term not in tokenize(question) for term in borrowed)


def test_expansion_model_is_empty_when_disabled():
    engine = RAGEngine.from_corpus(CORPUS)
    assert engine.retriever.expansion_model("what is bm25") == {}


def test_expansion_changes_at_least_one_ranking():
    engine = RAGEngine.from_corpus(CORPUS)
    config = ExpansionConfig(feedback_terms=5, original_weight=0.8)
    with open(GOLD, "r", encoding="utf-8") as handle:
        questions = [json.loads(line)["question"] for line in handle if line.strip()]
    changed = [
        question
        for question in questions
        if engine.ranked_doc_ids(question, k=5)
        != engine.ranked_doc_ids(question, k=5, expansion=config)
    ]
    assert changed


def test_engine_config_expansion_is_applied():
    from ragkb.engine import EngineConfig

    config = ExpansionConfig(feedback_terms=5, original_weight=0.8)
    engine = RAGEngine.from_corpus(CORPUS, config=EngineConfig(expansion=config))
    plain = RAGEngine.from_corpus(CORPUS)
    question = "Why can what a system knows be changed without any training at all?"
    assert engine.ranked_doc_ids(question, k=5) != plain.ranked_doc_ids(question, k=5)


def test_expansion_recovers_a_question_the_plain_query_misses():
    """h18 is a documented miss at k=5; the borrowed terms find it."""
    engine = RAGEngine.from_corpus(CORPUS)
    question = "Why can what a system knows be changed without any training at all?"
    config = ExpansionConfig(feedback_terms=5, original_weight=0.8)
    assert "rag" not in engine.ranked_doc_ids(question, k=5)
    assert "rag" in engine.ranked_doc_ids(question, k=5, expansion=config)


def test_expansion_composes_with_mmr():
    engine = RAGEngine.from_corpus(CORPUS)
    results = engine.search(
        "how does bm25 saturate term frequency",
        k=5,
        mmr_lambda=0.7,
        expansion=ExpansionConfig(feedback_terms=5, original_weight=0.8),
    )
    assert len(results) == 5
    assert all(result.mmr_rank is not None for result in results)
