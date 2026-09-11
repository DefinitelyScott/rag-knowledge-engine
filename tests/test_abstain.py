"""Tests for the abstention gate."""

from __future__ import annotations

import os

import pytest

from ragkb.abstain import (
    REFUSAL,
    AbstentionConfig,
    AnswerabilityReport,
    assess,
    query_terms,
)
from ragkb.engine import EngineConfig, RAGEngine

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")

ANSWERABLE = "what does the b parameter control in bm25"
UNANSWERABLE = "how do I configure a kubernetes ingress controller to terminate tls"


@pytest.fixture(scope="module")
def engine():
    return RAGEngine.from_corpus(CORPUS, config=EngineConfig())


def report_for(engine, query, config=None):
    results = engine.search(query, k=4)
    return assess(engine.retriever, query, results, config=config)


def test_config_normalises_weights():
    config = AbstentionConfig(vocab_weight=2.0, match_weight=2.0, similarity_weight=0.0)
    total = config.vocab_weight + config.match_weight + config.similarity_weight
    assert total == pytest.approx(1.0)
    assert config.similarity_weight == 0.0


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        AbstentionConfig(threshold=1.5)
    with pytest.raises(ValueError):
        AbstentionConfig(vocab_weight=-1.0)
    with pytest.raises(ValueError):
        AbstentionConfig(vocab_weight=0.0, match_weight=0.0, similarity_weight=0.0)


def test_signals_stay_in_unit_interval(engine):
    for query in (ANSWERABLE, UNANSWERABLE, "the and of"):
        report = report_for(engine, query)
        assert 0.0 <= report.score <= 1.0
        assert 0.0 <= report.vocab_coverage <= 1.0
        assert 0.0 <= report.match_coverage <= 1.0
        assert 0.0 <= report.top_similarity <= 1.0


def test_in_domain_scores_above_out_of_domain(engine):
    """The ordering is the property worth asserting; the exact scores are the
    eval's business, not the test's."""
    assert report_for(engine, ANSWERABLE).score > report_for(engine, UNANSWERABLE).score


def test_shipped_threshold_separates_the_two(engine):
    assert report_for(engine, ANSWERABLE).answerable is True
    assert report_for(engine, UNANSWERABLE).answerable is False


def test_unknown_terms_are_reported(engine):
    report = report_for(engine, UNANSWERABLE)
    assert "kubernetes" in report.unknown_terms
    # Reported once even though the query may repeat it.
    assert len(report.unknown_terms) == len(set(report.unknown_terms))


def test_in_vocabulary_question_reports_no_unknown_terms(engine):
    assert report_for(engine, "how does reciprocal rank fusion work").unknown_terms == []


def test_morphology_counts_as_unknown():
    """A documented weakness, pinned so it cannot change silently.

    Tokenisation does no stemming, so a query saying "control" against a corpus
    that only ever says "controls" charges that term the maximum IDF and pushes
    the question toward abstention. The character n-gram index
    (:mod:`ragkb.ngram`) is the piece that already handles this for retrieval;
    the gate does not yet consult it.

    Run against a two-document corpus written for the purpose rather than
    against the shipped one. This test used to name "control" and assert it was
    unknown to the real corpus, which held only until some later document
    happened to use the singular - a corpus-growth increment then failed it
    without anything about stemming having changed. A property of the tokeniser
    should not be hostage to which words twenty-four documents happen to
    contain, so the condition is now constructed instead of found.
    """
    documents = {
        "knobs": (
            "The b parameter controls how strongly document length is used. "
            "Two knobs govern the ranking function and both have defaults."
        ),
        "other": "An unrelated document about chunking and overlap.",
    }
    engine = RAGEngine(documents, config=EngineConfig())
    report = report_for(engine, "what does the b parameter control")
    # "controls" is indexed; "control" is a different token entirely.
    assert "control" in report.unknown_terms
    assert "controls" not in report.unknown_terms


def test_one_unknown_term_costs_the_score_without_sinking_it(engine):
    """The other half of the weakness: it is a penalty, not a veto."""
    report = report_for(engine, ANSWERABLE + " frobnicate")
    assert "frobnicate" in report.unknown_terms
    assert report.answerable is True


def test_threshold_of_zero_never_abstains(engine):
    config = AbstentionConfig(threshold=0.0)
    assert report_for(engine, UNANSWERABLE, config).answerable is True


def test_threshold_of_one_always_abstains(engine):
    config = AbstentionConfig(threshold=1.0)
    assert report_for(engine, ANSWERABLE, config).answerable is False


def test_no_results_abstains(engine):
    report = assess(engine.retriever, ANSWERABLE, [])
    assert report.answerable is False
    assert report.score == 0.0


def test_empty_query_abstains(engine):
    report = assess(engine.retriever, "", engine.search(ANSWERABLE, k=4))
    assert report.answerable is False


def test_query_terms_falls_back_to_stopwords():
    assert query_terms("what is the") != []


def test_explain_mentions_the_verdict_and_the_numbers(engine):
    text = report_for(engine, UNANSWERABLE).explain()
    assert "abstain" in text
    assert "threshold" in text
    assert "kubernetes" in text


def test_engine_answers_normally_when_the_gate_is_off(engine):
    answer = engine.answer(UNANSWERABLE, k=4)
    assert answer.abstained is False
    assert answer.answerability is None
    assert answer.citations


def test_engine_abstains_when_the_gate_is_on(engine):
    answer = engine.answer(UNANSWERABLE, k=4, abstain=AbstentionConfig())
    assert answer.abstained is True
    assert answer.citations == []
    assert answer.text == REFUSAL
    assert answer.method == "abstained"
    assert isinstance(answer.answerability, AnswerabilityReport)


def test_engine_still_answers_in_domain_with_the_gate_on(engine):
    answer = engine.answer(ANSWERABLE, k=4, abstain=AbstentionConfig())
    assert answer.abstained is False
    assert answer.citations
    assert answer.answerability is not None


def test_gate_can_be_configured_on_the_engine():
    configured = RAGEngine.from_corpus(
        CORPUS, config=EngineConfig(abstain=AbstentionConfig())
    )
    assert configured.answer(UNANSWERABLE, k=4).abstained is True
    assert configured.answer(ANSWERABLE, k=4).abstained is False


def test_answerability_is_reachable_without_generating(engine):
    report = engine.answerability(UNANSWERABLE, k=4)
    assert isinstance(report, AnswerabilityReport)
    assert report.answerable is False


def test_gate_off_is_bit_for_bit_the_previous_behaviour(engine):
    """A threshold of 0.0 must return exactly the answer the ungated engine
    gives, so turning the gate on can only ever remove answers."""
    plain = engine.answer(ANSWERABLE, k=4)
    gated = engine.answer(ANSWERABLE, k=4, abstain=AbstentionConfig(threshold=0.0))
    assert gated.text == plain.text
    assert gated.citations == plain.citations
