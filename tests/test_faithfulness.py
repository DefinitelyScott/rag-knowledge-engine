import pytest

from ragkb.answerer import Answer, ExtractiveAnswerer
from ragkb.engine import RAGEngine
from ragkb.faithfulness import (
    citation_precision,
    citation_recall,
    evaluate_answer,
    mean_report,
    sentence_grounding,
    supporting_docs,
    token_grounding,
)

PASSAGES = [
    ("bm25", "The b parameter controls length normalisation. Setting b to zero disables it."),
    ("tfidf", "Cosine similarity compares the angle between two weighted vectors."),
]


def test_supporting_docs_finds_the_source_document():
    assert supporting_docs("Setting b to zero disables it.", PASSAGES) == {"bm25"}


def test_supporting_docs_ignores_casing_and_punctuation():
    assert supporting_docs("setting B to zero  disables it", PASSAGES) == {"bm25"}


def test_supporting_docs_does_not_match_inside_a_longer_token():
    passages = [("d", "the cats sat")]
    assert supporting_docs("cat", passages) == set()
    assert supporting_docs("cats", passages) == {"d"}


def test_sentence_grounding_is_one_for_copied_text():
    text = "The b parameter controls length normalisation."
    assert sentence_grounding(text, PASSAGES) == 1.0


def test_sentence_grounding_penalises_an_invented_sentence():
    text = (
        "The b parameter controls length normalisation. "
        "The default value of b is 4.2 in every implementation."
    )
    assert sentence_grounding(text, PASSAGES) == pytest.approx(0.5)


def test_token_grounding_catches_an_invented_term():
    grounded = token_grounding("cosine similarity compares vectors", PASSAGES)
    invented = token_grounding("cosine similarity uses Levenshtein distance", PASSAGES)
    assert grounded == 1.0
    assert invented < 1.0


def test_citation_precision_penalises_citing_an_unused_document():
    text = "The b parameter controls length normalisation."
    assert citation_precision(text, ["bm25"], PASSAGES) == 1.0
    assert citation_precision(text, ["bm25", "tfidf"], PASSAGES) == pytest.approx(0.5)


def test_citation_precision_is_zero_without_citations():
    text = "The b parameter controls length normalisation."
    assert citation_precision(text, [], PASSAGES) == 0.0


def test_citation_recall_flags_an_uncredited_source():
    text = (
        "The b parameter controls length normalisation. "
        "Cosine similarity compares the angle between two weighted vectors."
    )
    assert citation_recall(text, ["bm25", "tfidf"], PASSAGES) == 1.0
    assert citation_recall(text, ["bm25"], PASSAGES) == pytest.approx(0.5)


def test_citation_recall_ignores_ungrounded_sentences():
    # The invented sentence is sentence_grounding's problem, not recall's.
    text = "The b parameter controls length normalisation. Nothing supports this."
    assert citation_recall(text, ["bm25"], PASSAGES) == 1.0


def test_mean_report_averages_each_measure():
    answers = [
        Answer(text="The b parameter controls length normalisation.", citations=["bm25"], method="t"),
        Answer(text="An entirely invented claim about nothing.", citations=["tfidf"], method="t"),
    ]

    class FakeResult:
        def __init__(self, doc_id, text):
            self.doc_id = doc_id
            self.text = text

    results = [FakeResult(doc_id, text) for doc_id, text in PASSAGES]
    reports = [evaluate_answer(answer, results) for answer in answers]
    averaged = mean_report(reports)
    assert averaged["sentence_grounding"] == pytest.approx(0.5)
    assert 0.0 <= averaged["token_grounding"] <= 1.0


def test_extractive_answers_over_the_real_corpus_are_fully_grounded():
    engine = RAGEngine.from_corpus("corpus")
    answerer = ExtractiveAnswerer()
    questions = [
        "what does the b parameter control in bm25",
        "how does reciprocal rank fusion combine ranked lists",
        "why do chunks overlap",
    ]
    for question in questions:
        results = engine.search(question, k=4)
        report = evaluate_answer(answerer.answer(question, results), results)
        assert report.sentence_grounding == 1.0
        assert report.token_grounding == 1.0
        assert report.citation_precision == 1.0
        assert report.citation_recall == 1.0
