import json
import os

import pytest

from ragkb.answerer import Answer, ExtractiveAnswerer, OpenAIAnswerer
from ragkb.engine import EngineConfig, RAGEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
GOLD = os.path.join(ROOT, "evals", "gold.jsonl")


@pytest.fixture(scope="module")
def engine():
    return RAGEngine.from_corpus(CORPUS)


def test_corpus_loads_and_chunks(engine):
    stats = engine.stats()
    assert stats["documents"] >= 10
    assert stats["chunks"] > stats["documents"]
    assert stats["max_chunk_tokens"] < 400


def test_search_returns_cited_chunks(engine):
    results = engine.search("what does the b parameter control in bm25", k=3)
    assert results
    assert results[0].doc_id == "bm25"
    assert all(r.chunk.chunk_id.startswith(r.doc_id) for r in results)


def test_answer_is_extracted_from_the_corpus(engine):
    answer = engine.answer("how does reciprocal rank fusion combine ranked lists")
    assert isinstance(answer, Answer)
    assert answer.citations
    assert "rank-fusion" in answer.citations
    for sentence in answer.text.split(". "):
        stem = sentence.strip().rstrip(".")
        if len(stem) > 30:
            assert any(stem[:30] in engine.documents[doc] for doc in engine.documents)


def test_gold_questions_are_answerable_end_to_end(engine):
    with open(GOLD, "r", encoding="utf-8") as handle:
        gold = [json.loads(line) for line in handle if line.strip()]
    assert len(gold) >= 20
    assert all(
        doc in engine.documents for record in gold for doc in record["relevant"]
    )
    hits = sum(
        1
        for record in gold
        if set(engine.ranked_doc_ids(record["question"], k=3)) & set(record["relevant"])
    )
    # Guardrail, not a target: the real numbers live in evals/evaluate.py.
    assert hits / len(gold) >= 0.7


def test_ranked_doc_ids_are_unique(engine):
    ids = engine.ranked_doc_ids("chunk overlap and boundary loss", k=4)
    assert len(ids) == len(set(ids))


def test_empty_query_returns_a_polite_answer(engine):
    answer = engine.answer("   ")
    assert answer.citations == []
    assert "No relevant passage" in answer.text


def test_custom_config_changes_chunking():
    small = RAGEngine.from_corpus(CORPUS, config=EngineConfig(target_tokens=60, overlap_tokens=15))
    default = RAGEngine.from_corpus(CORPUS)
    assert small.stats()["chunks"] > default.stats()["chunks"]


def test_missing_corpus_raises():
    with pytest.raises(FileNotFoundError):
        RAGEngine.from_corpus(os.path.join(ROOT, "does-not-exist"))


def test_openai_answerer_requires_a_key(monkeypatch, engine):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answerer = OpenAIAnswerer()
    assert not answerer.available
    with pytest.raises(RuntimeError):
        answerer.answer("anything", engine.search("bm25", k=1))


def test_openai_prompt_numbers_the_sources(engine):
    results = engine.search("bm25 length normalisation", k=2)
    prompt = OpenAIAnswerer(api_key="test-key").build_prompt("why?", results)
    assert "[1]" in prompt and "[2]" in prompt
    assert "Question: why?" in prompt


def test_extractive_answerer_respects_sentence_budget(engine):
    answerer = ExtractiveAnswerer(max_sentences=1)
    answer = answerer.answer("what is a posting list", engine.search("posting list", k=3))
    assert answer.text.count(".") <= 2
    assert answer.method == "extractive"
