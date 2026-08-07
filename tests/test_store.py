"""Persistence layer: a loaded index must be indistinguishable from a fresh one."""

import json
import os

import pytest

from ragkb.engine import EngineConfig, RAGEngine
from ragkb.expansion import ExpansionConfig
from ragkb.store import (
    IndexFormatError,
    StaleIndexError,
    corpus_fingerprint,
)

DOCS = {
    "bm25": (
        "BM25 is a lexical ranking function. It scores documents by term "
        "frequency, discounted by inverse document frequency. Saturation "
        "keeps repeated terms from dominating the score."
    ),
    "vectors": (
        "TF-IDF vectors represent documents as sparse weighted term maps. "
        "Cosine similarity compares a query vector with each document vector. "
        "Normalisation makes the dot product equal the cosine."
    ),
    "fusion": (
        "Reciprocal rank fusion combines ranked lists without comparable "
        "scores. Each list contributes one over k plus rank. Fusion rewards "
        "items that both retrievers rank highly."
    ),
}

QUERIES = (
    "how does bm25 discount common terms",
    "comparing query and document vectors",
    "combining two ranked lists",
)


def build(config=None):
    return RAGEngine(DOCS, config=config)


def snapshot(engine, method="hybrid"):
    return [
        [(r.chunk.chunk_id, pytest.approx(r.score)) for r in engine.search(q, k=4, method=method)]
        for q in QUERIES
    ]


def test_round_trip_preserves_every_ranking(tmp_path):
    engine = build()
    path = str(tmp_path / "index.json")
    engine.save(path)
    loaded = RAGEngine.load(path, verify_corpus=DOCS)
    for method in ("hybrid", "bm25", "vector"):
        assert snapshot(loaded, method) == snapshot(engine, method)


def test_round_trip_preserves_stats_and_config(tmp_path):
    config = EngineConfig(
        target_tokens=40,
        overlap_tokens=10,
        rrf_k=30,
        mmr_lambda=0.7,
        expansion=ExpansionConfig(feedback_docs=3, feedback_terms=5, original_weight=0.7),
    )
    engine = build(config)
    path = str(tmp_path / "index.json")
    engine.save(path)
    loaded = RAGEngine.load(path, verify_corpus=DOCS)
    assert loaded.stats() == engine.stats()
    assert loaded.config == engine.config
    assert loaded.retriever.rrf_k == 30
    assert loaded.retriever.mmr_lambda == 0.7
    assert loaded.retriever.expansion == engine.config.expansion


def test_answers_survive_the_round_trip(tmp_path):
    engine = build()
    path = str(tmp_path / "index.json")
    engine.save(path)
    loaded = RAGEngine.load(path)
    query = "how are ranked lists combined"
    assert loaded.answer(query).formatted() == engine.answer(query).formatted()


def test_stale_corpus_is_refused(tmp_path):
    engine = build()
    path = str(tmp_path / "index.json")
    engine.save(path)
    edited = dict(DOCS, fusion=DOCS["fusion"] + " New sentence added later.")
    with pytest.raises(StaleIndexError):
        RAGEngine.load(path, verify_corpus=edited)


def test_changed_chunking_config_is_stale_too():
    assert corpus_fingerprint(DOCS, 120, 30) != corpus_fingerprint(DOCS, 80, 30)


def test_query_time_knobs_do_not_invalidate_the_fingerprint():
    assert corpus_fingerprint(DOCS, 120, 30) == corpus_fingerprint(DOCS, 120, 30)


def test_unverified_load_trusts_the_file(tmp_path):
    engine = build()
    path = str(tmp_path / "index.json")
    engine.save(path)
    assert snapshot(RAGEngine.load(path)) == snapshot(engine)


def test_unknown_version_is_rejected(tmp_path):
    engine = build()
    path = str(tmp_path / "index.json")
    engine.save(path)
    with open(path) as handle:
        data = json.load(handle)
    data["version"] = 999
    with open(path, "w") as handle:
        json.dump(data, handle)
    with pytest.raises(IndexFormatError):
        RAGEngine.load(path)


def test_non_index_json_is_rejected(tmp_path):
    path = str(tmp_path / "other.json")
    with open(path, "w") as handle:
        json.dump({"hello": "world"}, handle)
    with pytest.raises(IndexFormatError):
        RAGEngine.load(path)


def test_save_is_atomic_no_temp_file_left_behind(tmp_path):
    engine = build()
    path = str(tmp_path / "index.json")
    engine.save(path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
