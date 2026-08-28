"""Tests for the failure analysis in ``evals/failures.py``.

The script is a diagnostic instrument, so what has to be true of it is not that
any particular question fails but that it describes failure honestly: it must
report a pair only when the method really missed it, place the missed document
at its real rank rather than a truncated one, and never claim a lever recovered
something it did not. A diagnostic that flatters the system it measures is
worse than no diagnostic, because it is trusted.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from ragkb.engine import EngineConfig, RAGEngine

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "failures", os.path.join(ROOT, "evals", "failures.py")
)
failures = importlib.util.module_from_spec(_spec)
# Registered before execution because the module defines a dataclass, and
# ``dataclasses`` resolves annotations through ``sys.modules[cls.__module__]``.
sys.modules[_spec.name] = failures
_spec.loader.exec_module(failures)


CORPUS = {
    "alpha": (
        "Alpha discusses reciprocal rank fusion and how two ranked lists are "
        "combined into one ordering without comparing their raw scores."
    ),
    "beta": (
        "Beta is about chunking long documents into overlapping passages so "
        "that a sentence is never cut in half at a boundary."
    ),
    "gamma": (
        "Gamma covers tokenisation, stopwords, and the way punctuation is "
        "stripped before any term statistics are collected."
    ),
}


@pytest.fixture()
def engine() -> RAGEngine:
    return RAGEngine(CORPUS, config=EngineConfig(target_tokens=40, overlap_tokens=10))


def test_query_coverage_is_a_share_of_the_question_terms() -> None:
    assert failures.query_coverage("rank fusion", "reciprocal rank fusion") == 1.0
    assert failures.query_coverage("", "anything") == 0.0
    assert failures.query_coverage("rank fusion", "rank only") == pytest.approx(0.5)


def test_query_coverage_ignores_document_terms_the_question_never_used() -> None:
    """Coverage is an upper bound on what the *query* could match, so padding
    the document with unrelated text must not move it."""
    short = failures.query_coverage("chunking overlap", "chunking and overlap")
    padded = failures.query_coverage(
        "chunking overlap", "chunking and overlap " + "unrelated " * 50
    )
    assert short == padded == 1.0


def test_classify_blames_the_fusion_only_when_a_component_had_it() -> None:
    assert (
        failures.classify(9, {"bm25": 3, "vector": 12}, k=5) == failures.FUSION_LOSS
    )
    # Same fused rank, but neither component found it either: not the fusion.
    assert failures.classify(9, {"bm25": 9, "vector": 9}, k=5) == failures.NEAR_MISS


def test_classify_separates_ordering_problems_from_matching_problems() -> None:
    assert failures.classify(6, {"bm25": 6}, k=5) == failures.NEAR_MISS
    assert failures.classify(10, {"bm25": 10}, k=5) == failures.NEAR_MISS
    assert failures.classify(11, {"bm25": 11}, k=5) == failures.DEEP_MISS
    assert failures.classify(None, {"bm25": None}, k=5) == failures.UNRANKED


def test_document_ranks_are_dense_and_one_based(engine: RAGEngine) -> None:
    ranks = failures.document_ranks(engine, "reciprocal rank fusion", "hybrid")
    assert ranks
    assert sorted(ranks.values()) == list(range(1, len(ranks) + 1))


def test_a_question_the_method_answers_is_not_reported(engine: RAGEngine) -> None:
    gold = [
        {
            "id": "t1",
            "question": "how are two ranked lists combined",
            "relevant": ["alpha"],
            "difficulty": "easy",
        }
    ]
    assert failures.analyse(engine, gold, k=1) == []


def test_a_missed_pair_is_reported_with_its_real_rank(engine: RAGEngine) -> None:
    """The question is squarely about ``alpha``; asking for ``gamma`` forces a
    miss whose rank has to be reported as the rank it actually holds."""
    gold = [
        {
            "id": "t2",
            "question": "how are two ranked lists combined into one ordering",
            "relevant": ["gamma"],
            "difficulty": "hard",
        }
    ]
    found = failures.analyse(engine, gold, k=1)
    assert len(found) == 1
    failure = found[0]
    assert failure.question_id == "t2"
    assert failure.doc_id == "gamma"
    assert failure.rank is None or failure.rank > 1
    assert failure.displaced_by and "gamma" not in failure.displaced_by
    assert failure.category in failures.CATEGORIES
    assert 0.0 <= failure.coverage <= 1.0


def test_multi_document_questions_report_one_row_per_missed_document(
    engine: RAGEngine,
) -> None:
    gold = [
        {
            "id": "t3",
            "question": "how are two ranked lists combined into one ordering",
            "relevant": ["beta", "gamma"],
            "difficulty": "hard",
        }
    ]
    found = failures.analyse(engine, gold, k=1)
    assert {failure.doc_id for failure in found} == {"beta", "gamma"}
    assert all(failure.question_id == "t3" for failure in found)


def test_recovery_only_credits_a_lever_that_reaches_inside_k() -> None:
    failure = failures.Failure(
        question_id="t4",
        difficulty="hard",
        question="q",
        doc_id="d",
        rank=9,
        recovery={"expansion": 3, "rerank": 9, "ngram-only": None},
    )
    assert failure.recovered_by(k=5) == ["expansion"]
    assert failure.recovered_by(k=2) == []


def test_recovery_table_marks_and_totals_what_it_recovered() -> None:
    rows = [
        failures.Failure(
            question_id="t5",
            difficulty="hard",
            question="q",
            doc_id="d",
            rank=9,
            recovery={"expansion": 3, "rerank": 9},
        )
    ]
    table = failures.format_recovery(rows, ["expansion", "rerank"], k=5)
    assert "3*" in table
    assert "9*" not in table
    assert "1/1" in table and "0/1" in table


def test_levers_cover_the_index_time_and_query_time_halves() -> None:
    built = failures.build_levers(CORPUS, EngineConfig())
    names = [name for name, _, _ in built]
    assert "ngram-only" in names and "expansion" in names
    lookup = {name: engine for name, engine, _ in built}
    # The query-time levers must share one fitted index; refitting per lever
    # would make their ranks incomparable with the baseline's. The two n-gram
    # rows need a third index, so they get their own engine - and share it.
    query_time = {id(lookup[name]) for name in ("expansion", "rerank", "mmr")}
    index_time = {id(lookup[name]) for name in ("hybrid3", "ngram-only")}
    assert len(query_time) == 1
    assert len(index_time) == 1
    assert query_time != index_time


def test_analysis_runs_end_to_end_with_recovery(engine: RAGEngine) -> None:
    gold = [
        {
            "id": "t6",
            "question": "how are two ranked lists combined into one ordering",
            "relevant": ["gamma"],
            "difficulty": "hard",
        }
    ]
    levers = failures.build_levers(
        CORPUS, EngineConfig(target_tokens=40, overlap_tokens=10)
    )
    found = failures.analyse(engine, gold, k=1, levers=levers)
    assert len(found) == 1
    assert set(found[0].recovery) == {name for name, _, _ in levers}


def test_main_reports_on_the_bundled_gold_set(capsys: pytest.CaptureFixture) -> None:
    assert failures.main(["--no-recovery", "--slice", "hard"]) == 0
    out = capsys.readouterr().out
    assert "missed at k=5" in out
    assert "failures by category" in out
