"""Tests for the index-time parameter sweep in ``evals/tune.py``.

The sweep is a measurement instrument, so the tests here are about the
instrument being trustworthy rather than about any particular number it
produces: the default row has to reproduce the engine's shipped behaviour
exactly, an index-time knob has to actually rebuild the index, a query-time knob
has to leave it alone, and invalid chunking has to be refused rather than
silently producing nonsense.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "tune", os.path.join(ROOT, "evals", "tune.py")
)
tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune)

from ragkb.engine import EngineConfig, RAGEngine, read_corpus  # noqa: E402
from ragkb.metrics import aggregate  # noqa: E402


@pytest.fixture(scope="module")
def documents():
    return read_corpus(os.path.join(ROOT, "corpus"))


@pytest.fixture(scope="module")
def gold():
    return tune.load_gold(os.path.join(ROOT, "evals", "gold.jsonl"))


def test_parse_chunk_specs_reads_pairs():
    assert tune.parse_chunk_specs("120:30, 150:40") == [(120, 30), (150, 40)]


def test_parse_chunk_specs_defaults_missing_overlap_to_zero():
    assert tune.parse_chunk_specs("120") == [(120, 0)]


@pytest.mark.parametrize("spec", ["120:120", "120:200", "0:0", "-5:1"])
def test_parse_chunk_specs_rejects_impossible_chunking(spec):
    """An overlap >= the target repeats forever; a non-positive target is empty."""
    with pytest.raises(ValueError):
        tune.parse_chunk_specs(spec)


def test_default_chunk_row_matches_the_shipped_engine(documents, gold):
    """The starred row has to be the engine as it actually ships.

    This is the identity check for the whole sweep: if the default row drifts
    from a plain ``RAGEngine``, every delta in the table is measured against
    the wrong baseline.
    """
    rows = tune.sweep_chunking(
        documents,
        gold,
        [(tune.DEFAULT_CHUNK_TOKENS, tune.DEFAULT_OVERLAP_TOKENS)],
        k=5,
    )
    label, report = rows[0]
    assert label.endswith("*")

    engine = RAGEngine(documents)
    rankings = [engine.ranked_doc_ids(r["question"], k=5) for r in gold]
    expected = aggregate(rankings, [r["relevant"] for r in gold], ks=(1, 3, 5))
    for column in tune.COLUMNS:
        assert report[column] == pytest.approx(expected[column])


def test_chunking_sweep_actually_rebuilds_the_index(documents, gold):
    """Different chunk targets must produce different corpus shapes.

    A sweep that reported the same chunk count for every setting would be
    measuring one index under several labels.
    """
    rows = tune.sweep_chunking(documents, gold, [(60, 15), (200, 50)], k=5)
    small, large = rows[0][1], rows[1][1]
    assert small["chunks"] > large["chunks"]
    assert small["mean_tok"] < large["mean_tok"]


def test_zero_overlap_is_a_legal_setting(documents, gold):
    """Overlap is a design claim, so the sweep has to be able to turn it off."""
    rows = tune.sweep_chunking(documents, gold, [(120, 0)], k=5)
    assert rows[0][1]["chunks"] > 0


def test_rrf_sweep_restores_the_engine(documents, gold):
    """``rrf_k`` is mutated in place, so the engine must come back unchanged."""
    engine = RAGEngine(documents)
    before = engine.retriever.rrf_k
    tune.sweep_rrf_k(engine, gold[:5], [1, 60, 1000], k=5)
    assert engine.retriever.rrf_k == before


def test_rrf_sweep_reports_zero_changes_at_the_default(documents, gold):
    """The default row is compared against itself, so nothing can differ."""
    engine = RAGEngine(documents)
    rows = tune.sweep_rrf_k(engine, gold, [tune.DEFAULT_RRF_K], k=5)
    label, report = rows[0]
    assert label.endswith("*")
    assert report["changed"] == 0


def test_rrf_changed_column_tracks_real_ordering_differences(documents, gold):
    """``changed`` must count orderings, not accuracy.

    ``rrf_k`` is close to inert on this corpus - the accuracy columns barely
    move - so a column that only reported accuracy could not tell "equally
    good" from "identical". This asserts the distinction is measurable: some
    setting has to disagree with the default somewhere.

    The second assertion used to be ``rows["1000"]["changed"] == 0``. At forty
    documents ``rrf_k = 1000`` reorders one question (``h99``, two irrelevant
    documents swapping ranks 4 and 5), so that pinned the corpus rather than
    the code. What the column has to show is weaker and corpus-independent in
    practice: a setting that flattens the fusion towards the default reorders
    far fewer questions than one that sharpens it to near rank-only voting.
    """
    engine = RAGEngine(documents)
    rows = dict(tune.sweep_rrf_k(engine, gold, [1, 1000], k=5))
    assert rows["1"]["changed"] > 0
    assert rows["1000"]["changed"] < rows["1"]["changed"]


def test_rrf_k_does_not_change_the_index(documents, gold):
    """A query-time knob must not disturb chunks or fitted statistics."""
    engine = RAGEngine(documents)
    chunks_before = list(engine.chunks)
    tune.sweep_rrf_k(engine, gold[:5], [1, 200], k=5)
    assert engine.chunks == chunks_before


def test_format_table_marks_deltas_against_the_baseline():
    rows = [
        ("a", {"chunks": 10.0, "hit@1": 0.5, "mrr": 0.4}),
        ("b*", {"chunks": 20.0, "hit@1": 0.6, "mrr": 0.4}),
    ]
    text = tune.format_table(rows, "label", baseline=rows[1][1])
    assert "-0.100" in text  # row a is 0.1 below the baseline on hit@1
    assert "+" not in text.split("\n")[2]  # ...and never above it
    assert "10" in text  # shape columns print plain, with no delta


def test_main_runs_a_small_sweep(documents, capsys):
    """End-to-end smoke test over the real corpus and gold set."""
    exit_code = tune.main(
        [
            "--corpus",
            os.path.join(ROOT, "corpus"),
            "--gold",
            os.path.join(ROOT, "evals", "gold.jsonl"),
            "--chunk-specs",
            "120:30,150:40",
            "--rrf-ks",
            "1,60",
            "--slice",
            "hard",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "chunking" in out and "rrf_k" in out
    assert "120:30*" in out and "60*" in out
    assert "hard slice" in out


def test_engine_defaults_are_the_ones_the_sweep_baselines_against():
    """Guard against the sweep and the engine drifting apart."""
    config = EngineConfig()
    assert config.target_tokens == tune.DEFAULT_CHUNK_TOKENS
    assert config.overlap_tokens == tune.DEFAULT_OVERLAP_TOKENS
    assert config.rrf_k == tune.DEFAULT_RRF_K
