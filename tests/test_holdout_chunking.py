"""Tests for the index-time sweep added to ``evals/holdout.py``.

The three sweeps that harness already carried are query-time: one fitted engine
serves every configuration. Chunking is not, and the whole value of adding it is
that its configurations are genuinely different indexes. So these tests check
the two properties that would make the new row meaningless if either failed -
the default configuration has to be the engine as it actually ships, and the
other configurations have to actually differ from it - plus the offer-order
contract the tie-breaking in :mod:`ragkb.holdout` depends on.

They deliberately do not assert any optimism value. What that number *is* is a
property of the corpus and moves with it; what the harness must not do is report
one computed from an index it did not really rebuild.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "evals_holdout", os.path.join(ROOT, "evals", "holdout.py")
)
holdout_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(holdout_eval)

from ragkb.engine import EngineConfig, RAGEngine, read_corpus  # noqa: E402
from ragkb.text import parse_chunk_specs  # noqa: E402

DEFAULT_SPEC = (EngineConfig.target_tokens, EngineConfig.overlap_tokens)
DEFAULT_LABEL = "{}:{}".format(*DEFAULT_SPEC)


@pytest.fixture(scope="module")
def documents():
    return read_corpus(os.path.join(ROOT, "corpus"))


@pytest.fixture(scope="module")
def questions():
    gold = holdout_eval.load_gold(os.path.join(ROOT, "evals", "gold.jsonl"))
    return [record["question"] for record in gold[:20]]


# --------------------------------------------------------------------------
# the parser moved to ragkb.text; tune.py must keep working through it
# --------------------------------------------------------------------------


def test_tune_re_exports_the_moved_parser():
    """``evals/tune.py`` used to own this function and its tests still call it.

    Two eval scripts now sweep chunking, so the parser lives beside the chunker
    it validates. The re-export is what keeps that a move rather than a break.
    """
    tune_spec = importlib.util.spec_from_file_location(
        "tune_for_holdout_test", os.path.join(ROOT, "evals", "tune.py")
    )
    tune = importlib.util.module_from_spec(tune_spec)
    tune_spec.loader.exec_module(tune)
    assert tune.parse_chunk_specs is parse_chunk_specs


# --------------------------------------------------------------------------
# offer order
# --------------------------------------------------------------------------


def test_default_leads_the_offer_order_even_when_not_requested(documents):
    """``vs default`` is undefined without the default in the table.

    ``cross_validated_selection`` also breaks exact ties on offer order, so the
    shipped setting going first is what makes a tie resolve to "do not move".
    """
    name, default_label, labels, _ = holdout_eval.chunking_sweep(
        documents, parse_chunk_specs("80:20,150:40")
    )
    assert name == "chunking"
    assert default_label == DEFAULT_LABEL
    assert labels[0] == DEFAULT_LABEL
    assert labels == [DEFAULT_LABEL, "80:20", "150:40"]


def test_default_is_not_offered_twice(documents):
    """A caller who lists the default explicitly must not get a duplicate column."""
    _, _, labels, _ = holdout_eval.chunking_sweep(
        documents, parse_chunk_specs("80:20,120:30,150:40")
    )
    assert labels.count(DEFAULT_LABEL) == 1
    assert labels[0] == DEFAULT_LABEL


# --------------------------------------------------------------------------
# the configurations have to be real indexes
# --------------------------------------------------------------------------


def test_default_row_reproduces_the_shipped_engine(documents, questions):
    """The baseline column must be a plain ``RAGEngine``, not an approximation.

    Every optimism and ``vs default`` figure in the chunking rows is measured
    against this label. If it drifts from the shipped engine the whole row is
    scored against a system nobody runs.
    """
    _, _, _, rank = holdout_eval.chunking_sweep(
        documents, parse_chunk_specs("80:20,150:40")
    )
    shipped = RAGEngine(documents)
    for question in questions:
        assert rank(DEFAULT_LABEL, question, 5) == shipped.ranked_doc_ids(
            question, k=5, method="hybrid"
        )


def test_each_spec_is_a_separately_built_index(documents, questions):
    """Distinct chunk settings must produce distinct chunk counts.

    This is the property that separates an index-time sweep from a query-time
    one. If the sweep quietly reused one engine, every row would be identical
    and the harness would report exactly zero optimism - a clean-looking result
    that meant nothing.
    """
    sweep = holdout_eval.chunking_sweep(documents, parse_chunk_specs("60:15,200:50"))
    _, _, labels, rank = sweep
    counts = set()
    for label in labels:
        target, _, overlap = label.partition(":")
        engine = RAGEngine(
            documents,
            config=EngineConfig(
                target_tokens=int(target), overlap_tokens=int(overlap)
            ),
        )
        counts.add(len(engine.chunks))
    assert len(counts) == len(labels), "chunk settings collapsed to one index"

    # ...and the rankings actually differ somewhere, so the selection has
    # something to select between.
    baseline = [rank(DEFAULT_LABEL, question, 5) for question in questions]
    assert any(
        [rank(label, question, 5) for question in questions] != baseline
        for label in labels
        if label != DEFAULT_LABEL
    )


def test_rejects_impossible_chunking_before_building_anything(documents):
    with pytest.raises(ValueError):
        holdout_eval.chunking_sweep(documents, parse_chunk_specs("120:120"))


# --------------------------------------------------------------------------
# the flag
# --------------------------------------------------------------------------


def test_chunking_runs_by_default_and_no_chunking_skips_it(capsys):
    """Cheap settings: the flag is the subject, not the numbers."""
    argv = [
        "--metrics",
        "hit@1",
        "--repeats",
        "2",
        "--folds",
        "3",
        "--rrf-ks",
        "60",
        "--fusion-weights",
        "ngram=4",
        "--mmr-lambdas",
        "0.7",
        "--chunk-specs",
        "120:0",
    ]
    assert holdout_eval.main(argv) == 0
    assert "chunking" in capsys.readouterr().out

    assert holdout_eval.main(argv + ["--no-chunking"]) == 0
    assert "chunking" not in capsys.readouterr().out
