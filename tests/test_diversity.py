"""Tests for the MMR diversity audit in ``evals/diversity.py``.

The script makes three claims that a reader is entitled to have checked rather
than taken on the strength of a docstring:

1. recall@k and hit@k are the same number on a single-relevant-document
   question, which is why the sweep restricts itself to the multi-document ones.
2. ``lambda = 1.0`` reproduces the unmodified ranking, so any churn or recall
   movement it reports would be a bug in the harness rather than a property of
   MMR. This is the same identity check ``tests/test_rerank.py`` makes of the
   re-ranker itself, applied one level up to the measurement built on it.
3. The paired counts are a partition: every question is improved, worsened or
   tied exactly once, and the p-value depends only on the ones that moved.

The depth audit is checked for the structural fact it exists to expose - that
truncating to ``k`` documents pins ``kept_docs`` at ``k`` whenever the pool
holds that many, so a flat ``kept_docs`` column is not evidence that MMR did
nothing, and ``churn`` is what distinguishes the two.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from ragkb.engine import EngineConfig, RAGEngine
from ragkb.metrics import hit_at_k, recall_at_k

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_diversity():
    path = os.path.join(ROOT, "evals", "diversity.py")
    spec = importlib.util.spec_from_file_location("evals_diversity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


diversity = _load_diversity()


DOCS = {
    "alpha": (
        "BM25 saturates term frequency so a repeated word stops adding score. "
        "The saturation is controlled by k1. Longer documents are normalised by b. "
        "Saturation is what separates BM25 from raw term counting."
    ),
    "beta": (
        "TF-IDF weights a term by how rare it is across the collection. "
        "Cosine similarity compares two sparse vectors after L2 normalisation. "
        "Rare terms carry more weight than common ones."
    ),
    "gamma": (
        "Reciprocal rank fusion merges ranked lists without needing comparable scores. "
        "Each list contributes one over a constant plus the rank. "
        "Fusion is robust to one list being badly calibrated."
    ),
    "delta": (
        "Character n-grams survive misspellings that word tokenisers cannot match. "
        "A four-gram index shares the same scoring as the word index. "
        "Noise robustness is the reason the third list exists."
    ),
}


@pytest.fixture(scope="module")
def engine() -> RAGEngine:
    return RAGEngine(DOCS, EngineConfig(target_tokens=25, overlap_tokens=8))


def test_multi_document_selects_only_questions_with_more_than_one_relevant():
    gold = [
        {"id": "a", "question": "q", "relevant": ["alpha"]},
        {"id": "b", "question": "q", "relevant": ["alpha", "beta"]},
        {"id": "c", "question": "q", "relevant": ["alpha", "beta", "gamma"]},
    ]
    picked = diversity.multi_document(gold)
    assert [record["id"] for record in picked] == ["b", "c"]


def test_recall_equals_hit_on_single_document_questions(engine):
    """The premise the whole script rests on, checked on a real index."""
    gold = [
        {"id": "s1", "question": "how does BM25 saturate term frequency", "relevant": ["alpha"]},
        {"id": "s2", "question": "what does cosine similarity compare", "relevant": ["beta"]},
        {"id": "s3", "question": "how are ranked lists merged", "relevant": ["gamma"]},
    ]
    checked, mismatches = diversity.recall_equals_hit(engine, gold, k=2)
    assert checked == 3
    assert mismatches == 0


def test_recall_equals_hit_skips_multi_document_questions(engine):
    gold = [
        {"id": "m1", "question": "how does BM25 saturate", "relevant": ["alpha", "beta"]},
    ]
    checked, mismatches = diversity.recall_equals_hit(engine, gold, k=2)
    assert (checked, mismatches) == (0, 0)


def test_metric_identity_holds_directly():
    """recall@k == hit@k whenever exactly one document is relevant."""
    ranked = ["beta", "alpha", "gamma"]
    for k in (1, 2, 3):
        for gold_doc in ("alpha", "beta", "delta"):
            assert recall_at_k(ranked, [gold_doc], k) == hit_at_k(ranked, [gold_doc], k)


def test_lambda_one_leaves_the_scored_doc_list_untouched(engine):
    """``lambda = 1.0`` is the identity, so churn must be exactly zero.

    If this fails, the depth audit's churn column is measuring harness noise
    rather than re-ranking, and every conclusion drawn from it is void.
    """
    gold = [
        {"id": "m1", "question": "how does BM25 saturate term frequency", "relevant": ["alpha", "beta"]},
        {"id": "m2", "question": "why do rare terms carry more weight", "relevant": ["beta", "gamma"]},
    ]
    base = diversity.baseline_doc_lists(engine, gold, k=3)
    report = diversity.depth_audit(engine, gold, k=3, mmr_lambda=1.0, baseline=base)
    assert report["churn"] == 0.0
    off = diversity.depth_audit(engine, gold, k=3, mmr_lambda=None, baseline=base)
    assert off["churn"] == 0.0
    assert report["kept_docs"] == off["kept_docs"]
    assert report["shallow_docs"] == off["shallow_docs"]


def test_depth_audit_reports_both_selection_depths(engine):
    """deep_docs must be measured at k*4 and so cannot be below the shallow count."""
    gold = [
        {"id": "m1", "question": "how does BM25 saturate term frequency", "relevant": ["alpha", "beta"]},
    ]
    base = diversity.baseline_doc_lists(engine, gold, k=2)
    report = diversity.depth_audit(engine, gold, k=2, mmr_lambda=0.5, baseline=base)
    assert report["deep_docs"] >= report["shallow_docs"]
    # kept_docs is capped by k, which is the pinning this column exists to show.
    assert report["kept_docs"] <= 2.0


def test_recall_profile_counts_documents_not_questions(engine):
    gold = [
        {"id": "m1", "question": "how does BM25 saturate term frequency", "relevant": ["alpha", "beta"]},
        {"id": "m2", "question": "why do rare terms carry more weight", "relevant": ["beta", "gamma"]},
    ]
    profile = diversity.recall_profile(engine, gold, k=4, mmr_lambda=None)
    assert profile["needed"] == 4
    assert 0 <= profile["found"] <= 4
    assert len(profile["per_question"]) == 2
    assert profile["recall"] == pytest.approx(
        sum(profile["per_question"]) / 2
    )


def test_paired_change_partitions_the_questions():
    baseline = [0.0, 0.5, 1.0, 0.5]
    candidate = [0.5, 0.5, 0.5, 0.0]
    improved, worsened, tied, p_value = diversity.paired_change(baseline, candidate)
    assert (improved, worsened, tied) == (1, 2, 1)
    assert improved + worsened + tied == len(baseline)
    assert 0.0 <= p_value <= 1.0


def test_paired_change_reports_no_evidence_when_nothing_moves():
    series = [0.0, 0.5, 1.0]
    improved, worsened, tied, p_value = diversity.paired_change(series, series)
    assert (improved, worsened, tied) == (0, 0, 3)
    assert p_value == 1.0


def test_paired_change_ignores_the_ties_when_computing_p():
    """Padding a comparison with tied questions must not move the p-value."""
    _, _, _, short = diversity.paired_change([0.0, 1.0], [1.0, 0.0])
    _, _, _, padded = diversity.paired_change(
        [0.0, 1.0] + [0.5] * 20, [1.0, 0.0] + [0.5] * 20
    )
    assert short == padded


def test_paired_change_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        diversity.paired_change([0.0, 1.0], [1.0])


def test_main_runs_end_to_end_on_the_real_corpus(capsys):
    """The script must survive a real run, and say what it measured."""
    exit_code = diversity.main(["--lambdas", "1.0,0.5", "--slice", "multi"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "selection depth audit" in out
    assert "churn" in out
    assert "recall@5" in out
    assert "0 mismatches" in out


def test_main_accepts_the_all_slice(capsys):
    exit_code = diversity.main(["--lambdas", "0.7", "--slice", "all"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "on the all slice" in out
