"""Tests for the rank-agreement measures and the audit built on them.

The module exists to make one argument: that set overlap and order agreement
are different measurements, and that a claim about a *fusion* - which reads
rank and nothing else - has to be made in the second. So the tests that matter
most here are the ones that pin that difference down rather than the ones that
check arithmetic.

Four groups:

1. Identities. Identical lists, reversed lists, disjoint lists. These fix the
   endpoints of every measure, and reversed-versus-identical is the case that
   separates the set view from the order view by construction: set overlap
   cannot tell them apart and both other measures must.
2. Hand-computed RBO. The one formula here nobody can check by eye, worked
   through at a ``p`` chosen to keep the arithmetic exact in binary.
3. Top-weighting. RBO's entire reason for being chosen over plain overlap is
   that disagreement near rank 1 costs more than the same disagreement at
   rank 5. If that ordering ever broke, every conclusion drawn from the RBO
   column would still look plausible, so it is asserted directly.
4. The conjoint caveat, tested as behaviour rather than trusted as prose:
   tau-b must refuse to return a number it cannot support.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from ragkb.engine import EngineConfig, RAGEngine
from ragkb.rankagree import (
    DEFAULT_RBO_P,
    compare_rankings,
    kendall_tau_b,
    rank_biased_overlap,
    top_k_set_overlap,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_rank_agreement():
    path = os.path.join(ROOT, "evals", "rank_agreement.py")
    spec = importlib.util.spec_from_file_location("evals_rank_agreement", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rank_agreement = _load_rank_agreement()


DOCS = {
    "alpha": (
        "BM25 saturates term frequency so a repeated word stops adding score. "
        "The saturation is controlled by k1 and longer documents are normalised "
        "by b. Saturation is what separates BM25 from raw term counting."
    ),
    "beta": (
        "TF-IDF weights a term by how rare it is across the collection. Cosine "
        "similarity compares two sparse vectors after L2 normalisation. Rare "
        "terms carry more weight than common ones do."
    ),
    "gamma": (
        "Reciprocal rank fusion combines ranked lists without comparing their "
        "scores. Each list contributes one over a constant plus the rank. The "
        "constant flattens the contribution of the very top ranks."
    ),
}


@pytest.fixture(scope="module")
def engine():
    return RAGEngine(DOCS, config=EngineConfig(target_tokens=30, overlap_tokens=8))


# --------------------------------------------------------------------------
# 1. Identities
# --------------------------------------------------------------------------


def test_identical_lists_agree_completely_in_every_measure():
    a = [1, 2, 3, 4, 5]
    result = compare_rankings(a, list(a), depth=5)
    assert result.set_overlap == 1.0
    assert result.tau == 1.0
    assert result.identical is True
    assert result.same_head is True
    # RBO over a finite prefix cannot reach 1.0 - the weight below the cut is
    # unobserved - but base plus residual must, and that is the guarantee the
    # residual exists to give.
    assert result.rbo == pytest.approx(1.0 - DEFAULT_RBO_P**5)
    assert result.rbo + result.rbo_residual == pytest.approx(1.0)


def test_disjoint_lists_agree_in_nothing():
    result = compare_rankings([1, 2, 3], [4, 5, 6], depth=3)
    assert result.set_overlap == 0.0
    assert result.rbo == 0.0
    assert result.tau is None
    assert result.conjoint == 0
    assert result.same_head is False


def test_reversed_lists_are_the_case_set_overlap_cannot_see():
    """The whole argument for this module, as an assertion.

    Same five items, opposite order. Set overlap calls this perfect agreement.
    A fusion reading ``1 / (rrf_k + rank)`` would produce entirely different
    scores, and both order-aware measures say so.
    """
    a = [1, 2, 3, 4, 5]
    result = compare_rankings(a, list(reversed(a)), depth=5)

    assert result.set_overlap == 1.0, "set overlap is order-blind by definition"
    assert result.tau == -1.0, "every pair is discordant"
    assert result.rbo < 1.0 - DEFAULT_RBO_P**5
    assert result.identical is False
    assert result.same_head is False


def test_every_measure_is_symmetric():
    a = [1, 2, 3, 4, 5]
    b = [3, 1, 9, 2, 7]
    forward, backward = (
        compare_rankings(a, b, depth=5),
        compare_rankings(b, a, depth=5),
    )
    assert forward.set_overlap == backward.set_overlap
    assert forward.rbo == pytest.approx(backward.rbo)
    assert forward.tau == pytest.approx(backward.tau)


# --------------------------------------------------------------------------
# 2. Hand-computed RBO
# --------------------------------------------------------------------------


def test_rbo_matches_a_hand_computation():
    """``a = [1,2,3]``, ``b = [2,1,3]``, ``p = 0.5``, depth 3.

    Overlap at each depth: {1} vs {2} is 0; {1,2} vs {2,1} is 2; {1,2,3} vs
    {2,1,3} is 3. So the weighted sum is
    ``1*(0/1) + 0.5*(2/2) + 0.25*(3/3) = 0.75`` and RBO is
    ``(1 - 0.5) * 0.75 = 0.375``, with residual ``0.5**3 = 0.125``.

    ``p = 0.5`` is chosen so every term is exact in binary and the assertion
    cannot pass on rounding.
    """
    base, residual = rank_biased_overlap([1, 2, 3], [2, 1, 3], p=0.5, depth=3)
    assert base == pytest.approx(0.375)
    assert residual == pytest.approx(0.125)


def test_a_short_list_is_penalised_for_the_ranks_it_never_filled():
    """The denominator stays ``d``, so returning three of five is not a tie.

    ``[1,2,3]`` against ``[1,2,3,4,5]`` at depth 5: overlap is 1, 2, 3, 3, 3
    against depths 1..5, so the deep terms are fractional rather than full.
    """
    short, _ = rank_biased_overlap([1, 2, 3], [1, 2, 3, 4, 5], p=0.5, depth=5)
    full, _ = rank_biased_overlap(
        [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], p=0.5, depth=5
    )
    assert short < full


def test_rbo_rejects_a_degenerate_p():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            rank_biased_overlap([1], [1], p=bad, depth=1)


# --------------------------------------------------------------------------
# 3. Top-weighting
# --------------------------------------------------------------------------


def test_rbo_punishes_a_swap_at_the_head_more_than_one_at_the_tail():
    """The property that justifies choosing RBO over plain overlap.

    Both variants below hold the same five items and differ from the reference
    by exactly one adjacent transposition. Set overlap scores them identically.
    RBO must not, because a fused ranking is not indifferent between losing
    rank 1 and losing rank 5.
    """
    reference = [1, 2, 3, 4, 5]
    head_swap = [2, 1, 3, 4, 5]
    tail_swap = [1, 2, 3, 5, 4]

    assert top_k_set_overlap(reference, head_swap, depth=5) == top_k_set_overlap(
        reference, tail_swap, depth=5
    )
    # Kendall tau is equally blind here: one discordant pair either way.
    assert kendall_tau_b(reference, head_swap, depth=5).tau == pytest.approx(
        kendall_tau_b(reference, tail_swap, depth=5).tau
    )

    head, _ = rank_biased_overlap(reference, head_swap, depth=5)
    tail, _ = rank_biased_overlap(reference, tail_swap, depth=5)
    assert head < tail, "RBO must weight the top of the ranking more heavily"


# --------------------------------------------------------------------------
# 4. The conjoint caveat
# --------------------------------------------------------------------------


def test_tau_refuses_to_report_a_correlation_over_fewer_than_two_items():
    """``None`` rather than 0.0, because undefined is not uncorrelated.

    This is the behaviour ``evals/rank_agreement.py`` depends on when it drops
    those questions instead of averaging them in: scoring them 0.0 would pull
    the cross-representation pairs toward the middle for exactly the reason
    that should push them to the edge.
    """
    assert kendall_tau_b([1, 2, 3], [1, 8, 9], depth=3).tau is None
    assert kendall_tau_b([1, 2, 3], [1, 8, 9], depth=3).conjoint == 1
    assert kendall_tau_b([], [], depth=0).tau is None


def test_two_shared_items_can_only_ever_score_plus_or_minus_one():
    """Stated as a test because it is the reason ``conjoint`` is reported.

    A tau column averaging values that can only be the extremes is not a
    correlation in any useful sense, and a reader needs the conjoint size to
    know that is what they are looking at.
    """
    assert kendall_tau_b([1, 2, 7], [1, 2, 9], depth=3).tau == 1.0
    assert kendall_tau_b([1, 2, 7], [2, 1, 9], depth=3).tau == -1.0


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


def test_per_list_rankings_reads_the_inputs_not_the_fused_output(engine):
    """The measurement is of what RRF is handed, so the fused list must not appear."""
    tops = rank_agreement.per_list_rankings(engine, "how does BM25 saturate?", 3)
    assert set(tops) == {"bm25", "vector"}
    assert "hybrid" not in tops
    for ranking in tops.values():
        assert len(ranking) <= 3
        assert len(set(ranking)) == len(ranking), "a list must not repeat a chunk"


def test_the_ngram_list_joins_the_comparison_when_it_is_fitted():
    three = RAGEngine(
        DOCS, config=EngineConfig(target_tokens=30, overlap_tokens=8, ngram=4)
    )
    tops = rank_agreement.per_list_rankings(three, "how does BM25 saturate?", 3)
    assert set(tops) == {"bm25", "vector", "ngram"}


def test_mean_pairwise_rbo_of_one_list_against_itself_is_the_identity(engine):
    ranking = [1, 2, 3]
    identical = {"bm25": ranking, "vector": list(ranking)}
    value = rank_agreement.mean_pairwise_rbo(identical, 3, DEFAULT_RBO_P)
    assert value == pytest.approx(1.0 - DEFAULT_RBO_P**3)


def test_pairwise_table_covers_every_unordered_pair_once(engine):
    gold = [
        {"question": "how does BM25 saturate term frequency?"},
        {"question": "what does cosine similarity compare?"},
    ]
    rows = rank_agreement.pairwise_table(engine, gold, 3, DEFAULT_RBO_P)
    assert [label for label, _ in rows] == ["bm25 vs vector"]
    for _, values in rows:
        assert 0.0 <= values["set_overlap"] <= 1.0
        assert 0.0 <= values["rbo"] <= 1.0
        assert 0.0 <= values["identical"] <= 1.0
        # tau is dropped where undefined, so its n can never exceed the gold set.
        assert values["tau_n"] <= len(gold)


def test_identical_share_never_exceeds_the_same_head_share(engine):
    """A structural invariant, and a cheap guard on the whole table.

    Two lists cannot be identical without agreeing at rank 1, so the
    ``identical`` column is bounded above by ``same top1`` on every row. If
    that ever inverted, the two columns would have drifted apart in a way no
    accuracy check downstream would catch.
    """
    gold = [
        {"question": "how does BM25 saturate term frequency?"},
        {"question": "what does cosine similarity compare?"},
        {"question": "how does reciprocal rank fusion combine lists?"},
    ]
    rows = rank_agreement.pairwise_table(engine, gold, 3, DEFAULT_RBO_P)
    for label, values in rows:
        assert values["identical"] <= values["same_top1"] + 1e-12, label
