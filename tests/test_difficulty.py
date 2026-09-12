"""Tests for measured question difficulty and the two exact tests that audit it."""

from __future__ import annotations

import json
import math
import os

import pytest

from ragkb.bm25 import BM25Index
from ragkb.difficulty import (
    BAND_EDGES,
    assign_band,
    band_label,
    bm25_idf,
    document_frequencies,
    document_term_sets,
    group_by_band,
    label_multidoc_table,
    overlap_for,
    overlaps,
)
from ragkb.engine import read_corpus
from ragkb.power import fisher_exact
from ragkb.significance import two_sample_permutation
from ragkb.text import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# containment
# --------------------------------------------------------------------------


def test_every_query_term_present_gives_containment_one():
    documents = {"d1": "Okapi BM25 saturates term frequency."}
    term_sets = document_term_sets(documents)
    record = {
        "id": "q",
        "question": "Does BM25 saturates term frequency?",
        "relevant": ["d1"],
    }
    result = overlap_for(record, term_sets)
    assert result.containment == pytest.approx(1.0)
    assert result.idf_containment == pytest.approx(1.0)
    assert result.missing == 0


def test_a_morphological_variant_is_counted_as_missing():
    """The case the whole module is built to expose, and the third list to fix.

    ``saturate`` and ``saturates`` are the same word to a reader and different
    terms to the tokenizer, so the word index scores this query term at exactly
    zero. Containment has to charge for that - if it quietly credited the
    variant it would be measuring a stemmer nobody wrote, and the low-band
    stratification in ``evals/difficulty.py`` would lose the questions the
    n-gram list is there to rescue.
    """
    documents = {"d1": "Okapi BM25 saturates term frequency."}
    term_sets = document_term_sets(documents)
    variant = overlap_for(
        {"id": "q", "question": "Does BM25 saturate term frequency?", "relevant": ["d1"]},
        term_sets,
    )
    assert variant.matched == variant.terms - 1
    assert variant.containment < 1.0
    # The character n-gram analyzer, unlike the word tokenizer, does reach it.
    from ragkb.ngram import char_ngrams

    assert set(char_ngrams(["saturate"])) & set(char_ngrams(["saturates"]))


def test_no_query_term_present_gives_containment_zero():
    documents = {"d1": "Okapi BM25 saturates term frequency."}
    term_sets = document_term_sets(documents)
    record = {"id": "q", "question": "How do dolphins navigate?", "relevant": ["d1"]}
    result = overlap_for(record, term_sets)
    assert result.containment == pytest.approx(0.0)
    assert result.idf_containment == pytest.approx(0.0)
    assert result.missing == result.terms


def test_containment_counts_distinct_terms_not_occurrences():
    """A term repeated in the question is one term, so it cannot skew the ratio."""
    documents = {"d1": "alpha beta", "d2": "gamma"}
    term_sets = document_term_sets(documents)
    once = {"id": "a", "question": "alpha gamma", "relevant": ["d1"]}
    thrice = {"id": "b", "question": "alpha alpha alpha gamma", "relevant": ["d1"]}
    assert overlap_for(once, term_sets).containment == pytest.approx(
        overlap_for(thrice, term_sets).containment
    )


def test_stopwords_are_excluded_from_both_sides():
    """`the` must not be counted as a match; it would inflate every question."""
    documents = {"d1": "the quorum of the index"}
    term_sets = document_term_sets(documents)
    record = {"id": "q", "question": "the of the", "relevant": ["d1"]}
    result = overlap_for(record, term_sets)
    # Nothing but stopwords survives tokenisation, so there is nothing to match.
    assert result.terms == 0
    assert result.containment == pytest.approx(0.0)


def test_multi_document_answer_uses_the_union():
    documents = {"d1": "alpha", "d2": "beta"}
    term_sets = document_term_sets(documents)
    record = {"id": "q", "question": "alpha beta", "relevant": ["d1", "d2"]}
    split = overlap_for(record, term_sets)
    assert split.containment == pytest.approx(1.0)
    assert split.documents == 2
    # Either document alone covers only half of it.
    half = overlap_for(
        {"id": "q", "question": "alpha beta", "relevant": ["d1"]}, term_sets
    )
    assert half.containment == pytest.approx(0.5)


def test_idf_weighting_reduces_to_plain_when_all_terms_share_a_df():
    """The weighted form is a generalisation, so it must collapse onto the plain one."""
    # Both terms occur in exactly one of two documents, so both carry the same idf.
    documents = {"d1": "alpha beta", "d2": "gamma delta"}
    term_sets = document_term_sets(documents)
    record = {"id": "q", "question": "alpha gamma", "relevant": ["d1"]}
    result = overlap_for(record, term_sets)
    assert result.containment == pytest.approx(0.5)
    assert result.idf_containment == pytest.approx(result.containment)


def test_idf_weighting_credits_the_rare_term_more_than_the_common_one():
    """Matching a corpus-wide term is worth less than matching a distinctive one."""
    documents = {
        "d1": "rare common",
        "d2": "common filler",
        "d3": "common filler",
        "d4": "common filler",
    }
    term_sets = document_term_sets(documents)
    # Matches the rare term, misses the ubiquitous one.
    rare = overlap_for({"id": "a", "question": "rare", "relevant": ["d1"]}, term_sets)
    common = overlap_for(
        {"id": "b", "question": "common", "relevant": ["d1"]}, term_sets
    )
    # Both have containment 1.0; the weighting cannot separate them on that.
    assert rare.containment == pytest.approx(common.containment)
    # But a match on `rare` is a match on more idf mass than a match on `common`.
    frequencies = document_frequencies(term_sets)
    assert bm25_idf(frequencies["rare"], 4) > bm25_idf(frequencies["common"], 4)


def test_out_of_vocabulary_query_term_counts_against_containment():
    """A term in no document at all is the least reachable kind there is."""
    documents = {"d1": "alpha"}
    term_sets = document_term_sets(documents)
    result = overlap_for(
        {"id": "q", "question": "alpha nonexistentterm", "relevant": ["d1"]}, term_sets
    )
    assert result.terms == 2
    assert result.matched == 1
    assert result.containment == pytest.approx(0.5)
    # df = 0 earns the maximum idf the corpus can award, so the unmatched
    # out-of-vocabulary term outweighs the matched one and drags the weighted
    # form below the plain one.
    assert result.idf_containment < result.containment


def test_bm25_idf_matches_the_index_it_claims_to_reproduce():
    """The whole point of the weighting is that it is the retriever's, not a new one."""
    documents = {
        "d1": "alpha beta gamma",
        "d2": "alpha beta",
        "d3": "alpha",
    }
    tokenised = [tokenize(text, remove_stopwords=True) for text in documents.values()]
    index = BM25Index().fit(tokenised)
    term_sets = document_term_sets(documents)
    frequencies = document_frequencies(term_sets)
    assert set(frequencies) == set(index.idf)
    for term, df in frequencies.items():
        assert bm25_idf(df, len(documents)) == pytest.approx(index.idf[term])


def test_bm25_idf_rejects_impossible_counts():
    with pytest.raises(ValueError):
        bm25_idf(3, 2)
    with pytest.raises(ValueError):
        bm25_idf(-1, 2)
    with pytest.raises(ValueError):
        bm25_idf(0, 0)


# --------------------------------------------------------------------------
# banding
# --------------------------------------------------------------------------


def test_bands_are_half_open_upwards_at_every_edge():
    for index, edge in enumerate(BAND_EDGES):
        assert assign_band(edge - 1e-9) == index
        # A value landing exactly on an edge belongs to the band above it.
        assert assign_band(edge) == index + 1


def test_band_assignment_is_monotone():
    previous = assign_band(0.0)
    for step in range(1, 101):
        current = assign_band(step / 100.0)
        assert current >= previous
        previous = current


def test_band_extremes_land_in_the_outer_bands():
    assert assign_band(0.0) == 0
    assert assign_band(1.0) == len(BAND_EDGES)


def test_band_labels_are_contiguous_and_cover_the_unit_interval():
    labels = [band_label(i) for i in range(len(BAND_EDGES) + 1)]
    assert labels[0].startswith("[0.00,")
    assert labels[-1].endswith("1.00]")
    assert len(set(labels)) == len(labels)


def test_top_band_label_is_open_ended_for_an_unbounded_quantity():
    """The question-length control reuses the banding on a term count."""
    assert band_label(len(BAND_EDGES), upper=None).endswith("inf)")
    # Bounded bands below the top are unaffected by the upper bound.
    assert band_label(0, upper=None) == band_label(0)


def test_grouping_passes_the_upper_bound_through_to_its_labels():
    rows = [{"id": "a"}, {"id": "b"}]
    labels = [label for label, _ in group_by_band(rows, [0.1, 9.0], (1.0,), upper=None)]
    assert labels[-1] == "[1.00, inf)"


def test_unsorted_edges_are_rejected():
    with pytest.raises(ValueError):
        assign_band(0.5, (0.7, 0.3))


def test_grouping_is_a_partition_and_keeps_empty_bands():
    rows = [{"id": str(i)} for i in range(6)]
    values = [0.0, 0.1, 0.2, 0.3, 0.34, 0.8]
    groups = group_by_band(rows, values)
    assert len(groups) == len(BAND_EDGES) + 1
    assert sum(len(bucket) for _, bucket in groups) == len(rows)
    # Middle bands are empty here and must still be reported.
    assert [len(bucket) for _, bucket in groups] == [5, 0, 0, 1]


def test_grouping_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        group_by_band([{"id": "a"}], [0.1, 0.2])


# --------------------------------------------------------------------------
# Fisher exact
# --------------------------------------------------------------------------


def test_fisher_exact_matches_the_textbook_tea_tasting_table():
    """Fisher's own 3-3/1-3 example: the one-sided 3,1,1,3 table is 0.2429."""
    # The classic 4x4 tea-tasting design; the two-sided p for a 3/1 split is
    # the sum of the two equally-improbable tails.
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857142857142857, abs=1e-12)


def test_fisher_exact_hand_computed_small_table():
    """2,0,0,2 has exactly two tables at or below its probability: itself and its mirror."""
    # Margins are 2,2 by 2,2. P(a=2) = C(2,2)C(2,0)/C(4,2) = 1/6, same for a=0,
    # and P(a=1) = 4/6. So the two-sided p is 2/6.
    assert fisher_exact(2, 0, 0, 2) == pytest.approx(2.0 / 6.0, abs=1e-12)


def test_fisher_exact_is_invariant_under_the_four_table_symmetries():
    a, b, c, d = 7, 3, 2, 9
    base = fisher_exact(a, b, c, d)
    assert fisher_exact(a, c, b, d) == pytest.approx(base)  # transpose
    assert fisher_exact(c, d, a, b) == pytest.approx(base)  # swap rows
    assert fisher_exact(b, a, d, c) == pytest.approx(base)  # swap columns


def test_fisher_exact_returns_one_for_a_perfectly_balanced_table():
    """The most probable table under the null cannot be evidence against it."""
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)


def test_fisher_exact_agrees_with_a_direct_hypergeometric_sum():
    a, b, c, d = 4, 6, 8, 2
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2

    def probability(cell):
        return (
            math.comb(col1, cell)
            * math.comb(total - col1, row1 - cell)
            / math.comb(total, row1)
        )

    observed = probability(a)
    expected = sum(
        probability(cell)
        for cell in range(max(0, row1 - (total - col1)), min(row1, col1) + 1)
        if probability(cell) <= observed * (1 + 1e-9)
    )
    assert fisher_exact(a, b, c, d) == pytest.approx(expected, abs=1e-12)


def test_fisher_exact_perfect_separation_is_significant_at_the_gold_set_sizes():
    """The shape the real audit finds: one cell empty, and a p far below alpha."""
    p_value = fisher_exact(31, 21, 0, 66)
    assert 0.0 < p_value < 1e-10


def test_fisher_exact_handles_degenerate_margins_and_bad_input():
    # An empty row leaves no table free to vary, so independence stands.
    assert fisher_exact(0, 0, 3, 4) == pytest.approx(1.0)
    assert fisher_exact(0, 3, 0, 4) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fisher_exact(0, 0, 0, 0)
    with pytest.raises(ValueError):
        fisher_exact(-1, 1, 1, 1)


def test_fisher_exact_is_a_probability():
    for table in ((1, 2, 3, 4), (10, 1, 1, 10), (6, 6, 1, 0), (2, 3, 5, 7)):
        assert 0.0 < fisher_exact(*table) <= 1.0


# --------------------------------------------------------------------------
# two-sample permutation test
# --------------------------------------------------------------------------


def test_two_sample_identical_groups_give_p_one():
    values = [0.1, 0.2, 0.3, 0.4]
    assert two_sample_permutation(values, list(values)) == pytest.approx(1.0)


def test_two_sample_constant_groups_give_p_one():
    """Zero observed difference must not be read as significance."""
    assert two_sample_permutation([0.5] * 5, [0.5] * 4) == pytest.approx(1.0)


def test_two_sample_perfect_separation_lands_on_its_own_floor():
    """With 3 and 3 there are C(6,3)=20 labellings; only 2 reach a full separation."""
    p_value = two_sample_permutation([0.0, 0.1, 0.2], [0.8, 0.9, 1.0])
    assert p_value == pytest.approx(2.0 / 20.0, abs=1e-12)


def test_two_sample_is_exact_and_seedless_for_small_groups():
    """Below the enumeration limit the seed must not be able to change the answer."""
    a, b = [0.1, 0.4, 0.2], [0.7, 0.9, 0.6, 0.8]
    first = two_sample_permutation(a, b, seed=0)
    assert first == two_sample_permutation(a, b, seed=99)
    assert first == two_sample_permutation(a, b, resamples=37)


def test_two_sample_is_symmetric_in_its_arguments():
    a, b = [0.1, 0.4, 0.2], [0.7, 0.9, 0.6, 0.8]
    assert two_sample_permutation(a, b) == pytest.approx(two_sample_permutation(b, a))


def test_two_sample_uses_the_labels_not_the_pairing():
    """Unequal group sizes are the case a paired test cannot even accept."""
    # Nine versus three: a paired test would raise; this one must not.
    p_value = two_sample_permutation([0.0] * 9, [1.0] * 3)
    assert 0.0 < p_value < 0.05


def test_two_sample_monte_carlo_is_reproducible_and_never_zero():
    a = [i / 60.0 for i in range(60)]
    b = [0.5 + i / 60.0 for i in range(60)]
    first = two_sample_permutation(a, b, resamples=500, seed=3, exact_limit=0)
    assert first == two_sample_permutation(a, b, resamples=500, seed=3, exact_limit=0)
    assert first >= 1.0 / 501.0
    # A different seed gives a different draw but the same verdict.
    assert two_sample_permutation(a, b, resamples=500, seed=4, exact_limit=0) < 0.05


def test_two_sample_empty_group_is_no_evidence_rather_than_an_error():
    assert two_sample_permutation([], [0.1, 0.2]) == pytest.approx(1.0)
    assert two_sample_permutation([0.1], []) == pytest.approx(1.0)


def test_two_sample_rejects_bad_parameters():
    with pytest.raises(ValueError):
        two_sample_permutation([0.1], [0.2], resamples=0)
    with pytest.raises(ValueError):
        two_sample_permutation([0.1], [0.2], exact_limit=-1)


# --------------------------------------------------------------------------
# the real gold set
# --------------------------------------------------------------------------


def _real_rows():
    documents = read_corpus(os.path.join(ROOT, "corpus"))
    gold = []
    with open(os.path.join(ROOT, "evals", "gold.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                record.setdefault("difficulty", "easy")
                gold.append(record)
    return gold, overlaps(gold, documents)


def test_real_gold_set_containments_are_well_formed():
    gold, rows = _real_rows()
    assert len(rows) == len(gold)
    for row in rows:
        assert 0.0 <= row.containment <= 1.0
        assert 0.0 <= row.idf_containment <= 1.0
        assert row.terms > 0, "{} tokenised to nothing".format(row.question_id)
        assert 0 <= row.matched <= row.terms
        assert row.missing == row.terms - row.matched


def test_real_gold_set_hard_questions_have_lower_measured_containment():
    """The audit's headline: the hand label is not arbitrary.

    Pinned as an inequality rather than a value so corpus growth cannot make it
    fail spuriously - but a growth increment that *inverts* it should fail here,
    because it would mean the label had stopped describing the questions.
    """
    _, rows = _real_rows()
    easy = [row.idf_containment for row in rows if row.label == "easy"]
    hard = [row.idf_containment for row in rows if row.label == "hard"]
    assert easy and hard
    assert sum(hard) / len(hard) < sum(easy) / len(easy)


def test_real_gold_set_label_is_confounded_with_answer_key_size():
    """Pins the confound the report exists to disclose, so it cannot vanish silently."""
    _, rows = _real_rows()
    hard_multi, hard_single, easy_multi, easy_single = label_multidoc_table(rows)
    assert hard_multi + hard_single + easy_multi + easy_single == len(rows)
    # Every multi-document question is labelled hard. If a later increment adds
    # an easy multi-document question this fails, and the README's claim that
    # the axes coincide has to be re-measured rather than restated.
    assert easy_multi == 0
    assert hard_multi > 0
    assert fisher_exact(hard_multi, hard_single, easy_multi, easy_single) < 0.05
