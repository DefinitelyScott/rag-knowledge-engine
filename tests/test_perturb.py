"""Tests for query corruption and the robustness harness built on it.

A perturbation module is a measuring instrument, and the failure mode of a
measuring instrument is not a crash - it is a number that looks fine and means
nothing. So the tests that matter here pin the properties the robustness tables
silently assume: rate 0.0 is exactly the identity (otherwise the "clean" row is
not a baseline), the same seed reproduces the same corruption in a way that does
not depend on process start-up or on the position of the query in the file
(otherwise no run is comparable to any other), and the damage actually lands
where the docstring claims - inside long alphabetic words and nowhere else.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from ragkb.engine import EngineConfig, RAGEngine
from ragkb.perturb import (
    NEIGHBOURS,
    OPERATORS,
    TypoConfig,
    character_error_rate,
    damerau_levenshtein,
    perturb,
    perturb_all,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location(
    "robustness", os.path.join(ROOT, "evals", "robustness.py")
)
robustness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = robustness
_spec.loader.exec_module(robustness)


QUERY = "Why does BM25 saturate term frequency instead of counting occurrences?"

CORPUS = {
    "fusion": (
        "Reciprocal rank fusion merges two ranked lists into one ordering "
        "without ever comparing their raw scores against each other."
    ),
    "chunking": (
        "Chunking splits a long document into overlapping passages so that an "
        "explanation spanning a boundary stays retrievable from both sides."
    ),
}


# --------------------------------------------------------------------------
# the identity, which every other number in the harness is measured against
# --------------------------------------------------------------------------


def test_rate_zero_is_exactly_the_identity():
    for text in (QUERY, "", "a", "BM25", "   spaced   out   "):
        assert perturb(text, TypoConfig(rate=0.0)) == text


def test_rate_zero_ignores_the_seed():
    assert perturb(QUERY, TypoConfig(rate=0.0, seed=7)) == perturb(
        QUERY, TypoConfig(rate=0.0, seed=99)
    )


def test_a_rate_that_rounds_to_no_edits_leaves_the_query_alone():
    # Honest under-delivery: a tiny rate on a short query buys nothing, and the
    # realised-noise table is what surfaces that rather than a silent minimum.
    tiny = perturb("retrieval", TypoConfig(rate=0.001))
    assert tiny == "retrieval"


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------


def test_same_seed_reproduces_the_same_corruption():
    config = TypoConfig(rate=0.2, seed=3)
    assert perturb(QUERY, config) == perturb(QUERY, config)


def test_corruption_does_not_depend_on_position_in_the_batch():
    """Appending a gold question must not re-roll the noise on the existing ones.

    A single ``random.Random`` walked across the batch would give every query a
    corruption determined by how many queries preceded it, so adding one
    question to ``gold.jsonl`` would invalidate every robustness number already
    recorded. The per-query stream is derived from the text instead.
    """
    config = TypoConfig(rate=0.2, seed=1)
    alone = perturb(QUERY, config)
    batch = perturb_all(["a decoy question about chunking", QUERY], config)
    longer = perturb_all(["one", "two", "three", QUERY], config)
    assert alone == batch[1] == longer[3]


def test_the_stream_does_not_use_the_salted_builtin_hash():
    """``hash`` is salted per process, so a keyed digest has to be doing the work.

    Pinning the derivation rather than a golden output keeps the test meaningful
    if the operator mix ever changes, while still failing loudly if someone
    swaps the keyed digest for ``hash(query)``.
    """
    import hashlib

    from ragkb.perturb import _stream

    expected = int.from_bytes(
        hashlib.blake2b(
            QUERY.encode("utf-8"), digest_size=8, key=b"5"
        ).digest(),
        "big",
    )
    assert _stream(QUERY, 5).random() == __import__("random").Random(expected).random()


def test_different_seeds_generally_differ():
    variants = {perturb(QUERY, TypoConfig(rate=0.2, seed=seed)) for seed in range(8)}
    assert len(variants) > 1


# --------------------------------------------------------------------------
# where the damage lands
# --------------------------------------------------------------------------


def test_digits_and_punctuation_survive():
    noisy = perturb(QUERY, TypoConfig(rate=0.5, seed=2))
    for character in "0123456789?":
        assert noisy.count(character) == QUERY.count(character)


def test_short_tokens_are_never_touched():
    short_only = "is it in on of to a an the"
    assert perturb(short_only, TypoConfig(rate=1.0, seed=4)) == short_only


def test_min_token_length_widens_what_is_eligible():
    text = "why is rag ok"
    assert perturb(text, TypoConfig(rate=1.0, seed=0)) == text
    assert perturb(text, TypoConfig(rate=1.0, seed=0, min_token_length=3)) != text


def test_no_word_is_deleted_out_of_existence():
    for seed in range(12):
        noisy = perturb(QUERY, TypoConfig(rate=1.0, seed=seed))
        assert all(word for word in noisy.split())
        assert noisy.split() and len(noisy.split()) == len(QUERY.split())


def test_insertion_only_never_shortens():
    config = TypoConfig(rate=0.4, seed=1, operators=("insert",))
    assert len(perturb(QUERY, config)) > len(QUERY)


def test_deletion_only_never_lengthens():
    config = TypoConfig(rate=0.4, seed=1, operators=("delete",))
    assert len(perturb(QUERY, config)) < len(QUERY)


def test_substitution_only_preserves_length():
    config = TypoConfig(rate=0.4, seed=1, operators=("substitute",))
    assert len(perturb(QUERY, config)) == len(QUERY)


def test_substitution_stays_on_the_keyboard():
    config = TypoConfig(rate=1.0, seed=0, operators=("substitute",))
    noisy = perturb("retrieval", config)
    assert noisy != "retrieval"
    for original, replaced in zip("retrieval", noisy):
        assert replaced == original or replaced in NEIGHBOURS[original]


# --------------------------------------------------------------------------
# the rate knob means something
# --------------------------------------------------------------------------


def test_more_noise_requested_means_more_noise_delivered():
    realised = [
        sum(
            character_error_rate(QUERY, perturb(QUERY, TypoConfig(rate=rate, seed=s)))
            for s in range(8)
        )
        / 8
        for rate in (0.05, 0.1, 0.2, 0.4)
    ]
    assert realised == sorted(realised)
    assert realised[0] > 0.0


def test_realised_rate_never_exceeds_the_requested_one_by_much():
    # Skipped short tokens mean the realised rate trails; it must not overshoot,
    # or the tables would be labelled with less damage than they applied.
    for rate in (0.05, 0.1, 0.2):
        for seed in range(6):
            noisy = perturb(QUERY, TypoConfig(rate=rate, seed=seed))
            assert character_error_rate(QUERY, noisy) <= rate + 1e-9


# --------------------------------------------------------------------------
# supporting arithmetic
# --------------------------------------------------------------------------


def test_distance_known_pairs():
    assert damerau_levenshtein("", "") == 0
    assert damerau_levenshtein("abc", "abc") == 0
    assert damerau_levenshtein("", "abc") == 3
    assert damerau_levenshtein("kitten", "sitting") == 3
    assert damerau_levenshtein("flaw", "lawn") == 2


def test_a_transposition_costs_one_edit_not_two():
    """The distance has to agree with the model that generated the damage.

    Plain Levenshtein charges two substitutions for one swapped pair, which
    would inflate every realised rate in the robustness tables above the budget
    that was actually spent.
    """
    assert damerau_levenshtein("ab", "ba") == 1
    assert damerau_levenshtein("frequency", "freuqency") == 1


def test_distance_is_symmetric():
    assert damerau_levenshtein("saturate", "staurate") == damerau_levenshtein(
        "staurate", "saturate"
    )


def test_character_error_rate_of_an_untouched_query_is_zero():
    assert character_error_rate(QUERY, QUERY) == 0.0
    assert character_error_rate("", "anything") == 0.0


def test_neighbours_are_symmetric_and_exclude_self():
    for key, near in NEIGHBOURS.items():
        assert key not in near
        for other in near:
            assert key in NEIGHBOURS[other]


def test_neighbours_match_the_physical_keyboard():
    assert set(NEIGHBOURS["g"]) == set("tyfhvb")
    assert set(NEIGHBOURS["a"]) == set("qwsz")
    # The bottom-row stagger has to be right or ``c`` comes out next to ``g``.
    assert "g" not in NEIGHBOURS["c"]


# --------------------------------------------------------------------------
# configuration is validated rather than silently misbehaving
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate": -0.1},
        {"rate": 1.5},
        {"min_token_length": 0},
        {"operators": ()},
        {"operators": ("substitute", "teleport")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TypoConfig(**kwargs)


def test_every_declared_operator_is_implemented():
    from ragkb.perturb import _APPLY

    assert set(_APPLY) == set(OPERATORS)


# --------------------------------------------------------------------------
# the harness on top
# --------------------------------------------------------------------------


def _engines():
    return {
        "base": RAGEngine(CORPUS),
        "ngram": RAGEngine(CORPUS, config=EngineConfig(ngram=4)),
    }


GOLD = [
    {"id": "t1", "question": "how are two ranked lists merged", "relevant": ["fusion"]},
    {
        "id": "t2",
        "question": "why do passages overlap at a boundary",
        "relevant": ["chunking"],
    },
]


def test_clean_row_reproduces_the_unperturbed_scores():
    """The rate-0.0 row must equal scoring the raw questions, or it is no baseline."""
    engines = _engines()
    means, _, realised = robustness.sweep(engines, GOLD, k=2, rates=[0.0], seeds=[0])
    direct = robustness.score(
        engines["base"],
        "hybrid",
        [record["question"] for record in GOLD],
        [record["relevant"] for record in GOLD],
        k=2,
    )
    assert means[0.0]["hybrid2"] == direct
    assert realised[0.0] == 0.0


def test_sweep_reports_every_routed_method_at_every_rate():
    means, per_seed, realised = robustness.sweep(
        _engines(), GOLD, k=2, rates=[0.0, 0.2], seeds=[0, 1]
    )
    names = {name for name, _, _ in robustness.METHOD_ROUTING}
    for rate in (0.0, 0.2):
        assert set(means[rate]) == names
        assert realised[rate] >= 0.0
    # One seed at the identity rate - extra draws of no noise measure nothing -
    # and every requested seed once there is noise to draw.
    assert len(per_seed[0.0]["ngram"]) == 1
    assert len(per_seed[0.2]["ngram"]) == 2


def test_parse_rates_always_includes_the_clean_baseline():
    assert robustness.parse_rates("0.1,0.2") == [0.0, 0.1, 0.2]


@pytest.mark.parametrize("text", ["", "0", "1.5", "-0.2"])
def test_parse_rates_rejects_nonsense(text):
    with pytest.raises(SystemExit):
        robustness.parse_rates(text)


def test_verdict_is_derived_from_the_numbers_not_asserted():
    """The printed conclusion has to flip when the measurements flip."""
    ndcg = "ndcg@2"
    favourable = {
        0.0: {
            "hybrid2": {ndcg: 0.8},
            "hybrid3": {ndcg: 0.8},
            "ngram": {ndcg: 0.8},
        },
        0.2: {
            "hybrid2": {ndcg: 0.4},
            "hybrid3": {ndcg: 0.7},
            "ngram": {ndcg: 0.5},
        },
    }
    text = robustness.verdict(favourable, 0.0, 0.2, k=2)
    assert "does buy robustness" in text
    assert "the fusion is the way to spend it" in text

    favourable[0.2]["ngram"][ndcg] = 0.75
    text = robustness.verdict(favourable, 0.0, 0.2, k=2)
    assert "not the way to spend it" in text

    favourable[0.2]["hybrid3"][ndcg] = 0.3
    text = robustness.verdict(favourable, 0.0, 0.2, k=2)
    assert "buys no robustness" in text


def test_main_runs_end_to_end(capsys, tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for doc_id, text in CORPUS.items():
        (corpus_dir / "{}.md".format(doc_id)).write_text(text, encoding="utf-8")
    import json

    gold_path.write_text(
        "\n".join(json.dumps(record) for record in GOLD), encoding="utf-8"
    )
    exit_code = robustness.main(
        [
            "--corpus",
            str(corpus_dir),
            "--gold",
            str(gold_path),
            "--k",
            "2",
            "--rates",
            "0.2",
            "--seeds",
            "2",
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "realised noise" in printed
    assert "retention" in printed
    for name, _, _ in robustness.METHOD_ROUTING:
        assert name in printed
