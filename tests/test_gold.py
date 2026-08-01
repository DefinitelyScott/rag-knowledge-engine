"""Integrity checks on the labelled gold set.

A gold set is the only thing standing between an honest number and a
comfortable one, so it gets tested like code: ids unique, every relevant
document actually present in the corpus, difficulty labels from a closed
vocabulary, and both slices populated. A typo in a document id would otherwise
show up as a silent, permanent retrieval "miss".
"""

from __future__ import annotations

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "evals", "gold.jsonl")
CORPUS = os.path.join(ROOT, "corpus")

DIFFICULTIES = {"easy", "hard"}


def load_gold():
    with open(GOLD, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corpus_doc_ids():
    return {
        os.path.splitext(name)[0]
        for name in os.listdir(CORPUS)
        if name.lower().endswith((".md", ".txt"))
    }


@pytest.fixture(scope="module")
def gold():
    return load_gold()


def test_gold_set_is_not_empty(gold):
    assert len(gold) >= 30


def test_ids_are_unique(gold):
    ids = [record["id"] for record in gold]
    assert len(ids) == len(set(ids))


def test_records_have_the_expected_shape(gold):
    for record in gold:
        assert record["question"].strip()
        assert isinstance(record["relevant"], list)
        assert record["relevant"], record["id"]
        assert record["difficulty"] in DIFFICULTIES, record["id"]


def test_relevant_documents_exist_in_the_corpus(gold):
    known = corpus_doc_ids()
    for record in gold:
        unknown = set(record["relevant"]) - known
        assert not unknown, "{} references missing documents {}".format(
            record["id"], sorted(unknown)
        )


def test_relevant_lists_have_no_duplicates(gold):
    for record in gold:
        assert len(record["relevant"]) == len(set(record["relevant"])), record["id"]


def test_questions_are_distinct(gold):
    questions = [record["question"].strip().lower() for record in gold]
    assert len(questions) == len(set(questions))


def test_both_difficulty_slices_are_populated(gold):
    counts = dict.fromkeys(DIFFICULTIES, 0)
    for record in gold:
        counts[record["difficulty"]] += 1
    for level, count in counts.items():
        assert count >= 10, "{} slice is too small to mean anything".format(level)


def test_hard_slice_contains_multi_document_questions(gold):
    multi = [
        record
        for record in gold
        if record["difficulty"] == "hard" and len(record["relevant"]) > 1
    ]
    # Without these, recall@k is numerically identical to hit@k across the
    # slice and the metric carries no information of its own.
    assert len(multi) >= 5


def test_every_corpus_document_is_exercised(gold):
    referenced = {doc for record in gold for doc in record["relevant"]}
    assert corpus_doc_ids() <= referenced
