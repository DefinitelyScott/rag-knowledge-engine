"""Does the README still describe the corpus and gold set it actually ships with?

Every accuracy table in the README is a mean over some number of questions, and
that number is written into the prose beside it. When the gold set grows, those
captions do not grow with it. Nothing was checking them, and the drift is
invisible by inspection because a stale table looks exactly like a fresh one -
it is the same shape, the same metrics, the same plausible decimals.

That is not hypothetical. The provenance note at the top of the README spent two
revisions asserting that the ablation and significance sections had been
recomputed at twenty-four documents. They had not: their `W/L/T` counts still
summed to 106 questions while the header three lines above them said 118. The
claim was wrong in the one direction that matters, because a provenance note
that overstates what was refreshed converts an unknown into a false assurance.

So the README gets the same treatment as the gold set: tested like code.

Two rules, both deliberately narrow so this fails on drift rather than on prose:

1. The header line that states the corpus shape must match the real corpus,
   the real chunk count and the real gold set.
2. Any table caption that claims to be over "all N questions" must either use
   the current gold count, or sit under a heading that carries an explicit
   stale marker. Labelling a stale table is allowed. Leaving it unlabelled is
   not. That is the whole convention, and this is the thing that enforces it.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from ragkb.engine import RAGEngine, read_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
GOLD = os.path.join(ROOT, "evals", "gold.jsonl")
CORPUS = os.path.join(ROOT, "corpus")

# "> **Stale (106 questions).**" / "> **Stale (20 documents, 92 questions).**"
STALE_MARKER = re.compile(r"^>\s*\*\*Stale\b", re.IGNORECASE)
HEADING = re.compile(r"^(#{2,4})\s")

# The header sentence every reader anchors on. Matched against the joined text
# rather than line by line, because it wraps:
#     Current numbers on the bundled corpus and gold set (24 documents, 116
#     chunks, 118 questions, k=5, document-level relevance).
# Anchoring on the sentence also stops this matching a stale marker, several of
# which quote an old shape in exactly the same parenthesised form.
HEADER = re.compile(
    r"Current numbers on the bundled corpus and gold set\s*"
    r"\((\d+) documents?,\s*(\d+) chunks?,\s*(\d+) questions",
    re.IGNORECASE,
)

# The caption forms the README uses to introduce a live table. Kept to the
# explicit "all N questions" phrasings on purpose: prose that merely mentions a
# count while discussing history ("at twenty documents the 92-question table
# said...") is legitimate and must not trip this test.
LIVE_CAPTION = re.compile(
    r"(?:over\s+)?all\s+(\d+)\s+(?:gold\s+)?questions",
    re.IGNORECASE,
)

# The other caption form, and the one that actually hid the drift: a bolded
# slice label immediately above a table, e.g. "**easy (60 questions)**" or
# "**hard (46 questions, 28 with two relevant documents)**". These quote a slice
# size rather than the whole gold set, so they are checked against the slice
# counts as well as the total.
SLICE_CAPTION = re.compile(
    r"^\*\*[^*()]*\((\d+)\s+questions",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def readme_lines():
    with open(README, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


@pytest.fixture(scope="module")
def gold_rows():
    with open(GOLD, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def gold_count(gold_rows):
    return len(gold_rows)


@pytest.fixture(scope="module")
def valid_counts(gold_rows):
    """Counts a caption may legitimately quote: the whole set, or any slice."""
    counts = {len(gold_rows)}
    slices = {}
    for row in gold_rows:
        label = row.get("difficulty")
        slices[label] = slices.get(label, 0) + 1
    counts.update(slices.values())
    return counts


@pytest.fixture(scope="module")
def corpus_shape():
    docs = read_corpus(CORPUS)
    engine = RAGEngine.from_corpus(CORPUS)
    return len(docs), len(engine.chunks)


def test_header_states_the_real_corpus_and_gold_shape(
    readme_lines, gold_count, corpus_shape
):
    """The one line every reader anchors on has to be true."""
    n_docs, n_chunks = corpus_shape
    # Collapse newlines so the wrapped header matches as one sentence.
    text = " ".join(line.strip() for line in readme_lines)
    found = HEADER.search(text)
    assert found, "README no longer states its corpus shape in the expected form"

    stated_docs, stated_chunks, stated_questions = (int(g) for g in found.groups())
    assert stated_docs == n_docs, (
        f"README says {stated_docs} documents, corpus/ holds {n_docs}"
    )
    assert stated_chunks == n_chunks, (
        f"README says {stated_chunks} chunks, the engine builds {n_chunks}"
    )
    assert stated_questions == gold_count, (
        f"README says {stated_questions} questions, gold.jsonl holds {gold_count}"
    )


def _sections_with_stale_markers(lines):
    """Index every line by whether its enclosing section was marked stale.

    A marker applies from where it appears until the next heading at the same or
    a higher level, so a marker under `## Persistence` still covers the
    `### Growing the index` subsection beneath it. Scoping it to "the next
    heading of any level" instead would silently drop the marker at the first
    subheading, which is exactly the false negative this whole file exists to
    prevent - the section would read as labelled and test as unlabelled.
    """
    stale_at = [False] * len(lines)
    marked_level = None
    for i, line in enumerate(lines):
        heading = HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if marked_level is not None and level <= marked_level:
                marked_level = None
        elif STALE_MARKER.match(line) and marked_level is None:
            # Attribute the marker to the most recent heading above it.
            marked_level = _enclosing_heading_level(lines, i)
        stale_at[i] = marked_level is not None
    return stale_at


def _enclosing_heading_level(lines, index):
    for line in reversed(lines[:index]):
        heading = HEADING.match(line)
        if heading:
            return len(heading.group(1))
    return 1


def test_live_table_captions_match_the_current_gold_set(readme_lines, gold_count):
    """An unlabelled table caption must be quoting the gold set it ships with."""
    stale_at = _sections_with_stale_markers(readme_lines)

    drifted = []
    for i, line in enumerate(readme_lines):
        # Table rows are data, not captions. The power section keeps a
        # deliberate history table whose rows read "22 documents, all 106
        # questions"; those are records of past measurements and must stay.
        if line.lstrip().startswith("|"):
            continue
        for match in LIVE_CAPTION.finditer(line):
            stated = int(match.group(1))
            if stated == gold_count or stale_at[i]:
                continue
            drifted.append((i + 1, stated, line.strip()))

    assert not drifted, (
        "README table captions claim a question count that is neither the current "
        f"gold set ({gold_count}) nor inside a section marked stale:\n"
        + "\n".join(f"  line {n}: says {said} - {text}" for n, said, text in drifted)
        + "\n\nEither re-run the section and update the caption, or add a "
        '"> **Stale (N questions).**" marker under its heading.'
    )


def test_slice_captions_match_a_real_slice(readme_lines, valid_counts):
    """`**easy (60 questions)**` above a table must name a slice that exists.

    This is the form the 106-question ablation tables were hiding behind for two
    revisions: the header said 118, the caption said 60, and nothing compared
    them.
    """
    stale_at = _sections_with_stale_markers(readme_lines)

    drifted = []
    for i, line in enumerate(readme_lines):
        match = SLICE_CAPTION.match(line.strip())
        if not match:
            continue
        stated = int(match.group(1))
        if stated in valid_counts or stale_at[i]:
            continue
        drifted.append((i + 1, stated, line.strip()))

    assert not drifted, (
        "README slice captions name a question count matching no current slice "
        f"(valid: {sorted(valid_counts)}):\n"
        + "\n".join(f"  line {n}: says {said} - {text}" for n, said, text in drifted)
        + "\n\nEither re-run the section and update the caption, or add a "
        '"> **Stale (N questions).**" marker under its heading.'
    )


def test_the_stale_marker_convention_is_actually_in_use(readme_lines):
    """Guard the guard.

    If the marker regex ever stops matching what the README writes, the test
    above silently degrades into checking nothing at all, because every caption
    would look unlabelled and the author would 'fix' it by editing numbers.
    A repo that documents stale sections should have some.
    """
    markers = [line for line in readme_lines if STALE_MARKER.match(line)]
    assert markers, (
        "no stale markers found - either every section is current (delete this "
        "test) or the marker format drifted from what the parser expects"
    )
