"""Deterministic, keyboard-aware query corruption for robustness evaluation.

Every table in ``evals/evaluate.py`` scores queries that were typed correctly.
That is the easy half of the problem. The character n-gram third list in
:mod:`ragkb.ngram` exists for the other half: BM25 and TF-IDF both index whole
words, so - as the module docstring of :mod:`ragkb.retriever` puts it - "both
miss the same misspelling and the same morphological variant". That is a
falsifiable claim about noisy input, and until now the only evidence for it was
a single unit test. The accuracy tables that decide whether the third list
earns its cost are computed entirely on clean text, which is the one condition
under which the list was never expected to help.

This module supplies the missing input: a corruption of a query that looks like
a typo rather than like random noise. :mod:`evals.robustness` turns it into the
measurement.

Three properties matter more here than the choice of operators.

**Seeded and position-independent.** The corruption of a query depends only on
the query text and the seed, never on where the query sits in the gold file, so
appending a question does not silently re-roll the noise applied to the
questions already there - otherwise every corpus increment would quietly
invalidate the previous robustness numbers. Python's builtin ``hash`` is salted
per process and would make even a fixed seed irreproducible across runs, so the
per-query stream is derived with BLAKE2b instead.

**Rate 0.0 is the identity.** ``perturb`` returns the query unchanged, not a
query that happens to survive the coin flips. That is the same identity check
``mmr_lambda = 1.0`` and a missing n-gram index provide elsewhere in the
project: the noisy pipeline has to reproduce the clean baseline exactly before
any degradation it reports means anything.

**Budgeted, not per-character coin flips.** The number of edits is
``round(rate * eligible_characters)`` and those positions are sampled without
replacement. Independent per-character flips would make the realised error rate
a random variable, so two methods measured at "10%" would in fact be measured
against different amounts of damage. Budgeting removes that source of variance
between rows of the same table.

Two deliberate limits. Corruption is confined to runs of ASCII letters, so
``BM25`` loses at most its ``BM``; digits and punctuation are left alone because
mistyping them is a different failure mode with a different fix. And tokens
shorter than ``min_token_length`` are skipped entirely - corrupting a three-
letter word usually destroys it rather than misspelling it, and no retriever of
any design recovers a token with nothing left of it.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

#: The four Damerau-Levenshtein edit types, which is the standard model of a
#: single-character typing error. They are drawn uniformly: that is a modelling
#: choice for lack of a measured typo distribution over this domain's
#: vocabulary, not a claim that real typists distribute their mistakes evenly.
OPERATORS: Tuple[str, ...] = ("substitute", "transpose", "delete", "insert")

#: Runs of ASCII letters - the only spans eligible for corruption.
_WORD_RE = re.compile(r"[A-Za-z]+")

#: Never delete a word down below this length. Deletion is the one operator
#: that can shrink a token toward nothing, and a two-character remnant is not a
#: misspelling of anything.
_MIN_SURVIVING_LENGTH = 3

#: A QWERTY keyboard as three rows of horizontally offset keys. Substituting a
#: uniformly random letter models a transmission error; substituting a
#: physically adjacent key models a human hand, and the two produce very
#: different character n-grams - a neighbour substitution usually leaves most of
#: the surrounding grams intact, which is exactly the signal a character index
#: is supposed to survive on.
#: The offsets are the real stagger of a physical keyboard rather than round
#: numbers: at offset 1.5 the bottom row lines up one key too far right and
#: ``c`` comes out adjacent to ``g``, which no hand would confuse.
_KEYBOARD_ROWS = (
    ("qwertyuiop", 0.0),
    ("asdfghjkl", 0.5),
    ("zxcvbnm", 1.25),
)


def _build_neighbours() -> dict:
    """Map each letter to the keys within one key-width of it on QWERTY."""
    placed = []
    for row_index, (letters, offset) in enumerate(_KEYBOARD_ROWS):
        for column, letter in enumerate(letters):
            placed.append((letter, row_index, column + offset))
    neighbours = {}
    for letter, row, x in placed:
        near = [
            other
            for other, other_row, other_x in placed
            if other != letter and abs(other_row - row) <= 1 and abs(other_x - x) <= 1.0
        ]
        neighbours[letter] = tuple(sorted(near))
    return neighbours


NEIGHBOURS = _build_neighbours()


@dataclass(frozen=True)
class TypoConfig:
    """How much noise to inject, and the stream that decides where.

    ``rate`` is a character error rate: the fraction of *eligible* characters
    that receive an edit. Because short tokens are excluded, the realised error
    rate over the whole query is always somewhat lower than ``rate``, which is
    why :func:`character_error_rate` exists and why the robustness tables report
    what they achieved rather than only what they asked for.
    """

    rate: float = 0.1
    seed: int = 0
    min_token_length: int = 4
    operators: Tuple[str, ...] = OPERATORS

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        if self.min_token_length < 1:
            raise ValueError("min_token_length must be positive")
        if not self.operators:
            raise ValueError("at least one operator is required")
        unknown = [name for name in self.operators if name not in OPERATORS]
        if unknown:
            raise ValueError("unknown operators: {}".format(", ".join(unknown)))


def _stream(query: str, seed: int) -> random.Random:
    """A random stream determined by ``query`` and ``seed`` alone.

    Deriving the stream from the text rather than from a running counter is what
    makes one query's corruption independent of the other queries in the file.
    """
    digest = hashlib.blake2b(
        query.encode("utf-8"), digest_size=8, key=str(seed).encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _substitute(chars: List[str], index: int, rng: random.Random) -> bool:
    original = chars[index]
    options = NEIGHBOURS.get(original.lower())
    if not options:
        return False
    replacement = rng.choice(options)
    chars[index] = replacement.upper() if original.isupper() else replacement
    return True


def _transpose(chars: List[str], index: int, rng: random.Random) -> bool:
    partner = index + 1 if index + 1 < len(chars) else index - 1
    if partner < 0 or chars[partner] == chars[index]:
        # Swapping a letter with its twin is a no-op that would spend budget
        # without changing the query, and would leave the realised error rate
        # below the requested one for no reason the reader could see.
        return False
    chars[index], chars[partner] = chars[partner], chars[index]
    return True


def _delete(chars: List[str], index: int, rng: random.Random) -> bool:
    if len(chars) <= _MIN_SURVIVING_LENGTH:
        return False
    del chars[index]
    return True


def _insert(chars: List[str], index: int, rng: random.Random) -> bool:
    # Doubling the character under the cursor, which is the overwhelmingly
    # common insertion typo - a stray *unrelated* letter is a rarer accident.
    chars.insert(index, chars[index])
    return True


_APPLY = {
    "substitute": _substitute,
    "transpose": _transpose,
    "delete": _delete,
    "insert": _insert,
}


def perturb(query: str, config: TypoConfig = TypoConfig()) -> str:
    """Return ``query`` with roughly ``config.rate`` of its letters mistyped.

    Edits are applied right to left within each word so that a deletion or an
    insertion cannot invalidate the position of an edit not yet applied. An
    operator that cannot run at its position - a deletion that would gut a short
    word, a transposition between two identical letters - falls back to
    substitution rather than silently returning the budget, so the amount of
    damage stays comparable across the methods being compared.
    """
    if config.rate == 0.0:
        return query

    words = [
        (match.start(), match.group(0))
        for match in _WORD_RE.finditer(query)
        if len(match.group(0)) >= config.min_token_length
    ]
    if not words:
        return query

    positions = [
        (word_index, char_index)
        for word_index, (_, word) in enumerate(words)
        for char_index in range(len(word))
    ]
    budget = int(round(config.rate * len(positions)))
    if budget <= 0:
        return query

    rng = _stream(query, config.seed)
    chosen = rng.sample(positions, min(budget, len(positions)))

    edits: dict = {}
    for word_index, char_index in chosen:
        edits.setdefault(word_index, []).append(char_index)

    rebuilt = list(query)
    for word_index, char_indices in sorted(edits.items(), reverse=True):
        start, word = words[word_index]
        chars = list(word)
        for char_index in sorted(char_indices, reverse=True):
            if char_index >= len(chars):
                # A previous deletion in this word shortened it past this
                # position; nudge the edit onto the last surviving character
                # rather than dropping it.
                char_index = len(chars) - 1
            operator = rng.choice(config.operators)
            if not _APPLY[operator](chars, char_index, rng):
                _substitute(chars, char_index, rng)
        rebuilt[start : start + len(word)] = chars
    return "".join(rebuilt)


def perturb_all(queries: Sequence[str], config: TypoConfig = TypoConfig()) -> List[str]:
    """Corrupt every query independently under the same configuration."""
    return [perturb(query, config) for query in queries]


def damerau_levenshtein(left: str, right: str) -> int:
    """Optimal string alignment distance: edits *including* transposition.

    Plain Levenshtein has to spend two edits to swap a pair of adjacent
    characters, because it can only substitute. Measuring this module's damage
    with it would therefore report one mistyped ``freuqency`` as two errors and
    inflate every realised rate in the robustness tables - a measurement that
    disagrees with the model that produced the data. The four operators here are
    exactly the four Damerau-Levenshtein edit types, so the distance that scores
    them has to be the one that knows a transposition costs one.

    This is the restricted (optimal string alignment) variant: no substring is
    edited more than once. That is the right restriction for this use, since the
    corruption applies at most one operator per sampled position.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    # Three rolling rows: the transposition case needs the row before last.
    before_previous: List[int] = []
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            )
            if (
                i > 1
                and j > 1
                and left_char == right[j - 2]
                and left[i - 2] == right_char
            ):
                cost = min(cost, before_previous[j - 2] + 1)
            current.append(cost)
        before_previous, previous = previous, current
    return previous[-1]


def character_error_rate(original: str, corrupted: str) -> float:
    """Realised error rate: edit distance normalised by the original length.

    The requested rate and the realised rate differ - short tokens are skipped,
    and rounding a budget to a whole number of edits truncates - so the tables
    report this alongside the knob rather than assuming the knob was obeyed.
    """
    if not original:
        return 0.0
    return damerau_levenshtein(original, corrupted) / len(original)
