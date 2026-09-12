"""Is the gold set's ``difficulty`` label measuring anything, or is it a vibe?

Every "hard slice" number in the README - the expansion table, the re-ranking
table, the weighted-RRF table, ``evals/holdout.py --slice hard`` - is computed
over the questions carrying ``"difficulty": "hard"`` in ``evals/gold.jsonl``.
That label was **assigned by hand, by the person who also wrote the corpus**,
and until now nothing in the repo checked whether it corresponds to any
measurable property of the question. A load-bearing partition that has never
been audited is exactly the kind of thing the rest of this project takes apart,
so this module audits it.

The measurable property it audits against is the one the retrievers actually
depend on: **how much of the question's wording is present in its own answer
document at all.** Call it containment.

    containment = |query terms that occur in the relevant documents|
                  -----------------------------------------------------
                              |distinct query terms|

This is a *ceiling* diagnostic, not a score. A word-level lexical retriever
cannot match on a term the answer document does not contain - BM25 gives it
weight zero, TF-IDF gives it no coordinate - so a question whose terms are
largely absent from its answer is unreachable by lexical matching however the
scores are combined. That is the lexical gap, and it is the precise condition
the character n-gram list in :mod:`ragkb.ngram` was added to bridge: n-grams
match ``normalisation`` against ``normalise`` where the word index scores
exactly zero. So containment is not an arbitrary proxy for difficulty. It is
the quantity the third list exists to attack, which makes it the right axis to
stratify the third list's gain along.

Two forms are computed, because the plain count over-credits stopword-adjacent
filler:

``containment``
    Unweighted share of distinct query terms present. Easy to state, easy to
    check by hand, and the version the tests pin.
``idf_containment``
    The same share with each term weighted by its inverse document frequency,
    so matching ``okapi`` counts for more than matching ``parameter``. This is
    the form the report leads with: a question can contain most of its terms in
    the answer and still be hard if the only ones that match are the ones every
    document has.

The IDF here is BM25's own - ``log(1 + (N - df + 0.5) / (df + 0.5))``, the
formula in :meth:`ragkb.bm25.BM25Index._refresh_idf` - so the weighting is the
retriever's notion of term value rather than a new constant invented for this
module. It is computed over **documents** rather than chunks, because a gold
row's answer key is a document: the question being asked is "is this term
anywhere in the answer", and a term present in one chunk of the answer and
absent from the other nine is still present in the answer.

What containment is not
-----------------------
It measures *presence in the answer*, not *discriminativeness against the rest
of the corpus*. A question whose every term appears in its answer document may
have those same terms in six other documents, in which case containment is
1.000 and retrieval is still a coin flip between seven candidates. The report
finds exactly that at the top of the range, and it is a real limitation of the
proxy rather than noise - stated here so the number is not read as a difficulty
oracle. Containment bounds what lexical matching *can* do; it does not predict
what it *will* do.

Standard library only, and deterministic: no seeds, no sampling. The same gold
set and corpus produce the same bands to the digit.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .text import tokenize

#: Upper edges of the containment bands the report stratifies along, chosen
#: before any metric was computed so the bands are not fitted to the result.
#: They split the 118-question gold set into roughly comparable counts rather
#: than at round numbers that happen to flatter a column.
BAND_EDGES: Tuple[float, ...] = (0.35, 0.55, 0.75)


@dataclass(frozen=True)
class Overlap:
    """Containment of one question's terms in its own relevant documents."""

    question_id: str
    #: Distinct content terms in the question, after stopword removal.
    terms: int
    #: How many of those occur somewhere in the union of the relevant documents.
    matched: int
    containment: float
    idf_containment: float
    #: Size of the answer key. The audit needs this: the hand label turns out to
    #: be perfectly confounded with it.
    documents: int
    #: The hand-assigned label being audited, carried through for crosstabs.
    label: str

    @property
    def missing(self) -> int:
        """Query terms absent from the answer - the part no word index can reach."""
        return self.terms - self.matched


def bm25_idf(doc_frequency: int, total_documents: int) -> float:
    """BM25's IDF, reproduced so this module weights terms as the retriever does.

    Identical to the expression in :meth:`ragkb.bm25.BM25Index._refresh_idf`;
    ``tests/test_difficulty.py`` asserts the two agree term for term on a
    fitted index rather than trusting this comment.
    """
    if total_documents <= 0:
        raise ValueError("total_documents must be positive")
    if not 0 <= doc_frequency <= total_documents:
        raise ValueError(
            "doc_frequency must be in [0, {}], got {}".format(
                total_documents, doc_frequency
            )
        )
    return math.log(
        1.0 + (total_documents - doc_frequency + 0.5) / (doc_frequency + 0.5)
    )


def document_term_sets(documents: Mapping[str, str]) -> Dict[str, frozenset]:
    """``doc_id -> set of content terms``, tokenised exactly as the index would."""
    return {
        doc_id: frozenset(tokenize(text, remove_stopwords=True))
        for doc_id, text in documents.items()
    }


def document_frequencies(term_sets: Mapping[str, Iterable[str]]) -> Dict[str, int]:
    """How many documents each term occurs in, for the IDF weighting."""
    frequencies: Counter = Counter()
    for terms in term_sets.values():
        frequencies.update(set(terms))
    return dict(frequencies)


def overlap_for(
    record: Mapping,
    term_sets: Mapping[str, Iterable[str]],
    frequencies: Optional[Mapping[str, int]] = None,
    total_documents: Optional[int] = None,
) -> Overlap:
    """Containment for one gold row.

    A term absent from the whole corpus still counts against containment: the
    question asked it and the answer does not contain it, which is the
    condition being measured. Its IDF is the maximum the corpus can award
    (``df = 0``), so an unmatched out-of-vocabulary term is weighted heavily -
    correctly, since it is the least reachable kind of term there is.
    """
    if frequencies is None:
        frequencies = document_frequencies(term_sets)
    if total_documents is None:
        total_documents = len(term_sets)

    query_terms = sorted(set(tokenize(record["question"], remove_stopwords=True)))
    answer_terms: set = set()
    for doc_id in record["relevant"]:
        answer_terms |= set(term_sets.get(doc_id, ()))

    matched = [term for term in query_terms if term in answer_terms]
    weights = {
        term: bm25_idf(frequencies.get(term, 0), total_documents)
        for term in query_terms
    }
    total_weight = sum(weights.values())
    matched_weight = sum(weights[term] for term in matched)

    # A question with no content terms at all would divide by zero. It cannot
    # occur in this gold set and `tests/test_gold.py` would catch one, but a
    # containment of 0.0 is the right reading of "nothing to match on" rather
    # than a crash, so it is handled instead of asserted away.
    return Overlap(
        question_id=record.get("id", "?"),
        terms=len(query_terms),
        matched=len(matched),
        containment=(len(matched) / len(query_terms)) if query_terms else 0.0,
        idf_containment=(matched_weight / total_weight) if total_weight > 0 else 0.0,
        documents=len(record["relevant"]),
        label=record.get("difficulty", "easy"),
    )


def overlaps(
    gold: Sequence[Mapping], documents: Mapping[str, str]
) -> List[Overlap]:
    """Containment for every gold row, computed against one corpus."""
    term_sets = document_term_sets(documents)
    frequencies = document_frequencies(term_sets)
    total = len(term_sets)
    return [
        overlap_for(record, term_sets, frequencies, total) for record in gold
    ]


def assign_band(value: float, edges: Sequence[float] = BAND_EDGES) -> int:
    """Index of the band ``value`` falls in, counting from the bottom.

    Half-open upwards - ``value < edge`` - so a value landing exactly on an
    edge belongs to the band above it, and the top band is unbounded. With the
    default edges that is four bands: ``[0, 0.35)``, ``[0.35, 0.55)``,
    ``[0.55, 0.75)``, ``[0.75, 1]``.
    """
    if list(edges) != sorted(edges):
        raise ValueError("edges must be ascending")
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def band_label(
    index: int,
    edges: Sequence[float] = BAND_EDGES,
    upper: Optional[float] = 1.0,
) -> str:
    """Printable interval for a band index, e.g. ``[0.35, 0.55)``.

    ``upper`` closes the top band. It defaults to 1.0 because containment is a
    ratio, but the same banding is reused for unbounded quantities - the
    question-length control in :mod:`evals.difficulty` - and labelling a term
    count as ending at 1.00 would be worse than useless. Pass ``None`` there.
    """
    low = 0.0 if index == 0 else edges[index - 1]
    if index >= len(edges):
        if upper is None:
            return "[{:.2f}, inf)".format(low)
        return "[{:.2f}, {:.2f}]".format(low, upper)
    return "[{:.2f}, {:.2f})".format(low, edges[index])


def group_by_band(
    rows: Sequence[Mapping],
    values: Sequence[float],
    edges: Sequence[float] = BAND_EDGES,
    upper: Optional[float] = 1.0,
) -> List[Tuple[str, List[Mapping]]]:
    """Partition gold rows by the band of their containment, lowest band first.

    Every band is returned even when empty, so a table built from this has a
    stable number of rows as the corpus grows and a band emptying out is
    visible rather than silently dropped.
    """
    if len(rows) != len(values):
        raise ValueError("rows and values must be the same length")
    buckets: List[List[Mapping]] = [[] for _ in range(len(edges) + 1)]
    for record, value in zip(rows, values):
        buckets[assign_band(value, edges)].append(record)
    return [
        (band_label(index, edges, upper), bucket)
        for index, bucket in enumerate(buckets)
    ]


def label_multidoc_table(rows: Sequence[Overlap]) -> Tuple[int, int, int, int]:
    """The 2x2 the audit turns on: hand label against answer-key size.

    Returns ``(hard_multi, hard_single, easy_multi, easy_single)``, laid out for
    :func:`ragkb.power.fisher_exact`. If the off-diagonal is empty the label is
    not an independent axis from "how many documents answer this question", and
    every hard-slice result is a blend of two effects rather than one.
    """
    hard_multi = sum(1 for row in rows if row.label == "hard" and row.documents > 1)
    hard_single = sum(1 for row in rows if row.label == "hard" and row.documents == 1)
    easy_multi = sum(1 for row in rows if row.label != "hard" and row.documents > 1)
    easy_single = sum(1 for row in rows if row.label != "hard" and row.documents == 1)
    return hard_multi, hard_single, easy_multi, easy_single


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 on an empty sequence."""
    return sum(values) / len(values) if values else 0.0
