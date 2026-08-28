"""Okapi BM25 lexical retrieval.

BM25 scores a document by how often the query terms occur in it, discounted by
how common those terms are in the collection (IDF) and saturated so that the
tenth occurrence of a word counts for much less than the first. See Robertson &
Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009).

Ranking used to score every document in the collection against every query
term. This module now inverts that loop: the index keeps a postings list -
term -> the ``(document, term frequency)`` pairs for the documents containing
it - and ranking walks one query term's postings at a time, accumulating each
document's running score. See ``corpus/inverted-index.md``.

Two things are worth separating, because the first is the obvious win and the
second is the one that actually pays here:

* **Skipping documents that contain no query term.** Exact - BM25 sums a
  per-term contribution, so those documents score precisely 0.0 and the
  ranking already discards zero scores. It is also nearly worthless on
  natural-language questions: a question containing one common word matches
  almost the whole collection through that word alone. Measured on this
  corpus, the candidate set for a gold question averages 96% of the chunks,
  so pruning by membership saves about 4% of the work. That number is
  reported rather than buried because it is the reason the second change
  exists.

* **Not looking up the terms a document does not have.** This is where the
  work was. Scoring document-at-a-time costs one dictionary lookup per
  (candidate document, query term) pair whether or not the term is present -
  ~96% of the collection times every term in the question, most of them
  misses. Term-at-a-time costs one accumulator update per posting actually
  recorded, which is the sum of the query terms' document frequencies. The
  loop no longer pays for absences.

Neither change is an approximation: contributions are accumulated in query-term
order, the same order the document-at-a-time loop added them in, so the scores
are bit-for-bit identical and so is the ranking. ``tests/test_postings.py``
reconstructs the old full scan and asserts exactly that.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


class BM25Index:
    """An in-memory Okapi BM25 index over pre-tokenised documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_length = 0.0
        self.doc_lengths: List[int] = []
        self.term_frequencies: List[Counter] = []
        self.document_frequency: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}

    def fit(self, documents: Sequence[Sequence[str]]) -> "BM25Index":
        """Build the index from a sequence of token lists."""
        self.term_frequencies = [Counter(tokens) for tokens in documents]
        self.doc_lengths = [len(tokens) for tokens in documents]
        self.doc_count = len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )

        # Document frequency and the postings list are the same traversal seen
        # from two sides - df is the length of the postings list - so they are
        # built together rather than in two passes over the same counters.
        self.document_frequency = {}
        self.postings = {}
        for index, frequencies in enumerate(self.term_frequencies):
            for term, count in frequencies.items():
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
                self.postings.setdefault(term, []).append((index, count))

        self._refresh_idf()
        return self

    def _refresh_idf(self) -> None:
        """Recompute IDF for every term from the current document frequencies.

        Robertson/Sparck Jones IDF with the +1 smoothing that keeps the value
        non-negative for terms present in more than half the collection.

        Every value depends on ``doc_count``, so this is all-or-nothing: one
        new document changes the IDF of terms it does not contain. That is why
        :meth:`add_documents` cannot touch only the terms it saw, and it is the
        cheapest global work in this package - one pass over the vocabulary,
        with no tokenisation and no document access.
        """
        self.idf = {
            term: math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in self.document_frequency.items()
        }

    def add_documents(self, documents: Sequence[Sequence[str]]) -> "BM25Index":
        """Extend a fitted index with new documents, without re-reading the old.

        The state BM25 keeps is additive almost everywhere. Term frequencies
        and document lengths are per-document and simply append; document
        frequency is a running count; the postings list gains one entry per
        (new document, term) pair and keeps its ascending document order
        because new documents take the highest indices. Only two things are
        collection-wide, and neither needs the old documents to update:
        ``avg_doc_length`` is a sum already being maintained, and IDF is one
        pass over the vocabulary.

        So the old documents are never re-tokenised, re-counted, or re-scanned,
        and the result is exactly what :meth:`fit` would have produced over the
        concatenation - the same counters, lengths, frequencies and postings,
        in the same order. ``tests/test_incremental.py`` asserts that against a
        real from-scratch build rather than trusting the argument.

        The catch is not in this method but in what it implies for the callers
        that share an index with it: see :meth:`ragkb.vector.TfidfIndex.fit`
        and ``evals/incremental.py`` for why the vector half cannot do this.
        """
        if not documents:
            return self
        start = self.doc_count
        for offset, tokens in enumerate(documents):
            frequencies = Counter(tokens)
            index = start + offset
            self.term_frequencies.append(frequencies)
            self.doc_lengths.append(len(tokens))
            for term, count in frequencies.items():
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
                self.postings.setdefault(term, []).append((index, count))
        self.doc_count = start + len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        self._refresh_idf()
        return self

    def _build_postings(self) -> Dict[str, List[Tuple[int, int]]]:
        """Rebuild ``term -> (document, term frequency)`` from the term frequencies."""
        postings: Dict[str, List[Tuple[int, int]]] = {}
        for index, frequencies in enumerate(self.term_frequencies):
            for term, count in frequencies.items():
                postings.setdefault(term, []).append((index, count))
        return postings

    def candidates(self, terms: Iterable[str]) -> List[int]:
        """Document indices containing at least one of ``terms``, ascending.

        Every document outside this set scores exactly 0.0 for these terms.
        Exposed mainly so the cost of the pipeline can be measured honestly -
        on natural-language questions this set is nearly the whole collection,
        and :func:`_accumulate` is what makes the inverted index pay.
        """
        matched: Set[int] = set()
        for term in terms:
            matched.update(index for index, _ in self.postings.get(term, ()))
        return sorted(matched)

    def _length_norm(self, doc_index: int) -> float:
        """The ``k1 * (1 - b + b * len / avg_len)`` denominator term."""
        if not self.avg_doc_length:
            return self.k1
        length = self.doc_lengths[doc_index]
        return self.k1 * (1.0 - self.b + self.b * length / self.avg_doc_length)

    def _accumulate(self, weights: Mapping[str, float]) -> Dict[int, float]:
        """Score every document that any query term reaches, term-at-a-time.

        One pass per query term over that term's postings, adding its
        contribution into a per-document accumulator. Work is proportional to
        the postings actually traversed - the sum of the query terms' document
        frequencies - rather than to (documents x query terms), so the loop
        never pays for a term a document does not contain.

        The addition order per document is the query-term order, which is the
        order :meth:`score_document_weighted` uses too. That is not a detail:
        it is what makes this bit-for-bit identical to the scan it replaced
        rather than merely equal to within floating-point noise.
        """
        totals: Dict[int, float] = {}
        norms: Dict[int, float] = {}
        for term, weight in weights.items():
            if not weight:
                continue
            idf = self.idf.get(term)
            if not idf:
                continue
            factor = weight * idf
            for index, tf in self.postings.get(term, ()):
                norm = norms.get(index)
                if norm is None:
                    norm = self._length_norm(index)
                    norms[index] = norm
                totals[index] = totals.get(index, 0.0) + factor * (
                    tf * (self.k1 + 1.0)
                ) / (tf + norm)
        return totals

    def to_dict(self) -> Dict:
        """Serialisable snapshot of the fitted state.

        Everything derivable is still stored (df, idf, averages): recomputing
        it on load would re-run most of :meth:`fit`, and the point of
        persisting is to skip that work, not to re-do it from a different
        starting point.

        The postings list is the one exception, and deliberately so. It is a
        pure transposition of ``term_frequencies``, which is already here:
        rebuilding it on load costs one pass over data the loader has just
        parsed anyway, while storing it would roughly double the snapshot for
        no new information. IDF is stored because recomputing *it* would mean
        re-tokenising the corpus, which is the expensive thing.
        """
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": list(self.doc_lengths),
            "term_frequencies": [dict(tf) for tf in self.term_frequencies],
            "document_frequency": dict(self.document_frequency),
            "idf": dict(self.idf),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BM25Index":
        """Restore a fitted index from :meth:`to_dict` output without refitting."""
        index = cls(k1=data["k1"], b=data["b"])
        index.doc_count = data["doc_count"]
        index.avg_doc_length = data["avg_doc_length"]
        index.doc_lengths = list(data["doc_lengths"])
        index.term_frequencies = [Counter(tf) for tf in data["term_frequencies"]]
        index.document_frequency = dict(data["document_frequency"])
        index.idf = dict(data["idf"])
        index.postings = index._build_postings()
        return index

    def score_document(self, query_tokens: Sequence[str], doc_index: int) -> float:
        """Score one document against an unweighted list of query tokens.

        A term repeated in the query contributes its score once per occurrence,
        so this is exactly the weighted scorer with the term counts as weights.
        """
        return self.score_document_weighted(Counter(query_tokens), doc_index)

    def score_document_weighted(
        self, weights: Mapping[str, float], doc_index: int
    ) -> float:
        """Score one document against a weighted query model.

        Query-term weighting is linear in BM25: saturation applies to the
        *document* term frequency, not the query's, so a weight of 2.0 really
        does count a term twice. That is what lets query expansion express a
        proper term distribution instead of faking one by repeating tokens.
        """
        frequencies = self.term_frequencies[doc_index]
        norm = self._length_norm(doc_index)
        score = 0.0
        for term, weight in weights.items():
            tf = frequencies.get(term, 0)
            if not tf or not weight:
                continue
            score += (
                weight * self.idf.get(term, 0.0) * (tf * (self.k1 + 1.0)) / (tf + norm)
            )
        return score

    def scores(self, query_tokens: Sequence[str]) -> List[float]:
        """Score every document, in index order.

        Still one entry per document - callers index into this by document id -
        but the values come from the term-at-a-time pass. Documents no query
        term reaches stay 0.0, which is the value a full scan would have
        computed for them anyway.
        """
        values = [0.0] * self.doc_count
        for index, score in self._accumulate(Counter(query_tokens)).items():
            values[index] = score
        return values

    def search(self, query_tokens: Sequence[str], k: int = 10) -> List[Tuple[int, float]]:
        """Return the ``k`` best (document index, score) pairs, best first."""
        return self.search_weighted(Counter(query_tokens), k=k)

    def search_weighted(
        self, weights: Mapping[str, float], k: int = 10
    ) -> List[Tuple[int, float]]:
        """Rank documents against a weighted query model, best first.

        Ties break on the document index. The accumulator is a dict, so the
        sort is given an explicitly ascending key rather than relying on
        insertion order - postings-order and scan-order differ, and a tie that
        resolved differently between them would be a real ranking change
        hiding behind an implementation detail.
        """
        scored = [
            (index, score)
            for index, score in self._accumulate(weights).items()
            if score > 0.0
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
