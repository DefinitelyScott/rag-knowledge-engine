"""Okapi BM25 lexical retrieval.

BM25 scores a document by how often the query terms occur in it, discounted by
how common those terms are in the collection (IDF) and saturated so that the
tenth occurrence of a word counts for much less than the first. See Robertson &
Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple


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

    def fit(self, documents: Sequence[Sequence[str]]) -> "BM25Index":
        """Build the index from a sequence of token lists."""
        self.term_frequencies = [Counter(tokens) for tokens in documents]
        self.doc_lengths = [len(tokens) for tokens in documents]
        self.doc_count = len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )

        self.document_frequency = {}
        for frequencies in self.term_frequencies:
            for term in frequencies:
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

        # Robertson/Sparck Jones IDF with the +1 smoothing that keeps the value
        # non-negative for terms present in more than half the collection.
        self.idf = {
            term: math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in self.document_frequency.items()
        }
        return self

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
        length = self.doc_lengths[doc_index]
        norm = (
            self.k1 * (1.0 - self.b + self.b * length / self.avg_doc_length)
            if self.avg_doc_length
            else self.k1
        )
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
        return [self.score_document(query_tokens, i) for i in range(self.doc_count)]

    def search(self, query_tokens: Sequence[str], k: int = 10) -> List[Tuple[int, float]]:
        """Return the ``k`` best (document index, score) pairs, best first."""
        return self.search_weighted(Counter(query_tokens), k=k)

    def search_weighted(
        self, weights: Mapping[str, float], k: int = 10
    ) -> List[Tuple[int, float]]:
        """Rank documents against a weighted query model, best first."""
        scored = [
            (index, self.score_document_weighted(weights, index))
            for index in range(self.doc_count)
        ]
        scored = [(index, score) for index, score in scored if score > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
