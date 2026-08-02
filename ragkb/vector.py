"""TF-IDF vector retrieval with cosine similarity.

This is the "semantic-ish" half of the hybrid retriever. It has no neural
embeddings and no external dependency: documents become sparse L2-normalised
TF-IDF vectors and are ranked by cosine similarity against the query vector.
It generalises better than BM25 across wording differences within a shared
vocabulary, and it fails in different ways, which is exactly what makes fusing
the two worthwhile.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple


class TfidfIndex:
    """Sparse TF-IDF index with cosine similarity search."""

    def __init__(self, sublinear_tf: bool = True) -> None:
        self.sublinear_tf = sublinear_tf
        self.doc_count = 0
        self.document_frequency: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vectors: List[Dict[str, float]] = []

    def fit(self, documents: Sequence[Sequence[str]]) -> "TfidfIndex":
        self.doc_count = len(documents)
        self.document_frequency = {}
        counters = [Counter(tokens) for tokens in documents]
        for frequencies in counters:
            for term in frequencies:
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

        # Smoothed IDF: log((1 + N) / (1 + df)) + 1, so a term appearing in every
        # document still contributes a little rather than collapsing to zero.
        self.idf = {
            term: math.log((1.0 + self.doc_count) / (1.0 + df)) + 1.0
            for term, df in self.document_frequency.items()
        }
        self.vectors = [self._vectorize(counter) for counter in counters]
        return self

    def _vectorize(self, frequencies: Counter) -> Dict[str, float]:
        vector: Dict[str, float] = {}
        for term, count in frequencies.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            vector[term] = tf * idf
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        if norm:
            for term in vector:
                vector[term] /= norm
        return vector

    def vectorize_query(self, query_tokens: Sequence[str]) -> Dict[str, float]:
        return self._vectorize(Counter(query_tokens))

    def vectorize_weights(self, weights: Mapping[str, float]) -> Dict[str, float]:
        """Build a query vector from explicit term weights.

        Unlike :meth:`vectorize_query` this does not apply the sublinear ``1 +
        log(tf)`` damping. The damping exists to stop a term repeated twenty
        times in a document from dominating; a query model that already states
        "this term is worth 0.3" has no such runaway to control, and taking a
        logarithm of it would distort the distribution it was asked to honour.
        """
        vector: Dict[str, float] = {}
        for term, weight in weights.items():
            idf = self.idf.get(term)
            if idf is None or weight <= 0.0:
                continue
            vector[term] = weight * idf
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            for term in vector:
                vector[term] /= norm
        return vector

    def similarity(self, query_vector: Dict[str, float], doc_index: int) -> float:
        document = self.vectors[doc_index]
        if len(query_vector) > len(document):
            query_vector, document = document, query_vector
        return sum(
            weight * document.get(term, 0.0) for term, weight in query_vector.items()
        )

    def pairwise_similarity(self, left: int, right: int) -> float:
        """Cosine similarity between two indexed documents.

        The stored vectors are already L2-normalised, so the dot product is the
        cosine directly. Values are in [0, 1] because TF-IDF weights are
        non-negative, which is what lets MMR subtract this from a normalised
        relevance score.
        """
        if left == right:
            return 1.0
        first, second = self.vectors[left], self.vectors[right]
        if len(first) > len(second):
            first, second = second, first
        return sum(weight * second.get(term, 0.0) for term, weight in first.items())

    def search(self, query_tokens: Sequence[str], k: int = 10) -> List[Tuple[int, float]]:
        return self._rank(self.vectorize_query(query_tokens), k)

    def search_weighted(
        self, weights: Mapping[str, float], k: int = 10
    ) -> List[Tuple[int, float]]:
        """Rank documents against a weighted query model, best first."""
        return self._rank(self.vectorize_weights(weights), k)

    def _rank(self, query_vector: Dict[str, float], k: int) -> List[Tuple[int, float]]:
        if not query_vector:
            return []
        scored = []
        for index in range(self.doc_count):
            score = self.similarity(query_vector, index)
            if score > 0.0:
                scored.append((index, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
