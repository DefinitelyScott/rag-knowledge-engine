"""TF-IDF vector retrieval with cosine similarity.

This is the "semantic-ish" half of the hybrid retriever. It has no neural
embeddings and no external dependency: documents become sparse L2-normalised
TF-IDF vectors and are ranked by cosine similarity against the query vector.
It generalises better than BM25 across wording differences within a shared
vocabulary, and it fails in different ways, which is exactly what makes fusing
the two worthwhile.

Ranking is term-at-a-time over a postings list, for the reasons set out in
:mod:`ragkb.bm25`: a cosine between sparse vectors is a sum of per-term
products, so walking each query term's postings and accumulating into a
per-document total does exactly the work the dot products represent and none
of the work their zeros represent.

The saving is largest here, and largest of all for
:class:`ragkb.ngram.CharNgramIndex`, which inherits this loop unchanged. A word
query carries perhaps eight terms; the same query as 4-grams carries fifty, and
each document holds roughly six times as many features. Scoring
document-at-a-time makes that a fifty-lookup probe against every document in
the collection, most of them missing.

One caveat stated rather than glossed: :meth:`similarity` iterates whichever of
the two vectors is shorter, so it can add the same products in a different
order from this loop. The ranking is the same and the scores agree to within
floating-point summation order, but unlike the BM25 path they are not
guaranteed bit-identical, and ``tests/test_postings.py`` asserts the two
strengths separately rather than pretending they are one.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


class TfidfIndex:
    """Sparse TF-IDF index with cosine similarity search."""

    def __init__(self, sublinear_tf: bool = True) -> None:
        self.sublinear_tf = sublinear_tf
        self.doc_count = 0
        self.document_frequency: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vectors: List[Dict[str, float]] = []
        self.postings: Dict[str, List[Tuple[int, float]]] = {}

    def fit(self, documents: Sequence[Sequence[str]]) -> "TfidfIndex":
        """Fit over all documents. There is deliberately no incremental variant.

        :class:`~ragkb.bm25.BM25Index` gained :meth:`~ragkb.bm25.BM25Index.add_documents`
        because its per-document state is additive and only IDF is global -
        and IDF there is a scoring *coefficient*, applied at query time, so
        refreshing it is one pass over the vocabulary and nothing stored has
        to change.

        Here IDF is baked into the stored data. Each vector holds
        ``tf * idf`` and is then L2-normalised, so a single new document
        changes ``doc_count``, changes every IDF, and therefore changes both
        the weights *and* the norm of every vector already stored - including
        vectors for documents that share no term with the new one. Rebuilding
        them is the same traversal as fitting them, so an ``add_documents``
        here could only save the tokenisation, and it would cost an extra copy
        of the raw counters to do it.

        Offering the method anyway would be the worst option: an API named for
        an incremental update, doing a full refit, is a promise about cost that
        the implementation does not keep. ``evals/incremental.py`` measures
        what that asymmetry is worth on this corpus.
        """
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
        self.postings = self._build_postings()
        return self

    def _build_postings(self) -> Dict[str, List[Tuple[int, float]]]:
        """Rebuild ``term -> (document, weight)`` from the stored vectors.

        Built from the vectors rather than from the raw counters so the
        postings carry the *final* weights - already IDF-scaled and already
        L2-normalised. Scoring is then a multiply-and-add per posting with no
        further arithmetic, and terms ``_vectorize`` dropped for want of an IDF
        entry are absent here too, so the postings can never promise a document
        the scorer would not credit.
        """
        postings: Dict[str, List[Tuple[int, float]]] = {}
        for index, vector in enumerate(self.vectors):
            for term, weight in vector.items():
                postings.setdefault(term, []).append((index, weight))
        return postings

    def candidates(self, terms: Iterable[str]) -> List[int]:
        """Document indices sharing at least one term with ``terms``, ascending."""
        matched: Set[int] = set()
        for term in terms:
            matched.update(index for index, _ in self.postings.get(term, ()))
        return sorted(matched)

    def _accumulate(self, query_vector: Mapping[str, float]) -> Dict[int, float]:
        """Cosine against every reachable document, term-at-a-time.

        The stored vectors are L2-normalised, so the dot product accumulated
        here is the cosine directly - there is no per-document normalisation
        step to defer, which is what lets the whole ranking be one pass over
        postings.
        """
        totals: Dict[int, float] = {}
        for term, weight in query_vector.items():
            if not weight:
                continue
            for index, stored in self.postings.get(term, ()):
                totals[index] = totals.get(index, 0.0) + weight * stored
        return totals

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

    def to_dict(self) -> Dict:
        """Serialisable snapshot of the fitted state (see ``BM25Index.to_dict``)."""
        return {
            "sublinear_tf": self.sublinear_tf,
            "doc_count": self.doc_count,
            "document_frequency": dict(self.document_frequency),
            "idf": dict(self.idf),
            "vectors": [dict(vector) for vector in self.vectors],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TfidfIndex":
        """Restore a fitted index from :meth:`to_dict` output without refitting."""
        index = cls(sublinear_tf=data["sublinear_tf"])
        index.doc_count = data["doc_count"]
        index.document_frequency = dict(data["document_frequency"])
        index.idf = dict(data["idf"])
        index.vectors = [dict(vector) for vector in data["vectors"]]
        index.postings = index._build_postings()
        return index

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
        """Rank the documents sharing a term with ``query_vector``, best first.

        Ties break on the document index so the order does not depend on which
        term happened to reach a document first.
        """
        if not query_vector:
            return []
        scored = [
            (index, score)
            for index, score in self._accumulate(query_vector).items()
            if score > 0.0
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
