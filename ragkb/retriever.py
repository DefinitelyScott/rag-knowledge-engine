"""Hybrid retrieval: BM25 + TF-IDF fused with Reciprocal Rank Fusion.

Lexical and vector retrieval fail differently. BM25 is unbeatable when the query
repeats the exact rare term used in the document; TF-IDF cosine is more
forgiving about wording and document length. Reciprocal Rank Fusion combines the
two ranked lists without needing their scores to be on a comparable scale:
each list contributes 1 / (rrf_k + rank) to every item it ranks. See Cormack,
Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual
Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .bm25 import BM25Index
from .text import Chunk, tokenize
from .vector import TfidfIndex

METHODS = ("hybrid", "bm25", "vector")


@dataclass
class RetrievalResult:
    """A retrieved chunk plus the evidence for why it ranked where it did."""

    chunk: Chunk
    score: float
    method: str
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def text(self) -> str:
        return self.chunk.text


class HybridRetriever:
    """Retrieve chunks with BM25, TF-IDF, or the RRF fusion of both."""

    def __init__(self, chunks: Sequence[Chunk], rrf_k: int = 60) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.chunks: List[Chunk] = list(chunks)
        self.rrf_k = rrf_k
        tokenised = [chunk.tokens for chunk in self.chunks]
        self.bm25 = BM25Index().fit(tokenised)
        self.vector = TfidfIndex().fit(tokenised)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
        candidates: int = 25,
    ) -> List[RetrievalResult]:
        """Return the top ``k`` chunks for ``query`` under the given method."""
        if method not in METHODS:
            raise ValueError("method must be one of {}".format(METHODS))
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        pool = max(candidates, k)
        bm25_hits = self.bm25.search(query_tokens, k=pool)
        vector_hits = self.vector.search(query_tokens, k=pool)

        if method == "bm25":
            return self._single(bm25_hits, "bm25", k)
        if method == "vector":
            return self._single(vector_hits, "vector", k)

        bm25_ranks = {index: rank for rank, (index, _) in enumerate(bm25_hits, start=1)}
        vector_ranks = {
            index: rank for rank, (index, _) in enumerate(vector_hits, start=1)
        }

        fused: Dict[int, float] = {}
        for ranks in (bm25_ranks, vector_ranks):
            for index, rank in ranks.items():
                fused[index] = fused.get(index, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
        results = []
        for index, score in ordered:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=score,
                    method="hybrid",
                    bm25_rank=bm25_ranks.get(index),
                    vector_rank=vector_ranks.get(index),
                    components={
                        "bm25_rrf": 1.0 / (self.rrf_k + bm25_ranks[index])
                        if index in bm25_ranks
                        else 0.0,
                        "vector_rrf": 1.0 / (self.rrf_k + vector_ranks[index])
                        if index in vector_ranks
                        else 0.0,
                    },
                )
            )
        return results

    def _single(self, hits, method: str, k: int) -> List[RetrievalResult]:
        results = []
        for rank, (index, score) in enumerate(hits[:k], start=1):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=score,
                    method=method,
                    bm25_rank=rank if method == "bm25" else None,
                    vector_rank=rank if method == "vector" else None,
                    components={method: score},
                )
            )
        return results
