"""Hybrid retrieval: BM25 + TF-IDF fused with Reciprocal Rank Fusion.

Lexical and vector retrieval fail differently. BM25 is unbeatable when the query
repeats the exact rare term used in the document; TF-IDF cosine is more
forgiving about wording and document length. Reciprocal Rank Fusion combines the
two ranked lists without needing their scores to be on a comparable scale:
each list contributes 1 / (rrf_k + rank) to every item it ranks. See Cormack,
Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual
Rank Learning Methods" (SIGIR 2009).

Fusion decides what is relevant; it says nothing about whether the top ``k``
are redundant with each other. Because the chunker overlaps adjacent chunks on
purpose, they frequently are. An optional MMR pass (see :mod:`ragkb.rerank`)
trades a little relevance for coverage when that matters.

Neither retriever can match a word the query does not contain, and fusing two
retrievers that both missed the same synonym changes nothing. An optional
pseudo-relevance feedback pass (see :mod:`ragkb.expansion`) borrows terms from
the first-pass results and retrieves again with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .bm25 import BM25Index
from .expansion import ExpansionConfig, expand_query
from .rerank import mean_pairwise_similarity, mmr_rerank
from .text import Chunk, tokenize
from .vector import TfidfIndex

METHODS = ("hybrid", "bm25", "vector")


@dataclass
class RetrievalResult:
    """A retrieved chunk plus the evidence for why it ranked where it did."""

    chunk: Chunk
    score: float
    method: str
    chunk_index: int = -1
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    mmr_rank: Optional[int] = None
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def text(self) -> str:
        return self.chunk.text


class HybridRetriever:
    """Retrieve chunks with BM25, TF-IDF, or the RRF fusion of both."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        rrf_k: int = 60,
        mmr_lambda: Optional[float] = None,
        expansion: Optional[ExpansionConfig] = None,
        indexes: Optional[Tuple[BM25Index, TfidfIndex]] = None,
    ) -> None:
        """``indexes`` supplies already-fitted (BM25, TF-IDF) indexes over
        exactly these chunks - the persistence layer's path around refitting.
        When omitted, both indexes are fitted here from the chunk tokens."""
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        self.chunks: List[Chunk] = list(chunks)
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.expansion = expansion
        if indexes is None:
            tokenised = [chunk.tokens for chunk in self.chunks]
            self.bm25 = BM25Index().fit(tokenised)
            self.vector = TfidfIndex().fit(tokenised)
        else:
            self.bm25, self.vector = indexes
            if self.bm25.doc_count != len(self.chunks) or self.vector.doc_count != len(self.chunks):
                raise ValueError("prebuilt indexes do not cover these chunks")

    def __len__(self) -> int:
        return len(self.chunks)

    def _rank_pool(
        self,
        method: str,
        pool_size: int,
        query_tokens: Optional[Sequence[str]] = None,
        weights: Optional[Mapping[str, float]] = None,
    ) -> tuple:
        """One retrieval pass: return ``(pool, bm25_ranks, vector_ranks)``.

        Exactly one of ``query_tokens`` and ``weights`` is supplied. The token
        path is the original, unweighted one and is what runs when expansion is
        off, so disabling expansion is bit-for-bit the behaviour that existed
        before it - the same property the ``lambda = 1.0`` row buys for MMR.
        """
        if weights is None:
            tokens = list(query_tokens or ())
            bm25_hits = self.bm25.search(tokens, k=pool_size)
            vector_hits = self.vector.search(tokens, k=pool_size)
        else:
            bm25_hits = self.bm25.search_weighted(weights, k=pool_size)
            vector_hits = self.vector.search_weighted(weights, k=pool_size)
        bm25_ranks = {index: rank for rank, (index, _) in enumerate(bm25_hits, start=1)}
        vector_ranks = {
            index: rank for rank, (index, _) in enumerate(vector_hits, start=1)
        }

        if method == "bm25":
            pool = list(bm25_hits)
        elif method == "vector":
            pool = list(vector_hits)
        else:
            fused: Dict[int, float] = {}
            for ranks in (bm25_ranks, vector_ranks):
                for index, rank in ranks.items():
                    fused[index] = fused.get(index, 0.0) + 1.0 / (self.rrf_k + rank)
            pool = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return pool, bm25_ranks, vector_ranks

    def expansion_model(
        self,
        query: str,
        method: str = "hybrid",
        candidates: int = 25,
        expansion: Optional[ExpansionConfig] = None,
    ) -> Dict[str, float]:
        """The weighted query model after one pseudo-relevance feedback pass.

        Exposed separately from :meth:`search` so a caller can show which terms
        were borrowed and how heavily - an expansion nobody can inspect is an
        expansion nobody can debug.
        """
        config = self.expansion if expansion is None else expansion
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks or config is None:
            return {}
        pool, _, _ = self._rank_pool(
            method, max(candidates, config.feedback_docs), query_tokens=query_tokens
        )
        return expand_query(
            query_tokens,
            pool,
            [chunk.tokens for chunk in self.chunks],
            self.bm25.idf,
            config,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
        candidates: int = 25,
        mmr_lambda: Optional[float] = None,
        expansion: Optional[ExpansionConfig] = None,
    ) -> List[RetrievalResult]:
        """Return the top ``k`` chunks for ``query`` under the given method.

        ``mmr_lambda`` and ``expansion`` override the retriever-level settings
        for one call. When ``expansion`` resolves to a config, a first pass
        supplies the feedback documents, terms are borrowed from them, and the
        pool is rebuilt from the expanded query. When ``mmr_lambda`` resolves to
        a value, the candidate pool is then re-ordered by Maximal Marginal
        Relevance before the top ``k`` are taken, so the returned passages cover
        more of the query instead of restating one passage.
        """
        if method not in METHODS:
            raise ValueError("method must be one of {}".format(METHODS))
        lambda_ = self.mmr_lambda if mmr_lambda is None else mmr_lambda
        if lambda_ is not None and not 0.0 <= lambda_ <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        config = self.expansion if expansion is None else expansion

        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        pool_size = max(candidates, k)
        pool, bm25_ranks, vector_ranks = self._rank_pool(
            method, pool_size, query_tokens=query_tokens
        )

        if config is not None:
            expanded = expand_query(
                query_tokens,
                pool,
                [chunk.tokens for chunk in self.chunks],
                self.bm25.idf,
                config,
            )
            # An expansion that borrowed nothing is not an expansion, and
            # re-running retrieval for it would only re-derive the same pool
            # through a different code path. Skipping keeps ``original_weight =
            # 1.0`` exactly equal to the unexpanded ranking, which is the
            # cheapest available correctness check on everything above.
            if set(expanded) - set(query_tokens):
                pool, bm25_ranks, vector_ranks = self._rank_pool(
                    method, pool_size, weights=expanded
                )

        if lambda_ is None:
            selected = pool[:k]
        else:
            selected = mmr_rerank(
                pool, self.vector.pairwise_similarity, k=k, lambda_=lambda_
            )

        return [
            self._build_result(
                index,
                score,
                method,
                bm25_ranks,
                vector_ranks,
                mmr_rank=position if lambda_ is not None else None,
            )
            for position, (index, score) in enumerate(selected, start=1)
        ]

    def _build_result(
        self,
        index: int,
        score: float,
        method: str,
        bm25_ranks: Dict[int, int],
        vector_ranks: Dict[int, int],
        mmr_rank: Optional[int],
    ) -> RetrievalResult:
        if method == "hybrid":
            components = {
                "bm25_rrf": 1.0 / (self.rrf_k + bm25_ranks[index])
                if index in bm25_ranks
                else 0.0,
                "vector_rrf": 1.0 / (self.rrf_k + vector_ranks[index])
                if index in vector_ranks
                else 0.0,
            }
        else:
            components = {method: score}
        return RetrievalResult(
            chunk=self.chunks[index],
            score=score,
            method=method,
            chunk_index=index,
            bm25_rank=bm25_ranks.get(index),
            vector_rank=vector_ranks.get(index),
            mmr_rank=mmr_rank,
            components=components,
        )

    def redundancy(self, results: Sequence[RetrievalResult]) -> float:
        """Mean pairwise cosine between the returned chunks.

        A diagnostic, not a quality metric: high values mean the result set is
        restating itself, which is the thing MMR exists to reduce.
        """
        ids = [result.chunk_index for result in results if result.chunk_index >= 0]
        return mean_pairwise_similarity(ids, self.vector.pairwise_similarity)
