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

Everything above scores the query and the passage independently, so none of it
can see where the matched terms sit relative to each other. An optional second
stage (see :mod:`ragkb.crossencoder`) re-orders the candidate pool with a
scorer that reads the pair jointly - the standard retrieve-then-rerank shape,
with a scorer cheap enough to run without a model.

BM25 and TF-IDF are two scores over one *representation*: both index whole
words, so both miss the same misspelling and the same morphological variant.
An optional third list (see :mod:`ragkb.ngram`) indexes character n-grams
instead, and joins the same fusion - the only change here that gives RRF lists
that can genuinely disagree.

Textbook RRF weights every list identically, which quietly assumes they are
equally trustworthy. On this corpus they are not: `evals/evaluate.py` measures
the character n-gram list alone at nDCG@5 0.868 against 0.834 for the
three-list fusion that contains it, so uniform fusion actively *dilutes* the
strongest list with two weaker ones. :func:`resolve_fusion_weights` lets each
list carry its own multiplier - the standard weighted-RRF generalisation,
``sum_i w_i / (rrf_k + rank_i)`` - so that finding can be acted on instead of
only reported. Weights are rescaled to mean 1.0, which means only their
*ratios* matter, uniform weights reproduce plain RRF bit-for-bit, and fused
scores stay on the scale that ``rrf_k`` and the abstention threshold were
calibrated against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .bm25 import BM25Index
from .crossencoder import RerankConfig
from .expansion import ExpansionConfig, expand_query
from .ngram import CharNgramIndex
from .rerank import mean_pairwise_similarity, mmr_rerank
from .text import Chunk, tokenize
from .vector import TfidfIndex

#: The methods always available. ``ngram`` joins them only when the retriever
#: was built with an n-gram index, so a method name can never silently resolve
#: to a list that was never fitted - see :attr:`HybridRetriever.methods`.
METHODS = ("hybrid", "bm25", "vector")
NGRAM_METHOD = "ngram"


def resolve_fusion_weights(
    weights: Optional[Mapping[str, float]], names: Sequence[str]
) -> Dict[str, float]:
    """Resolve per-list RRF weights over ``names``, rescaled to mean 1.0.

    ``None`` and any uniform mapping both resolve to all-1.0, so the weighted
    fusion is bit-for-bit the unweighted one unless someone asks for something
    else - the same identity property ``mmr_lambda = 1.0`` and ``ngram=None``
    carry elsewhere in this module.

    Rescaling to mean 1.0 is the load-bearing choice. A fused RRF score is a
    sum of ``w / (rrf_k + rank)`` terms, so scaling every weight by c scales
    every fused score by c while leaving the *ordering* untouched. Ordering is
    all retrieval cares about, but two things downstream read the magnitude:
    ``rrf_k``, tuned in `evals/tune.py` against fused scores, and the
    abstention threshold in :mod:`ragkb.abstain`, which compares a fused score
    to a constant. Fixing the mean pins the scale so neither has to be
    re-tuned when the ratios change, and makes ``{bm25: 2, vector: 2}``
    correctly mean the same thing as ``{bm25: 1, vector: 1}``.

    Lists absent from ``weights`` default to 1.0 rather than to 0.0: omitting a
    list should mean "no opinion", not "drop it". Dropping one is still
    expressible, by weighting it 0.0 explicitly.
    """
    if not names:
        raise ValueError("fusion needs at least one list")
    if weights is None:
        return {name: 1.0 for name in names}
    unknown = sorted(set(weights) - set(names))
    if unknown:
        raise ValueError(
            "unknown fusion list(s) {}; this retriever fuses {}".format(
                ", ".join(unknown), ", ".join(names)
            )
        )
    resolved = {name: float(weights.get(name, 1.0)) for name in names}
    negative = sorted(name for name, value in resolved.items() if value < 0.0)
    if negative:
        raise ValueError(
            "fusion weights must be non-negative; got negative for {}".format(
                ", ".join(negative)
            )
        )
    total = sum(resolved.values())
    if total <= 0.0:
        raise ValueError("at least one fusion weight must be positive")
    scale = len(resolved) / total
    return {name: value * scale for name, value in resolved.items()}


@dataclass
class RetrievalResult:
    """A retrieved chunk plus the evidence for why it ranked where it did."""

    chunk: Chunk
    score: float
    method: str
    chunk_index: int = -1
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    ngram_rank: Optional[int] = None
    mmr_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
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
        rerank: Optional[RerankConfig] = None,
        ngram: Optional[int] = None,
        fusion_weights: Optional[Mapping[str, float]] = None,
        indexes: Optional[Tuple[BM25Index, TfidfIndex]] = None,
        ngram_index: Optional[CharNgramIndex] = None,
    ) -> None:
        """``indexes`` supplies already-fitted (BM25, TF-IDF) indexes over
        exactly these chunks - the persistence layer's path around refitting.
        When omitted, both indexes are fitted here from the chunk tokens.

        ``ngram`` is the n of the optional character n-gram index; ``None``
        leaves it out entirely, which is bit-for-bit the two-list fusion that
        existed before it - the same identity property ``mmr_lambda = 1.0``
        buys for MMR. ``ngram_index`` supplies an already-fitted one.

        ``fusion_weights`` maps a list name to its weight in the RRF sum; see
        :func:`resolve_fusion_weights`. It is validated after the indexes are
        built, because whether ``ngram`` is a legal key depends on whether an
        n-gram index was actually fitted.
        """
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        self.chunks: List[Chunk] = list(chunks)
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.expansion = expansion
        self.rerank = rerank
        tokenised = None
        if indexes is None:
            tokenised = [chunk.tokens for chunk in self.chunks]
            self.bm25 = BM25Index().fit(tokenised)
            self.vector = TfidfIndex().fit(tokenised)
        else:
            self.bm25, self.vector = indexes
            if self.bm25.doc_count != len(self.chunks) or self.vector.doc_count != len(self.chunks):
                raise ValueError("prebuilt indexes do not cover these chunks")
        if ngram_index is not None:
            self.ngram: Optional[CharNgramIndex] = ngram_index
        elif ngram is None:
            self.ngram = None
        else:
            if tokenised is None:
                tokenised = [chunk.tokens for chunk in self.chunks]
            self.ngram = CharNgramIndex(n=ngram).fit(tokenised)
        if self.ngram is not None and self.ngram.doc_count != len(self.chunks):
            raise ValueError("prebuilt indexes do not cover these chunks")
        self.fusion_weights = resolve_fusion_weights(
            fusion_weights, [name for name, _ in self._lists()]
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> "HybridRetriever":
        """Index new chunks, updating what can be updated and refitting what cannot.

        BM25 is extended in place; the TF-IDF list - and the n-gram list, which
        is a TF-IDF index over a different analyzer - are refitted over every
        chunk, for the reason set out in :meth:`ragkb.vector.TfidfIndex.fit`.
        The mixture is the honest one: this method is cheaper than building a
        retriever from scratch, and not by as much as "incremental" suggests.

        New chunks are *appended*, not merged into corpus order. That is the
        whole point - keeping the chunk list sorted by document id would
        renumber existing chunks, and every posting in every index refers to a
        chunk by its index, so a sorted insert is a refit wearing a different
        name. The visible consequence is that an engine grown by
        :meth:`ragkb.engine.RAGEngine.add_document` holds its chunks in
        insertion order rather than corpus order. Scores are unaffected, but
        ranking ties break on the chunk index, so a tie between two equally
        scored chunks can resolve differently than in a from-scratch build over
        the same documents. When the added document sorts last the two orders
        coincide and the rankings are identical, which is what
        ``tests/test_incremental.py`` pins down.
        """
        new = list(chunks)
        if not new:
            return self
        self.bm25.add_documents([chunk.tokens for chunk in new])
        self.chunks.extend(new)
        tokenised = [chunk.tokens for chunk in self.chunks]
        self.vector.fit(tokenised)
        if self.ngram is not None:
            self.ngram.fit(tokenised)
        return self

    @property
    def methods(self) -> Tuple[str, ...]:
        """Retrieval methods this retriever can actually serve."""
        if self.ngram is None:
            return METHODS
        return METHODS + (NGRAM_METHOD,)

    def _lists(self) -> List[Tuple[str, object]]:
        """The (name, index) pairs that feed the fusion, in fusion order."""
        lists: List[Tuple[str, object]] = [("bm25", self.bm25), ("vector", self.vector)]
        if self.ngram is not None:
            lists.append((NGRAM_METHOD, self.ngram))
        return lists

    def _rank_pool(
        self,
        method: str,
        pool_size: int,
        query_tokens: Optional[Sequence[str]] = None,
        weights: Optional[Mapping[str, float]] = None,
        fusion_weights: Optional[Mapping[str, float]] = None,
    ) -> tuple:
        """One retrieval pass: return ``(pool, ranks)``.

        ``ranks`` maps each list's name to its ``chunk index -> rank`` mapping,
        so the fusion arithmetic below does not care how many lists there are.
        That is what lets the n-gram list join without a special case, and why
        the two-list and three-list fusions are literally the same code path.

        Exactly one of ``query_tokens`` and ``weights`` is supplied. The token
        path is the original, unweighted one and is what runs when expansion is
        off, so disabling expansion is bit-for-bit the behaviour that existed
        before it - the same property the ``lambda = 1.0`` row buys for MMR.

        ``fusion_weights`` overrides the retriever-level weights for one call;
        ``None`` uses them. Returns the resolved weights alongside the pool so
        the caller can attribute each list's contribution without re-resolving.
        """
        resolved = (
            self.fusion_weights
            if fusion_weights is None
            else resolve_fusion_weights(
                fusion_weights, [name for name, _ in self._lists()]
            )
        )
        hits: Dict[str, List[Tuple[int, float]]] = {}
        for name, index in self._lists():
            if weights is None:
                hits[name] = index.search(list(query_tokens or ()), k=pool_size)
            else:
                hits[name] = index.search_weighted(weights, k=pool_size)
        ranks = {
            name: {index: rank for rank, (index, _) in enumerate(hit_list, start=1)}
            for name, hit_list in hits.items()
        }

        if method in hits:
            pool = list(hits[method])
        else:
            fused: Dict[int, float] = {}
            for name in hits:
                weight = resolved[name]
                if weight == 0.0:
                    continue
                for index, rank in ranks[name].items():
                    fused[index] = fused.get(index, 0.0) + weight / (self.rrf_k + rank)
            pool = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return pool, ranks, resolved

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
        rerank: Optional[RerankConfig] = None,
        fusion_weights: Optional[Mapping[str, float]] = None,
    ) -> List[RetrievalResult]:
        """Return the top ``k`` chunks for ``query`` under the given method.

        ``mmr_lambda``, ``expansion``, ``rerank`` and ``fusion_weights``
        override the retriever-level settings for one call. The pipeline runs
        in this order:

        1. rank a candidate pool with ``method``;
        2. if ``expansion`` resolves to a config, borrow terms from that pool
           and rebuild it from the expanded query;
        3. if ``rerank`` resolves to a config, re-order the pool with a
           cross-encoder-style scorer that reads the (query, passage) pair
           jointly;
        4. if ``mmr_lambda`` resolves to a value, re-order again by Maximal
           Marginal Relevance so the returned passages cover more of the query
           instead of restating one passage;
        5. take the top ``k``.

        Re-ranking runs before MMR on purpose: the second stage is a better
        relevance judgement than fusion alone, and MMR's whole job is to trade
        relevance against diversity, which it can only do honestly if the
        relevance it is handed is the best one available.
        """
        if method not in self.methods:
            raise ValueError("method must be one of {}".format(self.methods))
        lambda_ = self.mmr_lambda if mmr_lambda is None else mmr_lambda
        if lambda_ is not None and not 0.0 <= lambda_ <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        config = self.expansion if expansion is None else expansion
        rerank_config = self.rerank if rerank is None else rerank

        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        pool_size = max(candidates, k)
        pool, ranks, resolved_weights = self._rank_pool(
            method, pool_size, query_tokens=query_tokens, fusion_weights=fusion_weights
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
                pool, ranks, resolved_weights = self._rank_pool(
                    method, pool_size, weights=expanded, fusion_weights=fusion_weights
                )

        reranked_positions: Dict[int, int] = {}
        if rerank_config is not None:
            pool = rerank_config.build(self.bm25.idf).rerank(
                query_tokens,
                pool,
                [chunk.tokens for chunk in self.chunks],
                k=None,
                alpha=rerank_config.alpha,
            )
            reranked_positions = {
                index: position for position, (index, _) in enumerate(pool, start=1)
            }

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
                ranks,
                resolved_weights,
                mmr_rank=position if lambda_ is not None else None,
                rerank_rank=reranked_positions.get(index),
            )
            for position, (index, score) in enumerate(selected, start=1)
        ]

    def _build_result(
        self,
        index: int,
        score: float,
        method: str,
        ranks: Mapping[str, Mapping[int, int]],
        fusion_weights: Mapping[str, float],
        mmr_rank: Optional[int],
        rerank_rank: Optional[int] = None,
    ) -> RetrievalResult:
        # ``components`` always describes the *first* stage: which list ranked
        # this chunk where, and what that contributed to the fusion. When a
        # second stage ran, ``score`` is the blended re-ranking score rather
        # than the fusion score, and ``rerank_rank`` says where the second
        # stage put it - so the two together still explain the final position.
        #
        # Each component is the *weighted* term, so the components still sum to
        # ``score``. Reporting the unweighted 1/(rrf_k + rank) would make the
        # arithmetic in `ragkb.cli explain` stop adding up the moment anyone
        # set a non-uniform weight, which is exactly when they most need it to.
        if method == "hybrid":
            components = {
                "{}_rrf".format(name): (
                    fusion_weights.get(name, 1.0) / (self.rrf_k + per_list[index])
                    if index in per_list
                    else 0.0
                )
                for name, per_list in ranks.items()
            }
        else:
            components = {method: score}
        return RetrievalResult(
            chunk=self.chunks[index],
            score=score,
            method=method,
            chunk_index=index,
            bm25_rank=ranks["bm25"].get(index),
            vector_rank=ranks["vector"].get(index),
            ngram_rank=ranks.get(NGRAM_METHOD, {}).get(index),
            mmr_rank=mmr_rank,
            rerank_rank=rerank_rank,
            components=components,
        )

    def redundancy(self, results: Sequence[RetrievalResult]) -> float:
        """Mean pairwise cosine between the returned chunks.

        A diagnostic, not a quality metric: high values mean the result set is
        restating itself, which is the thing MMR exists to reduce.
        """
        ids = [result.chunk_index for result in results if result.chunk_index >= 0]
        return mean_pairwise_similarity(ids, self.vector.pairwise_similarity)
