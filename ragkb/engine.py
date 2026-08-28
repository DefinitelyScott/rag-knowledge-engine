"""The end-to-end engine: load a corpus, chunk it, index it, answer questions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from .abstain import REFUSAL, AbstentionConfig, AnswerabilityReport, assess
from .answerer import Answer, ExtractiveAnswerer
from .crossencoder import RerankConfig
from .expansion import ExpansionConfig
from .retriever import HybridRetriever, RetrievalResult
from .text import Chunk, chunk_document

TEXT_EXTENSIONS = (".md", ".txt")


@dataclass
class EngineConfig:
    """Every knob that changes retrieval behaviour, in one place."""

    target_tokens: int = 120
    overlap_tokens: int = 30
    rrf_k: int = 60
    candidates: int = 25
    #: MMR relevance/diversity trade-off. ``None`` disables re-ranking and
    #: keeps the pure fusion ordering; 0.7 is a reasonable starting point.
    mmr_lambda: Optional[float] = None
    #: Pseudo-relevance feedback settings. ``None`` disables expansion and runs
    #: the query exactly as the user wrote it.
    expansion: Optional[ExpansionConfig] = None
    #: Second-stage re-ranking settings. ``None`` disables the cross-encoder
    #: stage and returns the first-stage ordering.
    rerank: Optional[RerankConfig] = None
    #: n for the optional character n-gram index (see :mod:`ragkb.ngram`).
    #: ``None`` leaves it out and fuses only BM25 and TF-IDF; 4 is the usual
    #: value. Unlike the query-time knobs this one is index-time - it fits a
    #: third index - which is why it belongs here next to the chunk sizes.
    ngram: Optional[int] = None
    #: Per-list weights in the RRF sum, e.g. ``{"ngram": 2.0}`` to let the
    #: character n-gram list count double. ``None`` weights every list equally,
    #: which is textbook RRF and the behaviour that existed before weighting.
    #: Query-time, not index-time: changing it re-fuses existing ranked lists
    #: and never re-fits an index, so a persisted index stays valid.
    fusion_weights: Optional[Mapping[str, float]] = None
    #: Abstention gate. ``None`` - the default - always answers, which is the
    #: behaviour that existed before the gate. Supplying a config makes
    #: :meth:`RAGEngine.answer` decline when the retrieved evidence scores
    #: below its threshold.
    abstain: Optional[AbstentionConfig] = None


def read_corpus(path: str) -> Dict[str, str]:
    """Read every ``.md`` / ``.txt`` file in ``path`` into a ``doc_id -> text`` map."""
    if not os.path.isdir(path):
        raise FileNotFoundError("corpus directory not found: {}".format(path))
    documents: Dict[str, str] = {}
    for name in sorted(os.listdir(path)):
        if not name.lower().endswith(TEXT_EXTENSIONS):
            continue
        full = os.path.join(path, name)
        with open(full, "r", encoding="utf-8") as handle:
            documents[os.path.splitext(name)[0]] = handle.read()
    if not documents:
        raise ValueError("no .md or .txt documents found in {}".format(path))
    return documents


class RAGEngine:
    """Chunk a set of documents, index them, and answer questions with citations."""

    def __init__(
        self,
        documents: Dict[str, str],
        config: Optional[EngineConfig] = None,
        chunks: Optional[List[Chunk]] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> None:
        """``chunks`` and ``retriever`` accept precomputed state so that
        :meth:`load` can restore an engine without re-chunking or re-fitting;
        normal construction leaves both ``None`` and builds them here."""
        self.config = config or EngineConfig()
        self.documents = dict(documents)
        if chunks is None:
            chunks = []
            for doc_id in sorted(self.documents):
                chunks.extend(
                    chunk_document(
                        doc_id,
                        self.documents[doc_id],
                        target_tokens=self.config.target_tokens,
                        overlap_tokens=self.config.overlap_tokens,
                    )
                )
        self.chunks: List[Chunk] = list(chunks)
        self.retriever = retriever or HybridRetriever(
            self.chunks,
            rrf_k=self.config.rrf_k,
            mmr_lambda=self.config.mmr_lambda,
            expansion=self.config.expansion,
            rerank=self.config.rerank,
            ngram=self.config.ngram,
            fusion_weights=self.config.fusion_weights,
        )

    @classmethod
    def from_corpus(
        cls, path: str, config: Optional[EngineConfig] = None
    ) -> "RAGEngine":
        """Load every ``.md`` / ``.txt`` file in ``path`` as one document."""
        return cls(read_corpus(path), config=config)

    def add_document(self, doc_id: str, text: str) -> List[Chunk]:
        """Add one document to a live engine and return the chunks it produced.

        Chunking uses this engine's configured sizes, so the new document is
        split exactly as the rest of the corpus was. The chunks are appended
        and handed to :meth:`ragkb.retriever.HybridRetriever.add_chunks`, which
        extends BM25 in place and refits the vector lists - see that method for
        why the two halves differ and what appending costs in tie-break order.

        Re-adding an existing ``doc_id`` raises rather than replacing it.
        Replacement is a different operation: the old document's chunks would
        have to be *removed*, and removal renumbers every chunk after them,
        which no postings list here survives. Growing the index is supported;
        editing it is a rebuild, and saying so is better than a method that
        silently double-indexes a document under one id.

        A persisted index stays consistent across this: :meth:`save` writes the
        fingerprint of ``self.documents``, which now includes the new document.
        """
        if doc_id in self.documents:
            raise ValueError("document already indexed: {}".format(doc_id))
        if not text.strip():
            raise ValueError("refusing to index an empty document: {}".format(doc_id))
        new_chunks = chunk_document(
            doc_id,
            text,
            target_tokens=self.config.target_tokens,
            overlap_tokens=self.config.overlap_tokens,
        )
        if not new_chunks:
            raise ValueError("document produced no chunks: {}".format(doc_id))
        self.documents[doc_id] = text
        self.retriever.add_chunks(new_chunks)
        self.chunks = self.retriever.chunks
        return new_chunks

    def save(self, path: str) -> None:
        """Persist the fitted index to ``path`` (see :mod:`ragkb.store`)."""
        from . import store

        store.save_index(self, path)

    @classmethod
    def load(
        cls, path: str, verify_corpus: Optional[Dict[str, str]] = None
    ) -> "RAGEngine":
        """Restore an engine saved with :meth:`save`, skipping the fit.

        ``verify_corpus`` optionally supplies the current corpus documents;
        a fingerprint mismatch then raises ``ragkb.store.StaleIndexError``
        instead of answering from stale data.
        """
        from . import store

        return store.load_index(path, verify_corpus=verify_corpus)

    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
        mmr_lambda: Optional[float] = None,
        expansion: Optional[ExpansionConfig] = None,
        rerank: Optional[RerankConfig] = None,
        fusion_weights: Optional[Mapping[str, float]] = None,
    ) -> List[RetrievalResult]:
        return self.retriever.search(
            query,
            k=k,
            method=method,
            candidates=self.config.candidates,
            mmr_lambda=mmr_lambda,
            expansion=expansion,
            rerank=rerank,
            fusion_weights=fusion_weights,
        )

    def answerability(
        self,
        query: str,
        k: int = 4,
        method: str = "hybrid",
        abstain: Optional[AbstentionConfig] = None,
    ) -> AnswerabilityReport:
        """How well the retrieved evidence covers ``query`` (see :mod:`ragkb.abstain`).

        Exposed on its own so the decision can be inspected without generating
        an answer - a gate whose reasoning is not visible is a gate nobody can
        tune.
        """
        results = self.search(query, k=k, method=method)
        return assess(self.retriever, query, results, config=abstain or self.config.abstain)

    def answer(
        self,
        query: str,
        k: int = 4,
        method: str = "hybrid",
        answerer=None,
        abstain: Optional[AbstentionConfig] = None,
    ) -> Answer:
        """Answer ``query`` from the corpus, optionally declining to.

        ``abstain`` (falling back to ``config.abstain``) gates generation: when
        the retrieved evidence scores below its threshold the answerer is never
        called and the returned :class:`~ragkb.answerer.Answer` carries no
        citations, because there is nothing it would be honest to cite.
        """
        results = self.search(query, k=k, method=method)
        config = abstain if abstain is not None else self.config.abstain
        if config is not None:
            report = assess(self.retriever, query, results, config=config)
            if not report.answerable:
                return Answer(
                    text=REFUSAL,
                    citations=[],
                    method="abstained",
                    abstained=True,
                    answerability=report,
                )
            answer = (answerer or ExtractiveAnswerer()).answer(query, results)
            answer.answerability = report
            return answer
        return (answerer or ExtractiveAnswerer()).answer(query, results)

    def ranked_doc_ids(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
        mmr_lambda: Optional[float] = None,
        expansion: Optional[ExpansionConfig] = None,
        rerank: Optional[RerankConfig] = None,
        fusion_weights: Optional[Mapping[str, float]] = None,
    ) -> List[str]:
        """Deduplicated document ids in rank order - the unit evaluation uses."""
        ordered: List[str] = []
        for result in self.search(
            query,
            k=max(k * 4, k),
            method=method,
            mmr_lambda=mmr_lambda,
            expansion=expansion,
            rerank=rerank,
            fusion_weights=fusion_weights,
        ):
            if result.doc_id not in ordered:
                ordered.append(result.doc_id)
            if len(ordered) >= k:
                break
        return ordered

    def stats(self) -> Dict[str, float]:
        token_counts = [chunk.token_count for chunk in self.chunks]
        stats = {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "vocabulary": len(self.retriever.bm25.document_frequency),
            "mean_chunk_tokens": (
                sum(token_counts) / len(token_counts) if token_counts else 0.0
            ),
            "min_chunk_tokens": min(token_counts) if token_counts else 0,
            "max_chunk_tokens": max(token_counts) if token_counts else 0,
        }
        if self.retriever.ngram is not None:
            # Reported next to the word vocabulary on purpose: the ratio is the
            # price of the third list, and it is large enough that guessing at
            # it would be misleading.
            stats["ngram_vocabulary"] = len(self.retriever.ngram.document_frequency)
        return stats
