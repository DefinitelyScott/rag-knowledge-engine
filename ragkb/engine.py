"""The end-to-end engine: load a corpus, chunk it, index it, answer questions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .answerer import Answer, ExtractiveAnswerer
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


class RAGEngine:
    """Chunk a set of documents, index them, and answer questions with citations."""

    def __init__(
        self,
        documents: Dict[str, str],
        config: Optional[EngineConfig] = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.documents = dict(documents)
        self.chunks: List[Chunk] = []
        for doc_id in sorted(self.documents):
            self.chunks.extend(
                chunk_document(
                    doc_id,
                    self.documents[doc_id],
                    target_tokens=self.config.target_tokens,
                    overlap_tokens=self.config.overlap_tokens,
                )
            )
        self.retriever = HybridRetriever(self.chunks, rrf_k=self.config.rrf_k)

    @classmethod
    def from_corpus(
        cls, path: str, config: Optional[EngineConfig] = None
    ) -> "RAGEngine":
        """Load every ``.md`` / ``.txt`` file in ``path`` as one document."""
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
        return cls(documents, config=config)

    def search(
        self, query: str, k: int = 5, method: str = "hybrid"
    ) -> List[RetrievalResult]:
        return self.retriever.search(
            query, k=k, method=method, candidates=self.config.candidates
        )

    def answer(
        self,
        query: str,
        k: int = 4,
        method: str = "hybrid",
        answerer=None,
    ) -> Answer:
        results = self.search(query, k=k, method=method)
        return (answerer or ExtractiveAnswerer()).answer(query, results)

    def ranked_doc_ids(
        self, query: str, k: int = 5, method: str = "hybrid"
    ) -> List[str]:
        """Deduplicated document ids in rank order - the unit evaluation uses."""
        ordered: List[str] = []
        for result in self.search(query, k=max(k * 4, k), method=method):
            if result.doc_id not in ordered:
                ordered.append(result.doc_id)
            if len(ordered) >= k:
                break
        return ordered

    def stats(self) -> Dict[str, float]:
        token_counts = [chunk.token_count for chunk in self.chunks]
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "vocabulary": len(self.retriever.bm25.document_frequency),
            "mean_chunk_tokens": (
                sum(token_counts) / len(token_counts) if token_counts else 0.0
            ),
            "min_chunk_tokens": min(token_counts) if token_counts else 0,
            "max_chunk_tokens": max(token_counts) if token_counts else 0,
        }
