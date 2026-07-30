"""ragkb - a dependency-free hybrid-retrieval RAG engine.

The package is deliberately built on the Python standard library only so that
every retrieval and scoring decision is inspectable end to end.
"""

from .text import Chunk, chunk_document, split_sentences, tokenize
from .bm25 import BM25Index
from .vector import TfidfIndex
from .rerank import mean_pairwise_similarity, mmr_rerank
from .retriever import HybridRetriever, RetrievalResult
from .answerer import Answer, ExtractiveAnswerer, OpenAIAnswerer
from .engine import EngineConfig, RAGEngine

__all__ = [
    "Chunk",
    "chunk_document",
    "split_sentences",
    "tokenize",
    "BM25Index",
    "TfidfIndex",
    "HybridRetriever",
    "RetrievalResult",
    "mmr_rerank",
    "mean_pairwise_similarity",
    "Answer",
    "ExtractiveAnswerer",
    "OpenAIAnswerer",
    "EngineConfig",
    "RAGEngine",
]

__version__ = "0.1.0"
