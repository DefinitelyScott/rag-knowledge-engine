"""ragkb - a dependency-free hybrid-retrieval RAG engine.

The package is deliberately built on the Python standard library only so that
every retrieval and scoring decision is inspectable end to end.
"""

from .text import Chunk, chunk_document, split_sentences, tokenize
from .bm25 import BM25Index
from .vector import TfidfIndex
from .ngram import CharNgramIndex, char_ngrams
from .expansion import ExpansionConfig, expand_query, relevance_model
from .rerank import mean_pairwise_similarity, mmr_rerank
from .retriever import HybridRetriever, RetrievalResult, resolve_fusion_weights
from .rankagree import (
    Agreement,
    TauResult,
    compare_rankings,
    kendall_tau_b,
    rank_biased_overlap,
    top_k_set_overlap,
)
from .attribution import (
    Attribution,
    ListAttribution,
    TermScore,
    attribute,
    attribute_bm25,
    attribute_vector,
    best_chunk_for_doc,
    query_model,
)
from .answerer import Answer, ExtractiveAnswerer, OpenAIAnswerer
from .abstain import AbstentionConfig, AnswerabilityReport, assess
from .holdout import (
    SelectionResult,
    cross_validated_selection,
    k_fold,
    repeated_selection,
    summarize,
)
from .engine import EngineConfig, RAGEngine, read_corpus
from .store import IndexFormatError, StaleIndexError, load_index, save_index
from .serve import DemoHandler, build_server, serve
from .bench import Timing, format_timings, measure, measure_one, percentile

__all__ = [
    "Chunk",
    "chunk_document",
    "split_sentences",
    "tokenize",
    "BM25Index",
    "TfidfIndex",
    "CharNgramIndex",
    "char_ngrams",
    "HybridRetriever",
    "resolve_fusion_weights",
    "RetrievalResult",
    "Attribution",
    "ListAttribution",
    "TermScore",
    "attribute",
    "attribute_bm25",
    "attribute_vector",
    "best_chunk_for_doc",
    "query_model",
    "mmr_rerank",
    "mean_pairwise_similarity",
    "Agreement",
    "TauResult",
    "compare_rankings",
    "kendall_tau_b",
    "rank_biased_overlap",
    "top_k_set_overlap",
    "ExpansionConfig",
    "expand_query",
    "relevance_model",
    "Answer",
    "ExtractiveAnswerer",
    "OpenAIAnswerer",
    "AbstentionConfig",
    "AnswerabilityReport",
    "assess",
    "SelectionResult",
    "cross_validated_selection",
    "k_fold",
    "repeated_selection",
    "summarize",
    "EngineConfig",
    "RAGEngine",
    "read_corpus",
    "save_index",
    "load_index",
    "StaleIndexError",
    "IndexFormatError",
    "build_server",
    "serve",
    "DemoHandler",
    "Timing",
    "measure",
    "measure_one",
    "percentile",
    "format_timings",
]

__version__ = "0.1.0"
