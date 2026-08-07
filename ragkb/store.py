"""Persist a fitted index to disk and load it back without re-fitting.

Building the engine does three jobs on every process start: read the corpus,
chunk it, and fit two indexes over the chunk tokens. For a CLI whose whole run
is a single query, all of that is recomputed work over inputs that did not
change. This module snapshots the fitted state - chunks, BM25 statistics,
TF-IDF vectors - into one JSON file and restores it directly, skipping
tokenisation and fitting entirely.

Two failure modes matter for a cache like this, and both are handled loudly
rather than silently:

* Staleness. The file records a fingerprint of the exact document texts plus
  the chunking parameters that produced them. :func:`load_index` recomputes
  that fingerprint from the corpus it is asked to trust and raises
  :class:`StaleIndexError` instead of serving quietly wrong rankings.
* Format drift. The file records a format name and version; loading an
  unrecognised one raises :class:`IndexFormatError` rather than guessing.

JSON keeps the artifact dependency-free, diffable and inspectable, and the
sparse structures involved (term counters, TF-IDF weight maps) serialise to it
naturally. The write is atomic - serialise to a sibling temp file, then
``os.replace`` - so a crash mid-save cannot leave a truncated index behind.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional

from .bm25 import BM25Index
from .engine import EngineConfig, RAGEngine
from .expansion import ExpansionConfig
from .retriever import HybridRetriever
from .text import Chunk
from .vector import TfidfIndex

FORMAT = "ragkb-index"
VERSION = 1

_CHUNK_FIELDS = (
    "doc_id",
    "chunk_id",
    "text",
    "token_count",
    "sentence_start",
    "sentence_end",
)


class IndexFormatError(ValueError):
    """The file is not an index this version of the code knows how to read."""


class StaleIndexError(ValueError):
    """The index was built from a corpus or config that no longer matches."""


def corpus_fingerprint(
    documents: Dict[str, str], target_tokens: int, overlap_tokens: int
) -> str:
    """SHA-256 over the document texts and the chunking parameters.

    The chunking parameters are part of the identity on purpose: the same
    corpus chunked at a different size produces a different index, and an
    index that no command could reproduce from the current inputs is stale.
    Query-time knobs (RRF k, MMR lambda, expansion) are deliberately excluded
    - they change ranking, not the fitted state, so they can be overridden on
    a loaded index without invalidating it.
    """
    payload = json.dumps(
        {
            "documents": documents,
            "target_tokens": target_tokens,
            "overlap_tokens": overlap_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_to_dict(config: EngineConfig) -> Dict:
    expansion = None
    if config.expansion is not None:
        expansion = {
            "feedback_docs": config.expansion.feedback_docs,
            "feedback_terms": config.expansion.feedback_terms,
            "original_weight": config.expansion.original_weight,
        }
    return {
        "target_tokens": config.target_tokens,
        "overlap_tokens": config.overlap_tokens,
        "rrf_k": config.rrf_k,
        "candidates": config.candidates,
        "mmr_lambda": config.mmr_lambda,
        "expansion": expansion,
    }


def _config_from_dict(data: Dict) -> EngineConfig:
    expansion = None
    if data.get("expansion") is not None:
        expansion = ExpansionConfig(**data["expansion"])
    return EngineConfig(
        target_tokens=data["target_tokens"],
        overlap_tokens=data["overlap_tokens"],
        rrf_k=data["rrf_k"],
        candidates=data["candidates"],
        mmr_lambda=data.get("mmr_lambda"),
        expansion=expansion,
    )


def _chunk_to_dict(chunk: Chunk) -> Dict:
    data = {name: getattr(chunk, name) for name in _CHUNK_FIELDS}
    data["metadata"] = chunk.metadata
    return data


def _chunk_from_dict(data: Dict) -> Chunk:
    return Chunk(
        doc_id=data["doc_id"],
        chunk_id=data["chunk_id"],
        text=data["text"],
        token_count=data["token_count"],
        sentence_start=data["sentence_start"],
        sentence_end=data["sentence_end"],
        metadata=data.get("metadata"),
    )


def save_index(engine: RAGEngine, path: str) -> None:
    """Write the engine's fitted state to ``path`` as one JSON document."""
    config = engine.config
    data = {
        "format": FORMAT,
        "version": VERSION,
        "fingerprint": corpus_fingerprint(
            engine.documents, config.target_tokens, config.overlap_tokens
        ),
        "config": _config_to_dict(config),
        "documents": engine.documents,
        "chunks": [_chunk_to_dict(chunk) for chunk in engine.chunks],
        "bm25": engine.retriever.bm25.to_dict(),
        "vector": engine.retriever.vector.to_dict(),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    os.replace(tmp, path)


def load_index(
    path: str, verify_corpus: Optional[Dict[str, str]] = None
) -> RAGEngine:
    """Restore a saved engine without re-chunking or re-fitting.

    When ``verify_corpus`` is given (a ``doc_id -> text`` mapping of the
    corpus the caller currently has), the stored fingerprint is checked
    against it and a mismatch raises :class:`StaleIndexError`. When it is
    ``None`` the index is trusted as-is, which is the right behaviour when
    the original corpus is not available at query time.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("format") != FORMAT:
        raise IndexFormatError(
            "not a ragkb index file: {}".format(path)
        )
    if data.get("version") != VERSION:
        raise IndexFormatError(
            "index version {} is not supported (expected {}); "
            "re-run `ragkb index` to rebuild".format(data.get("version"), VERSION)
        )

    config = _config_from_dict(data["config"])
    if verify_corpus is not None:
        expected = corpus_fingerprint(
            verify_corpus, config.target_tokens, config.overlap_tokens
        )
        if expected != data["fingerprint"]:
            raise StaleIndexError(
                "index at {} was built from a different corpus or chunking "
                "config; re-run `ragkb index` to rebuild".format(path)
            )

    chunks = [_chunk_from_dict(item) for item in data["chunks"]]
    retriever = HybridRetriever(
        chunks,
        rrf_k=config.rrf_k,
        mmr_lambda=config.mmr_lambda,
        expansion=config.expansion,
        indexes=(
            BM25Index.from_dict(data["bm25"]),
            TfidfIndex.from_dict(data["vector"]),
        ),
    )
    return RAGEngine(
        data["documents"], config=config, chunks=chunks, retriever=retriever
    )
