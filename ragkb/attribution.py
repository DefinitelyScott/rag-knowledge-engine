"""Per-term score attribution: why *this* chunk landed at *that* rank.

Every table in the README is an aggregate. Aggregates tell you that fusion
buys 0.015 nDCG on the hard slice; they cannot tell you why one particular
question put `reranking.md` above `late-interaction.md`, and they are useless
when the answer is wrong and you want to know which word did it.

This module decomposes a single (query, chunk) score back into the pieces the
scorers actually summed. Both retrievers are additive over terms, which is the
whole reason a decomposition exists at all:

* BM25 sums ``weight * idf(t) * tf(t)(k1+1) / (tf(t) + norm)`` over query terms;
* TF-IDF cosine sums ``query_vector[t] * doc_vector[t]`` over shared terms.

So the per-term contributions here are not an approximation or a saliency
heuristic - they add up to the score exactly, and
:mod:`tests.test_attribution` asserts that identity rather than trusting it.
The n-gram list decomposes the same way because
:class:`~ragkb.ngram.CharNgramIndex` *is* a TF-IDF index whose features happen
to be character n-grams, so its rows read as grams instead of words.

Reciprocal rank fusion is additive too, in ranks rather than scores, so the
same treatment applies one level up: each list contributes
``1 / (rrf_k + rank)`` and those contributions sum to the fused score.

The negative space matters as much as the positive. A query term that scored
nothing did so for one of two distinct reasons, and the fix differs:

``missing``
    the term exists in the corpus but not in this chunk - a chunking or
    ranking problem, and the sort of thing MMR or a bigger candidate pool can
    move.

``unknown``
    the term is not in the collection vocabulary at all - no amount of
    re-ranking will help, because there is nothing to match. This is the
    signature the n-gram list and query expansion exist to attack.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .bm25 import BM25Index
from .ngram import CharNgramIndex, char_ngrams
from .retriever import NGRAM_METHOD, HybridRetriever
from .text import tokenize
from .vector import TfidfIndex


@dataclass(frozen=True)
class TermScore:
    """One query term's contribution to one list's score for one chunk.

    ``term_frequency`` is ``None`` for the vector lists on purpose. BM25 scores
    a raw count; TF-IDF cosine scores an L2-normalised weight in which the
    original count is no longer recoverable, and printing a 1 there would be a
    number the scorer never used.
    """

    term: str
    weight: float
    document_frequency: int
    term_frequency: Optional[int]
    idf: float
    contribution: float


@dataclass
class ListAttribution:
    """One retrieval list's score for a chunk, broken down by term."""

    name: str
    score: float
    rank: Optional[int]
    terms: List[TermScore] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)

    @property
    def attributed(self) -> float:
        """The contributions summed - equal to :attr:`score` up to float error."""
        return sum(term.contribution for term in self.terms)


@dataclass
class Attribution:
    """A full account of one chunk's position under one method."""

    query: str
    chunk_index: int
    doc_id: str
    chunk_id: str
    method: str
    rank: Optional[int]
    score: float
    weights: Dict[str, float]
    lists: List[ListAttribution] = field(default_factory=list)
    rrf_k: int = 60
    #: Resolved per-list RRF weights (see
    #: :func:`ragkb.retriever.resolve_fusion_weights`). Defaults to empty,
    #: which reads as weight 1.0 everywhere - the unweighted fusion.
    fusion_weights: Dict[str, float] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        """Whether the explained query model borrowed terms it was not given."""
        return bool(set(self.weights) - set(tokenize(self.query)))

    def fusion_contribution(self, name: str) -> float:
        """What list ``name`` contributed to the fused score, or 0.0 if unranked."""
        for entry in self.lists:
            if entry.name == name:
                if entry.rank is None:
                    return 0.0
                weight = self.fusion_weights.get(name, 1.0)
                return weight / (self.rrf_k + entry.rank)
        return 0.0

    def format(self, top_terms: int = 6, width: Optional[int] = None) -> str:
        """A human-readable trace, heaviest contributions first.

        ``width`` is off by default. Term names are already clipped to keep the
        columns aligned; clipping the *lines* would silently amputate the
        header that says which query and method produced them, and a diagnostic
        that hides its own provenance is worse than a wide one.
        """
        position = "unranked" if self.rank is None else "#{}".format(self.rank)
        lines = [
            "{}  [{}]  {}  score={:.5f}".format(
                position, self.doc_id, self.chunk_id, self.score
            ),
            "query: {!r}  method: {}".format(self.query, self.method),
        ]
        if self.expanded:
            borrowed = sorted(
                (
                    (term, weight)
                    for term, weight in self.weights.items()
                    if term not in set(tokenize(self.query))
                ),
                key=lambda pair: (-pair[1], pair[0]),
            )
            lines.append(
                "expanded with: {}".format(
                    ", ".join("{} ({:.3f})".format(t, w) for t, w in borrowed)
                )
            )
        if self.method == "hybrid":
            lines.append("")
            lines.append("fusion (rrf_k={}):".format(self.rrf_k))
            for entry in self.lists:
                share = self.fusion_contribution(entry.name)
                weight = self.fusion_weights.get(entry.name, 1.0)
                # Print the weight only when it is doing something, so the
                # unweighted trace stays exactly as readable as it was.
                numerator = "1" if weight == 1.0 else "{:g}".format(weight)
                lines.append(
                    "  {:<8} {:>10}  {}/({}+{}) = {:.6f}".format(
                        entry.name,
                        "unranked" if entry.rank is None else "rank {}".format(entry.rank),
                        numerator,
                        self.rrf_k,
                        "-" if entry.rank is None else entry.rank,
                        share,
                    )
                )
            lines.append(
                "  {:<8} {:>10}  {:>18.6f}".format(
                    "total",
                    "",
                    sum(self.fusion_contribution(e.name) for e in self.lists),
                )
            )
        for entry in self.lists:
            lines.append("")
            lines.append(
                "{} score {:.5f} (top {} of {} matched terms)".format(
                    entry.name, entry.score, min(top_terms, len(entry.terms)), len(entry.terms)
                )
            )
            lines.append(
                "  {:<18}{:>8}{:>8}{:>8}{:>10}{:>12}".format(
                    "term", "qw", "tf", "df", "idf", "contrib"
                )
            )
            for term in entry.terms[:top_terms]:
                lines.append(
                    "  {:<18}{:>8.3f}{:>8}{:>8}{:>10.3f}{:>12.5f}".format(
                        term.term[:18],
                        term.weight,
                        "-" if term.term_frequency is None else term.term_frequency,
                        term.document_frequency,
                        term.idf,
                        term.contribution,
                    )
                )
            if entry.missing:
                lines.append(
                    "  missing from this chunk: {}".format(", ".join(entry.missing))
                )
            if entry.unknown:
                lines.append(
                    "  not in the corpus at all: {}".format(", ".join(entry.unknown))
                )
        return "\n".join(line[:width] if width else line for line in lines)


def _features(index, query_tokens: Sequence[str]) -> List[str]:
    """The features ``index`` actually scores for this query.

    Words for BM25 and TF-IDF, character n-grams for the n-gram list. The
    representation is the only thing that differs between them, which is
    exactly the point :mod:`ragkb.ngram` makes, so it is the only thing this
    helper has to know about.
    """
    if isinstance(index, CharNgramIndex):
        return char_ngrams(list(query_tokens), index.n)
    return list(query_tokens)


def attribute_bm25(
    index: BM25Index, weights: Mapping[str, float], doc_index: int
) -> ListAttribution:
    """Decompose one BM25 document score into its per-term summands.

    The returned contributions sum to
    :meth:`~ragkb.bm25.BM25Index.score_document_weighted` exactly, because they
    are that method's loop body with the accumulation unrolled.
    """
    frequencies = index.term_frequencies[doc_index]
    length = index.doc_lengths[doc_index]
    norm = (
        index.k1 * (1.0 - index.b + index.b * length / index.avg_doc_length)
        if index.avg_doc_length
        else index.k1
    )
    terms: List[TermScore] = []
    missing: List[str] = []
    unknown: List[str] = []
    for term, weight in weights.items():
        df = index.document_frequency.get(term, 0)
        if not df:
            unknown.append(term)
            continue
        tf = frequencies.get(term, 0)
        if not tf or not weight:
            missing.append(term)
            continue
        idf = index.idf.get(term, 0.0)
        terms.append(
            TermScore(
                term=term,
                weight=float(weight),
                document_frequency=df,
                term_frequency=tf,
                idf=idf,
                contribution=weight * idf * (tf * (index.k1 + 1.0)) / (tf + norm),
            )
        )
    terms.sort(key=lambda item: (-item.contribution, item.term))
    return ListAttribution(
        name="bm25",
        score=index.score_document_weighted(weights, doc_index),
        rank=None,
        terms=terms,
        missing=sorted(missing),
        unknown=sorted(unknown),
    )


def attribute_vector(
    index: TfidfIndex,
    name: str,
    query_tokens: Sequence[str],
    doc_index: int,
    weights: Optional[Mapping[str, float]] = None,
) -> ListAttribution:
    """Decompose one cosine similarity into its per-feature products.

    Cosine on L2-normalised sparse vectors is a dot product, so each shared
    feature contributes ``query_vector[f] * doc_vector[f]`` and the sum is the
    similarity itself. Features present in the query vector but not the
    document simply contribute nothing, which is what makes ``missing`` here
    the same diagnosis it is for BM25.
    """
    query_vector = (
        index.vectorize_query(list(query_tokens))
        if weights is None
        else index.vectorize_weights(weights)
    )
    document = index.vectors[doc_index]
    terms: List[TermScore] = []
    missing: List[str] = []
    for feature, weight in query_vector.items():
        doc_weight = document.get(feature, 0.0)
        if not doc_weight:
            missing.append(feature)
            continue
        terms.append(
            TermScore(
                term=feature,
                weight=weight,
                document_frequency=index.document_frequency.get(feature, 0),
                term_frequency=None,
                idf=index.idf.get(feature, 0.0),
                contribution=weight * doc_weight,
            )
        )
    unknown = sorted(
        {
            feature
            for feature in _features(index, query_tokens)
            if feature not in index.document_frequency
        }
    )
    terms.sort(key=lambda item: (-item.contribution, item.term))
    return ListAttribution(
        name=name,
        score=index.similarity(query_vector, doc_index),
        rank=None,
        terms=terms,
        missing=sorted(missing),
        unknown=unknown,
    )


def query_model(
    retriever: HybridRetriever, query: str, method: str = "hybrid"
) -> Tuple[Dict[str, float], bool]:
    """The query model :meth:`HybridRetriever.search` would really use.

    Returns ``(weights, expanded)``. The flag is not cosmetic: ``search`` takes
    a genuinely different code path when expansion is off, passing raw tokens
    rather than weights, and the TF-IDF side of that path applies sublinear
    damping to repeated query terms where the weighted path does not. Reporting
    weights for an unexpanded query would therefore describe a ranking the
    engine did not produce whenever a query repeats a word.

    The condition mirrors ``search`` exactly: an expansion that borrowed no new
    term is skipped there, so it is skipped here.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return {}, False
    if retriever.expansion is not None:
        expanded = retriever.expansion_model(query, method=method)
        if set(expanded) - set(query_tokens):
            return dict(expanded), True
    return {term: float(count) for term, count in Counter(query_tokens).items()}, False


def attribute(
    retriever: HybridRetriever,
    query: str,
    chunk_index: int,
    method: str = "hybrid",
    candidates: Optional[int] = None,
) -> Attribution:
    """Explain where ``chunk_index`` sits for ``query`` and what put it there.

    ``candidates`` bounds the ranking pass used to recover the ranks; it
    defaults to the whole collection so that a chunk which did *not* make the
    candidate pool still reports the rank it actually holds. That is the
    interesting case - "why did the right document lose" is answered by a
    number like rank 34, not by a blank.
    """
    if method not in retriever.methods:
        raise ValueError("method must be one of {}".format(retriever.methods))
    if not 0 <= chunk_index < len(retriever.chunks):
        raise IndexError("chunk_index out of range")

    query_tokens = tokenize(query)
    weights, expanded = query_model(retriever, query, method=method)
    pool_size = len(retriever.chunks) if candidates is None else candidates
    # ``_rank_pool`` is the same call ``search`` makes, down to which of its two
    # argument paths is taken, which is the point: re-deriving the ranks here
    # with a second implementation would explain a ranking the engine never
    # produced.
    if expanded:
        pool, ranks, _ = retriever._rank_pool(method, pool_size, weights=weights)
    else:
        pool, ranks, _ = retriever._rank_pool(
            method, pool_size, query_tokens=query_tokens
        )
    positions = {index: rank for rank, (index, _) in enumerate(pool, start=1)}
    scores = dict(pool)

    entries: List[ListAttribution] = []
    for name, index in retriever._lists():
        if isinstance(index, BM25Index):
            entry = attribute_bm25(index, weights, chunk_index)
        else:
            entry = attribute_vector(
                index,
                name,
                query_tokens,
                chunk_index,
                weights=weights if expanded else None,
            )
        entry.rank = ranks.get(name, {}).get(chunk_index)
        entries.append(entry)

    chunk = retriever.chunks[chunk_index]
    return Attribution(
        query=query,
        chunk_index=chunk_index,
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        method=method,
        rank=positions.get(chunk_index),
        score=scores.get(chunk_index, 0.0),
        weights=weights,
        lists=entries,
        rrf_k=retriever.rrf_k,
        fusion_weights=dict(retriever.fusion_weights),
    )


def best_chunk_for_doc(
    retriever: HybridRetriever,
    query: str,
    doc_id: str,
    method: str = "hybrid",
) -> int:
    """The index of ``doc_id``'s highest-ranked chunk under ``method``.

    Falls back to the document's first chunk when nothing in it scores at all,
    so ``explain --doc`` still has something to explain - a document that
    matched no query term is itself the finding.

    A trailing file extension is stripped if that is what makes the name
    resolve: corpus document ids come from the filename stem, and typing the
    filename you can see on disk should not be an error.
    """
    known = {chunk.doc_id for chunk in retriever.chunks}
    if doc_id not in known:
        stem = doc_id.rsplit(".", 1)[0]
        if stem in known:
            doc_id = stem
        else:
            raise KeyError("no chunk belongs to document {!r}".format(doc_id))
    weights, expanded = query_model(retriever, query, method=method)
    if expanded:
        pool, _, _ = retriever._rank_pool(method, len(retriever.chunks), weights=weights)
    else:
        pool, _, _ = retriever._rank_pool(
            method, len(retriever.chunks), query_tokens=tokenize(query)
        )
    for index, _ in pool:
        if retriever.chunks[index].doc_id == doc_id:
            return index
    for index, chunk in enumerate(retriever.chunks):
        if chunk.doc_id == doc_id:
            return index
    raise KeyError("no chunk belongs to document {!r}".format(doc_id))


__all__ = [
    "Attribution",
    "ListAttribution",
    "TermScore",
    "attribute",
    "attribute_bm25",
    "attribute_vector",
    "best_chunk_for_doc",
    "query_model",
    "NGRAM_METHOD",
]
