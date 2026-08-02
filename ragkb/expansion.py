"""Pseudo-relevance feedback query expansion (RM3-style).

The hard slice of the gold set was written to break lexical matching: the
question says "if someone writes automobiles but the page only ever says cars"
where the document says "vocabulary mismatch". Neither BM25 nor TF-IDF can
match a term the query never uses, so both fail the same way and fusion has
nothing to fuse. That is the vocabulary mismatch problem, and expansion is the
classical answer to it that does not require an embedding model.

The idea (Lavrenko & Croft, "Relevance-Based Language Models", SIGIR 2001; the
RM3 variant that interpolates back toward the original query) is to assume the
top few results of a first retrieval pass are relevant, read the terms they
actually use, and re-run the query with those terms added. No labels and no
second model: the corpus supplies the synonyms.

Two properties matter and both are deliberate here:

* **The original query stays dominant.** A pure feedback model drifts - one
  off-topic passage in the feedback set can pull the query somewhere else
  entirely. ``original_weight`` interpolates the feedback model back toward the
  query the user actually asked, which bounds the damage.
* **Candidate terms are scored by frequency *and* IDF.** Frequency alone
  promotes terms that are common everywhere, which adds noise rather than
  signal. Multiplying by IDF prefers terms that are frequent *in the feedback
  set specifically*, which is the thing worth borrowing.

The output is a term -> weight mapping, not a token list. Weighting a query by
repeating tokens happens to be exactly right for BM25 and only approximately
right for TF-IDF, so both indexes gained an explicit weighted-query entry point
instead (:meth:`ragkb.bm25.BM25Index.search_weighted` and
:meth:`ragkb.vector.TfidfIndex.search_weighted`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from .text import STOPWORDS

#: ``(document index, relevance score)`` pairs from a first retrieval pass.
Feedback = Sequence[Tuple[int, float]]


@dataclass(frozen=True)
class ExpansionConfig:
    """Knobs for pseudo-relevance feedback.

    Attributes:
        feedback_docs: How many top results of the first pass to treat as
            relevant. Small on purpose - precision at the very top is the only
            thing holding the assumption up.
        feedback_terms: How many new terms to add to the query.
        original_weight: Interpolation weight on the original query, in [0, 1].
            1.0 disables the feedback model entirely, which makes it the
            correctness check on this whole path.
    """

    feedback_docs: int = 5
    feedback_terms: int = 10
    original_weight: float = 0.6

    def __post_init__(self) -> None:
        if self.feedback_docs <= 0:
            raise ValueError("feedback_docs must be positive")
        if self.feedback_terms < 0:
            raise ValueError("feedback_terms must be non-negative")
        if not 0.0 <= self.original_weight <= 1.0:
            raise ValueError("original_weight must be in [0, 1]")


def _document_weights(feedback: Feedback) -> Dict[int, float]:
    """Normalise first-pass scores into weights over the feedback documents.

    Scores are only shifted when a scorer produces negative values; subtracting
    the minimum unconditionally would zero out the weakest feedback document
    every time, which silently shrinks the feedback set by one. RRF scores all
    sit near ``1 / rrf_k`` and therefore produce an almost uniform weighting;
    BM25 scores spread out and produce a sharply peaked one. Both are reasonable
    readings of "how much do I trust this passage", so neither is special-cased.
    """
    if not feedback:
        return {}
    scores = [score for _, score in feedback]
    floor = min(0.0, min(scores))
    shifted = [score - floor for score in scores]
    total = sum(shifted)
    if total <= 0.0:
        uniform = 1.0 / len(feedback)
        return {index: uniform for index, _ in feedback}
    return {
        index: value / total for (index, _), value in zip(feedback, shifted)
    }


def relevance_model(
    feedback: Feedback,
    documents: Sequence[Sequence[str]],
    idf: Mapping[str, float],
    exclude: Sequence[str] = (),
    top_terms: int = 10,
) -> Dict[str, float]:
    """Estimate P(term | relevant) from the assumed-relevant documents.

    Each feedback document contributes its within-document term probability,
    weighted by how strongly it was retrieved. The result is multiplied by IDF
    to prefer terms that distinguish the feedback set from the collection, then
    the best ``top_terms`` are kept and renormalised to sum to one.

    Stop words, pure numbers and terms already in the query are dropped: none of
    them can add anything the first pass did not already have.
    """
    if top_terms <= 0:
        return {}
    weights = _document_weights(feedback)
    banned = set(exclude) | set(STOPWORDS)
    scores: Dict[str, float] = {}
    for index, weight in weights.items():
        tokens = documents[index]
        if not tokens:
            continue
        length = float(len(tokens))
        for term, count in Counter(tokens).items():
            if term in banned or term.isdigit():
                continue
            scores[term] = scores.get(term, 0.0) + weight * (count / length)
    if not scores:
        return {}
    weighted = {
        term: probability * idf.get(term, 0.0)
        for term, probability in scores.items()
    }
    weighted = {term: value for term, value in weighted.items() if value > 0.0}
    if not weighted:
        return {}
    best = sorted(weighted.items(), key=lambda pair: (-pair[1], pair[0]))[:top_terms]
    total = sum(value for _, value in best)
    return {term: value / total for term, value in best}


def expand_query(
    query_tokens: Sequence[str],
    feedback: Feedback,
    documents: Sequence[Sequence[str]],
    idf: Mapping[str, float],
    config: ExpansionConfig,
) -> Dict[str, float]:
    """Build the interpolated query model ``alpha * P(t|q) + (1-alpha) * P(t|R)``.

    Args:
        query_tokens: The user's query, tokenised.
        feedback: First-pass ``(document index, score)`` pairs, best first.
        documents: Token lists for every indexed document.
        idf: Inverse document frequency per term.
        config: Feedback size and interpolation weight.

    Returns:
        Term -> weight, summing to 1.0. With ``original_weight == 1.0`` or no
        feedback terms this is just the query's own term distribution, which is
        rank-equivalent to the unexpanded query.
    """
    if not query_tokens:
        return {}
    counts = Counter(query_tokens)
    length = float(len(query_tokens))
    model = {term: count / length for term, count in counts.items()}

    alpha = config.original_weight
    if alpha >= 1.0 or config.feedback_terms == 0:
        return model

    expansion = relevance_model(
        feedback[: config.feedback_docs],
        documents,
        idf,
        exclude=list(counts),
        top_terms=config.feedback_terms,
    )
    if not expansion:
        return model

    blended = {term: alpha * weight for term, weight in model.items()}
    for term, weight in expansion.items():
        blended[term] = blended.get(term, 0.0) + (1.0 - alpha) * weight
    return blended
