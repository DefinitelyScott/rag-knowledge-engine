"""Second-stage re-ranking with a cross-encoder-style scorer.

First-stage retrieval scores the query and the passage *separately*. BM25 sums
one contribution per query term; TF-IDF cosine compares two bags of words. Both
throw away the thing a reader uses immediately: where the query terms sit
relative to each other inside the passage. A chunk that says "the b parameter"
in one paragraph and "length normalisation" three paragraphs later scores the
same as one that says "the b parameter controls length normalisation" in a
single clause.

That is the gap a cross-encoder fills. A real one concatenates query and
passage and runs them through a transformer, so every query token can attend to
every passage token. It is far too slow to run over a whole collection, which
is why it is always a *second* stage: the first stage cuts the collection down
to a candidate pool, the second stage re-orders that pool with a more expensive
joint model.

This module keeps the two-stage architecture and swaps the model for something
that runs in the standard library. :class:`CrossEncoder` is the interface -
implement :meth:`CrossEncoder.score` and the retriever will use it.
:class:`LexicalCrossEncoder` is the built-in implementation, scoring the pair
on three joint features no bag-of-words scorer can compute:

* **coverage** - IDF-weighted fraction of the distinct query terms the passage
  contains at all. BM25 can rank a passage top by repeating one rare query term
  while never mentioning the rest of the question; coverage cannot.
* **proximity** - how tight the smallest window containing those terms is. This
  is the ordering signal that only exists once you look at positions.
* **phrase** - fraction of adjacent query bigrams that appear adjacent in the
  passage, which is the cheapest possible stand-in for "said it the same way".

The second-stage score is blended with the (min-max normalised) first-stage
score rather than replacing it: the first stage is a genuine relevance signal
and discarding it throws away the fusion work. ``alpha = 1.0`` keeps the
first-stage ordering exactly, which makes disabling the re-ranker provably
free - the same correctness check ``mmr_lambda = 1.0`` and
``original_weight = 1.0`` provide elsewhere in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .rerank import normalize_scores
from .text import STOPWORDS

Candidate = Tuple[int, float]


@dataclass(frozen=True)
class CrossEncoderWeights:
    """Relative weight of each joint feature. Normalised before use.

    The defaults lean on coverage because it is the feature that most often
    changes the top result on this corpus, and lean away from phrase because
    natural-language questions rarely reuse the document's exact word order.
    """

    coverage: float = 0.5
    proximity: float = 0.3
    phrase: float = 0.2

    def normalized(self) -> "CrossEncoderWeights":
        total = self.coverage + self.proximity + self.phrase
        if total <= 0.0:
            raise ValueError("cross-encoder weights must sum to a positive value")
        return CrossEncoderWeights(
            coverage=self.coverage / total,
            proximity=self.proximity / total,
            phrase=self.phrase / total,
        )


class CrossEncoder:
    """Interface for a scorer that sees the (query, passage) pair jointly.

    Subclasses implement :meth:`score`, returning a value in ``[0, 1]`` where
    higher is more relevant. Anything satisfying that contract can be dropped
    in - a trained cross-encoder, a hosted re-ranking API, a hand-written
    heuristic - without touching the retriever.
    """

    def score(
        self, query_tokens: Sequence[str], passage_tokens: Sequence[str]
    ) -> float:
        raise NotImplementedError

    def explain(
        self, query_tokens: Sequence[str], passage_tokens: Sequence[str]
    ) -> Dict[str, float]:
        """Per-feature breakdown of :meth:`score`, for debugging and tests.

        The default returns just the total. A re-ranker that cannot say why it
        moved a result is a re-ranker nobody can debug, so implementations are
        encouraged to override this.
        """
        return {"score": self.score(query_tokens, passage_tokens)}

    def rerank(
        self,
        query_tokens: Sequence[str],
        candidates: Sequence[Candidate],
        documents: Sequence[Sequence[str]],
        k: Optional[int] = None,
        alpha: float = 0.5,
    ) -> List[Candidate]:
        """Re-order ``candidates`` by the blended score and return the top ``k``.

        Args:
            query_tokens: the tokenised query.
            candidates: ``(item_id, first_stage_score)`` pairs, best first.
            documents: token lists indexed by item id.
            k: how many to keep; ``None`` keeps the whole pool so a later stage
                (MMR) still has candidates to work with.
            alpha: weight on the first-stage score, in ``[0, 1]``. ``1.0``
                returns the input ordering unchanged.

        Returns:
            ``(item_id, blended_score)`` pairs in the new order. The score is
            the blended one, not the first-stage one, so a downstream stage
            that normalises relevance sees the re-ranked judgement rather than
            silently undoing it.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        limit = len(candidates) if k is None else max(k, 0)
        if not candidates or limit == 0:
            return []
        if alpha >= 1.0:
            return list(candidates[:limit])

        scored: List[Tuple[float, int, int]] = []
        for position, (index, first_stage) in enumerate(normalize_scores(candidates)):
            cross = self.score(query_tokens, documents[index])
            blended = alpha * first_stage + (1.0 - alpha) * cross
            scored.append((blended, position, index))
        # Sort by blended score, breaking ties by the incoming order so the
        # re-ranker never reshuffles items it cannot distinguish.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(index, blended) for blended, _, index in scored[:limit]]


@dataclass
class LexicalCrossEncoder(CrossEncoder):
    """A positional, dependency-free stand-in for a trained cross-encoder.

    Args:
        idf: term -> IDF, normally :attr:`ragkb.bm25.BM25Index.idf`. Weights
            coverage so that matching a rare term counts for more than matching
            a common one. Terms absent from the map contribute nothing to
            either side of the coverage ratio: an out-of-vocabulary term cannot
            appear in any candidate, so charging every passage for missing it
            would only rescale the whole pool.
        weights: relative feature weights, normalised on construction.
        remove_stopwords: drop stop words from the query before scoring
            coverage and proximity. Phrase matching keeps them, because word
            order is exactly where they carry information.
    """

    idf: Mapping[str, float] = field(default_factory=dict)
    weights: CrossEncoderWeights = field(default_factory=CrossEncoderWeights)
    remove_stopwords: bool = True

    def __post_init__(self) -> None:
        self.weights = self.weights.normalized()

    def _content_terms(self, query_tokens: Sequence[str]) -> List[str]:
        """Distinct query terms to match on, in first-appearance order."""
        seen: Dict[str, None] = {}
        for token in query_tokens:
            if self.remove_stopwords and token in STOPWORDS:
                continue
            seen.setdefault(token, None)
        if not seen:  # an all-stopword query still deserves a ranking
            for token in query_tokens:
                seen.setdefault(token, None)
        return list(seen)

    def coverage(
        self, terms: Sequence[str], positions: Mapping[str, List[int]]
    ) -> float:
        """IDF-weighted fraction of the query terms the passage contains."""
        total = sum(self.idf.get(term, 0.0) for term in terms)
        if total <= 0.0:
            # No IDF information (unfitted map, or every term unseen): fall
            # back to an unweighted count so the feature still discriminates.
            if not terms:
                return 0.0
            return len(positions) / len(terms)
        found = sum(self.idf.get(term, 0.0) for term in terms if term in positions)
        return found / total

    def proximity(
        self, terms: Sequence[str], positions: Mapping[str, List[int]]
    ) -> float:
        """Tightness of the smallest window covering the matched terms.

        ``matched / span`` is 1.0 when the matched terms are adjacent and falls
        off as they spread out; scaling by ``matched / len(terms)`` stops a
        passage that happens to contain one query word from claiming perfect
        proximity.
        """
        if not terms or not positions:
            return 0.0
        span = _minimal_window(positions)
        if span is None or span <= 0:
            return 0.0
        matched = len(positions)
        return (matched / span) * (matched / len(terms))

    def phrase(
        self, query_tokens: Sequence[str], passage_tokens: Sequence[str]
    ) -> float:
        """Fraction of adjacent query bigrams that occur adjacent in the passage."""
        if len(query_tokens) < 2:
            return 0.0
        query_bigrams = {
            (query_tokens[i], query_tokens[i + 1]) for i in range(len(query_tokens) - 1)
        }
        passage_bigrams = {
            (passage_tokens[i], passage_tokens[i + 1])
            for i in range(len(passage_tokens) - 1)
        }
        if not query_bigrams:
            return 0.0
        return len(query_bigrams & passage_bigrams) / len(query_bigrams)

    def explain(
        self, query_tokens: Sequence[str], passage_tokens: Sequence[str]
    ) -> Dict[str, float]:
        terms = self._content_terms(query_tokens)
        positions = _term_positions(terms, passage_tokens)
        features = {
            "coverage": self.coverage(terms, positions),
            "proximity": self.proximity(terms, positions),
            "phrase": self.phrase(list(query_tokens), list(passage_tokens)),
        }
        features["score"] = (
            self.weights.coverage * features["coverage"]
            + self.weights.proximity * features["proximity"]
            + self.weights.phrase * features["phrase"]
        )
        return features

    def score(
        self, query_tokens: Sequence[str], passage_tokens: Sequence[str]
    ) -> float:
        return self.explain(query_tokens, passage_tokens)["score"]


@dataclass(frozen=True)
class RerankConfig:
    """Knobs for the second stage.

    Attributes:
        alpha: weight kept on the first-stage score, in ``[0, 1]``. 1.0
            disables the re-ranker, which makes it the correctness check on
            this whole path. The 0.5 default is the sweep result, not a guess:
            on the gold set it is the strongest setting that lifts hit@1
            (0.696 -> 0.750) and MRR (0.792 -> 0.826) while leaving hit@5
            untouched at 0.946. Leaning further on the second stage
            (alpha <= 0.2) buys a little more hit@1 and pays for it in deep
            coverage, which is the trade a re-ranker should be expected to
            make: it reorders the pool, it cannot enlarge it.
        weights: feature weights for the default encoder.
        encoder: an explicit :class:`CrossEncoder`. ``None`` builds a
            :class:`LexicalCrossEncoder` over the retriever's own IDF table -
            which is why the default is a config value and not an object: the
            IDF does not exist until the index is fitted.
    """

    alpha: float = 0.5
    weights: CrossEncoderWeights = field(default_factory=CrossEncoderWeights)
    encoder: Optional[CrossEncoder] = None

    def build(self, idf: Mapping[str, float]) -> CrossEncoder:
        if self.encoder is not None:
            return self.encoder
        return LexicalCrossEncoder(idf=idf, weights=self.weights)


def _term_positions(
    terms: Sequence[str], passage_tokens: Sequence[str]
) -> Dict[str, List[int]]:
    """Positions of each query term inside the passage, matched terms only."""
    wanted = set(terms)
    positions: Dict[str, List[int]] = {}
    for index, token in enumerate(passage_tokens):
        if token in wanted:
            positions.setdefault(token, []).append(index)
    return positions


def _minimal_window(positions: Mapping[str, List[int]]) -> Optional[int]:
    """Length of the shortest span containing every term in ``positions``.

    Classic two-pointer sweep over the merged occurrence list: advance the
    right edge until every term is covered, then pull the left edge in as far
    as it will go before the window breaks. O(n log n) for the sort, O(n)
    for the sweep.
    """
    if not positions:
        return None
    events = sorted(
        (index, term) for term, indexes in positions.items() for index in indexes
    )
    needed = len(positions)
    counts: Dict[str, int] = {}
    have = 0
    left = 0
    best: Optional[int] = None
    for right, (_, term) in enumerate(events):
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            have += 1
        while have == needed:
            span = events[right][0] - events[left][0] + 1
            best = span if best is None else min(best, span)
            leaving = events[left][1]
            counts[leaving] -= 1
            if counts[leaving] == 0:
                have -= 1
            left += 1
    return best
