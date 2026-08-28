"""Deciding when *not* to answer.

Every stage before this one assumes the answer is in the corpus. Retrieval
always returns its top ``k``, and the extractive answerer always builds a
sentence out of whatever it is handed, so a question the knowledge base cannot
answer produces a confident-looking answer with real citations attached to
passages that do not contain the answer. Citations make that worse, not better:
they lend authority to a result nobody checked.

The gate here is a *pre-generation* check. It looks only at the query and the
retrieved evidence, so it runs identically in front of the offline extractive
answerer and in front of an LLM - and in front of the LLM it is the cheaper of
the two places to catch the problem, because it fires before any tokens are
paid for. This is the complement of :mod:`ragkb.faithfulness`, which checks a
generated answer *after* the fact: faithfulness asks "is this answer supported
by what was retrieved", abstention asks "was anything worth retrieving found at
all".

Three signals, all computable from indexes that are already fitted:

``vocab_coverage``
    Share of the query's IDF mass whose terms occur anywhere in the corpus. A
    question about a topic the corpus never mentions is made of words the
    corpus never indexed. Terms absent from the vocabulary have no IDF, so
    they are charged the maximum observed IDF - an unseen term is by
    definition maximally rare, and charging it less would let unknown words
    quietly cost nothing.

``match_coverage``
    Share of that same IDF mass actually present in the top-ranked chunk. This
    is the signal that separates a query whose *words* exist in the corpus from
    one whose words exist *together* in a passage. It catches the harder
    failure: a question assembled entirely from vocabulary the corpus knows,
    about a topic it does not cover.

``top_similarity``
    TF-IDF cosine between the query and the top-ranked chunk. Cosine is length
    normalised, so unlike a BM25 score it is comparable across queries and can
    be thresholded at all - which is the whole reason a raw retrieval score is
    not used here.

The three are blended into one score in [0, 1] and compared with a threshold.
Blending rather than requiring all three to pass is deliberate: the signals
disagree on exactly the interesting cases (a query with one unknown technical
term but strong overlap otherwise), and a hard AND would abstain on every
question containing a word the corpus happens not to use.

No threshold is universally right, and the default here is not a claim that it
is. ``evals/abstention_eval.py`` sweeps it against the answerable gold set and
a set of deliberately unanswerable questions and prints the trade-off, which is
the only honest way to pick one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .retriever import RetrievalResult
from .text import STOPWORDS, tokenize


@dataclass
class AbstentionConfig:
    """How the three signals are blended, and where the line is drawn."""

    #: Score at or above which the question is treated as answerable. 0.28 is
    #: the balanced-accuracy optimum ``evals/abstention_eval.py`` reports for
    #: the shipped corpus - not pooled accuracy, which is maximised by a gate
    #: that never fires because answerable questions outnumber unanswerable
    #: ones three to one. Re-run the sweep after changing the corpus; the
    #: right threshold also depends on which error costs more, and a system
    #: where a wrong answer is expensive should sit above this.
    threshold: float = 0.28
    vocab_weight: float = 0.25
    match_weight: float = 0.45
    similarity_weight: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        for name in ("vocab_weight", "match_weight", "similarity_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError("{} must be non-negative".format(name))
        total = self.vocab_weight + self.match_weight + self.similarity_weight
        if total <= 0.0:
            raise ValueError("signal weights must sum to a positive number")
        # Normalised on construction so the score stays in [0, 1] whatever
        # weights a caller passes, and so a threshold keeps its meaning when
        # the weights are re-tuned.
        self.vocab_weight /= total
        self.match_weight /= total
        self.similarity_weight /= total


@dataclass
class AnswerabilityReport:
    """The abstention decision plus every number that produced it."""

    score: float
    answerable: bool
    threshold: float
    vocab_coverage: float
    match_coverage: float
    top_similarity: float
    unknown_terms: List[str]

    def explain(self) -> str:
        verdict = "answerable" if self.answerable else "abstain"
        line = (
            "{} (score {:.3f} vs threshold {:.3f}; vocab {:.3f}, "
            "match {:.3f}, cosine {:.3f})".format(
                verdict,
                self.score,
                self.threshold,
                self.vocab_coverage,
                self.match_coverage,
                self.top_similarity,
            )
        )
        if self.unknown_terms:
            line += "\nterms absent from the corpus: {}".format(
                ", ".join(self.unknown_terms)
            )
        return line


#: What the answerer says when the gate fires. Deliberately names the corpus as
#: the limit rather than the question as bad: the question may be perfectly
#: good and simply outside what was indexed.
REFUSAL = (
    "The knowledge base does not appear to cover this question, so no answer "
    "is given rather than one built from passages that do not address it."
)


def query_terms(query: str) -> List[str]:
    """Content terms of a query, falling back to all tokens if it is all stopwords."""
    tokens = tokenize(query)
    content = [token for token in tokens if token not in STOPWORDS]
    return content or tokens


def assess(
    retriever,
    query: str,
    results: Sequence[RetrievalResult],
    config: Optional[AbstentionConfig] = None,
) -> AnswerabilityReport:
    """Score how well ``results`` actually cover ``query``.

    ``retriever`` supplies the fitted BM25 and TF-IDF indexes; nothing is
    re-fitted and no second retrieval pass is run, so the check costs one
    cosine and a set intersection on top of a search that already happened.
    """
    config = config or AbstentionConfig()
    terms = query_terms(query)
    if not terms or not results:
        return AnswerabilityReport(
            score=0.0,
            answerable=False,
            threshold=config.threshold,
            vocab_coverage=0.0,
            match_coverage=0.0,
            top_similarity=0.0,
            unknown_terms=list(dict.fromkeys(terms)),
        )

    idf = retriever.bm25.idf
    # An unseen term is rarer than anything the corpus contains, so it is
    # charged the largest IDF actually observed rather than zero.
    unseen_weight = max(idf.values()) if idf else 1.0

    weights: Dict[str, float] = {
        term: idf.get(term, unseen_weight) for term in terms
    }
    total_mass = sum(weights.values()) or 1.0

    known_mass = sum(weight for term, weight in weights.items() if term in idf)
    vocab_coverage = known_mass / total_mass

    top = results[0]
    top_terms = set(top.chunk.tokens)
    matched_mass = sum(weight for term, weight in weights.items() if term in top_terms)
    match_coverage = matched_mass / total_mass

    index = top.chunk_index
    if index is None or index < 0:
        top_similarity = 0.0
    else:
        query_vector = retriever.vector.vectorize_query(terms)
        top_similarity = max(0.0, retriever.vector.similarity(query_vector, index))

    score = (
        config.vocab_weight * vocab_coverage
        + config.match_weight * match_coverage
        + config.similarity_weight * top_similarity
    )
    unknown = [term for term in dict.fromkeys(terms) if term not in idf]
    return AnswerabilityReport(
        score=score,
        answerable=score >= config.threshold,
        threshold=config.threshold,
        vocab_coverage=vocab_coverage,
        match_coverage=match_coverage,
        top_similarity=top_similarity,
        unknown_terms=unknown,
    )
