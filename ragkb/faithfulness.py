"""Faithfulness and citation-accuracy measures for generated answers.

Retrieval metrics (hit@k, recall@k, MRR) stop at the passage. They say nothing
about the last step, which is where a RAG system is most often wrong in the way
users notice: the answer asserts something the retrieved passages do not
support, or it points at a source that did not contribute to it.

Four measures are defined here, all of them computable offline with no model
and no human labels:

``sentence_grounding``
    Fraction of answer sentences that appear verbatim in some retrieved
    passage. This is the strict test. An extractive answerer should score 1.0
    by construction, so anything less is a bug in sentence selection rather
    than a quality signal - which makes it a useful regression guard.

``token_grounding``
    Fraction of the answer's content tokens (stop words removed) that occur
    anywhere in the retrieved context. This is the lenient test, and the one
    that still means something for a paraphrasing LLM answerer, where verbatim
    overlap is not expected. It is the measure that catches an invented name,
    number, or acronym.

``citation_precision``
    Fraction of the cited documents that actually contributed a sentence to
    the answer. An answerer that cites everything it was handed scores high on
    recall and badly here, which is exactly the distinction worth making: a
    citation list that is not selective is not evidence.

``citation_recall``
    Fraction of grounded answer sentences whose source document is cited.
    Low values mean the answer used a passage without crediting it.

Comparison is done on a canonical token stream rather than raw characters, so
differences in whitespace, casing, and typographic punctuation do not count as
unfaithfulness. Matching is padded at both ends so ``"cat"`` never matches
inside ``"cats"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .text import STOPWORDS, split_sentences, tokenize

#: A retrieved passage reduced to what these measures need.
Passage = Tuple[str, str]  # (doc_id, text)


def _canonical(text: str) -> str:
    """Lowercase token stream, space-padded so matches land on token bounds."""
    tokens = tokenize(text)
    return " {} ".format(" ".join(tokens)) if tokens else " "


def _content_tokens(text: str) -> List[str]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    return tokens or tokenize(text)


def passages_from_results(results: Iterable) -> List[Passage]:
    """Adapt :class:`~ragkb.retriever.RetrievalResult` objects to passages."""
    return [(result.doc_id, result.text) for result in results]


def supporting_docs(sentence: str, passages: Sequence[Passage]) -> Set[str]:
    """Document ids whose passage text contains ``sentence`` verbatim.

    Empty when nothing supports the sentence, which is the ungrounded case.
    """
    needle = _canonical(sentence).strip()
    if not needle:
        return set()
    needle = " {} ".format(needle)
    return {
        doc_id for doc_id, text in passages if needle in _canonical(text)
    }


def sentence_grounding(answer_text: str, passages: Sequence[Passage]) -> float:
    """Fraction of answer sentences found verbatim in the retrieved passages."""
    sentences = split_sentences(answer_text)
    if not sentences:
        return 0.0
    grounded = sum(1 for s in sentences if supporting_docs(s, passages))
    return grounded / len(sentences)


def token_grounding(answer_text: str, passages: Sequence[Passage]) -> float:
    """Fraction of the answer's content tokens present in the context.

    Tokens are counted with multiplicity, so an answer that repeats an invented
    term is penalised each time it does so.
    """
    tokens = _content_tokens(answer_text)
    if not tokens:
        return 0.0
    context: Set[str] = set()
    for _, text in passages:
        context.update(tokenize(text))
    return sum(1 for token in tokens if token in context) / len(tokens)


def citation_precision(
    answer_text: str, citations: Sequence[str], passages: Sequence[Passage]
) -> float:
    """Fraction of cited documents that contributed a sentence to the answer.

    Returns 0.0 for an answer that cites nothing: with no citations there is
    nothing correct to credit, and treating the empty list as perfect would let
    an uncited answer top the table.
    """
    if not citations:
        return 0.0
    used: Set[str] = set()
    for sentence in split_sentences(answer_text):
        used |= supporting_docs(sentence, passages)
    unique = list(dict.fromkeys(citations))
    return sum(1 for doc_id in unique if doc_id in used) / len(unique)


def citation_recall(
    answer_text: str, citations: Sequence[str], passages: Sequence[Passage]
) -> float:
    """Fraction of grounded sentences whose source document is cited.

    Sentences with no support at all are excluded: they are an ungrounded-text
    problem, already counted by :func:`sentence_grounding`, and charging them
    here would double-count the same failure.
    """
    cited = set(citations)
    considered = 0
    credited = 0
    for sentence in split_sentences(answer_text):
        sources = supporting_docs(sentence, passages)
        if not sources:
            continue
        considered += 1
        if sources & cited:
            credited += 1
    return credited / considered if considered else 0.0


@dataclass
class FaithfulnessReport:
    """All four measures for a single answer."""

    sentence_grounding: float
    token_grounding: float
    citation_precision: float
    citation_recall: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "sentence_grounding": self.sentence_grounding,
            "token_grounding": self.token_grounding,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
        }


def evaluate_answer(answer, results: Sequence) -> FaithfulnessReport:
    """Score an :class:`~ragkb.answerer.Answer` against what was retrieved."""
    passages = passages_from_results(results)
    return FaithfulnessReport(
        sentence_grounding=sentence_grounding(answer.text, passages),
        token_grounding=token_grounding(answer.text, passages),
        citation_precision=citation_precision(
            answer.text, answer.citations, passages
        ),
        citation_recall=citation_recall(answer.text, answer.citations, passages),
    )


def mean_report(reports: Sequence[FaithfulnessReport]) -> Dict[str, float]:
    """Average each measure over a set of answers."""
    if not reports:
        return {
            "sentence_grounding": 0.0,
            "token_grounding": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
        }
    keys = reports[0].as_dict().keys()
    return {
        key: sum(report.as_dict()[key] for report in reports) / len(reports)
        for key in keys
    }
