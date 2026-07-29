"""Answer generation over retrieved chunks.

Two answerers ship with the engine and both cite their sources:

``ExtractiveAnswerer``
    Fully offline. It selects the sentences from the retrieved chunks that best
    cover the query terms. It can only ever return text that exists in the
    corpus, so it cannot hallucinate - the failure mode is an unhelpful answer,
    not an invented one.

``OpenAIAnswerer``
    Optional. Calls the OpenAI chat completions endpoint over ``urllib`` (still
    no third-party dependency) with the retrieved chunks as numbered context and
    an instruction to answer only from that context.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .retriever import RetrievalResult
from .text import STOPWORDS, split_sentences, tokenize


@dataclass
class Answer:
    """An answer plus the document ids that support it."""

    text: str
    citations: List[str]
    method: str
    supporting_chunks: List[str] = field(default_factory=list)

    def formatted(self) -> str:
        if not self.citations:
            return self.text
        return "{}\n\nSources: {}".format(self.text, ", ".join(self.citations))


class ExtractiveAnswerer:
    """Pick the best-covering sentences out of the retrieved chunks."""

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def answer(self, query: str, results: Sequence[RetrievalResult]) -> Answer:
        if not results:
            return Answer(
                text="No relevant passage was found in the knowledge base.",
                citations=[],
                method="extractive",
            )

        query_terms = [t for t in tokenize(query) if t not in STOPWORDS]
        if not query_terms:
            query_terms = tokenize(query)
        term_weights = self._term_weights(query_terms, results)

        scored = []
        for position, result in enumerate(results):
            # Later chunks are slightly discounted so retrieval order still
            # matters when two sentences cover the query equally well.
            rank_weight = 1.0 / (1.0 + position)
            for sentence in split_sentences(result.text):
                tokens = set(tokenize(sentence))
                if not tokens:
                    continue
                coverage = sum(
                    weight for term, weight in term_weights.items() if term in tokens
                )
                if coverage <= 0.0:
                    continue
                # Normalise by sentence length so a long sentence does not win
                # simply by containing more words.
                density = coverage / math.sqrt(len(tokens))
                scored.append((density * rank_weight, position, sentence, result))

        if not scored:
            best = results[0]
            return Answer(
                text=best.text,
                citations=[best.doc_id],
                method="extractive",
                supporting_chunks=[best.chunk.chunk_id],
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected: List[str] = []
        citations: List[str] = []
        chunk_ids: List[str] = []
        seen = set()
        for _, _, sentence, result in scored:
            if sentence in seen:
                continue
            seen.add(sentence)
            selected.append(sentence)
            if result.doc_id not in citations:
                citations.append(result.doc_id)
            if result.chunk.chunk_id not in chunk_ids:
                chunk_ids.append(result.chunk.chunk_id)
            if len(selected) >= self.max_sentences:
                break

        return Answer(
            text=" ".join(selected),
            citations=citations,
            method="extractive",
            supporting_chunks=chunk_ids,
        )

    @staticmethod
    def _term_weights(
        query_terms: Sequence[str], results: Sequence[RetrievalResult]
    ) -> Dict[str, float]:
        """Weight query terms by how rare they are across the retrieved set."""
        document_frequency: Counter = Counter()
        for result in results:
            for term in set(tokenize(result.text)):
                document_frequency[term] += 1
        total = len(results)
        weights = {}
        for term in set(query_terms):
            df = document_frequency.get(term, 0)
            weights[term] = math.log((1.0 + total) / (1.0 + df)) + 1.0
        return weights


class OpenAIAnswerer:
    """Optional LLM answerer. Requires ``OPENAI_API_KEY`` in the environment."""

    ENDPOINT = "https://api.openai.com/v1/chat/completions"
    SYSTEM_PROMPT = (
        "You answer strictly from the numbered sources provided. "
        "Cite the sources you use as [1], [2] and so on. "
        "If the sources do not contain the answer, say so plainly."
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def build_prompt(self, query: str, results: Sequence[RetrievalResult]) -> str:
        blocks = []
        for number, result in enumerate(results, start=1):
            blocks.append("[{}] ({}) {}".format(number, result.doc_id, result.text))
        return "Sources:\n{}\n\nQuestion: {}".format("\n\n".join(blocks), query)

    def answer(self, query: str, results: Sequence[RetrievalResult]) -> Answer:
        if not self.available:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; use ExtractiveAnswerer for offline use."
            )
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.build_prompt(query, results)},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(self.api_key),
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        return Answer(
            text=text,
            citations=[result.doc_id for result in results],
            method="openai:{}".format(self.model),
            supporting_chunks=[result.chunk.chunk_id for result in results],
        )
