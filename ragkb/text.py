"""Tokenisation, sentence splitting and sentence-aware overlapping chunking.

Chunking is the first place a RAG pipeline can silently lose recall: split in
the middle of an explanation and neither half contains the full answer. The
chunker here therefore never breaks a sentence, and it repeats trailing
sentences at the head of the next chunk so that ideas spanning a boundary
remain retrievable from both sides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
#: A markdown ATX heading line. Headings carry no terminal punctuation, so the
#: sentence-boundary regex above cannot see the end of one: without this, a
#: heading is glued to the first sentence of the section it introduces, and the
#: resulting pseudo-sentence exists nowhere in the source. That leaks into
#: extractive answers and makes them impossible to trace back to a passage.
_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+[^\n]*$")

#: A small, explicit English stop list. Kept short on purpose: BM25 already
#: discounts frequent terms through IDF, so aggressive removal mostly costs
#: recall on phrase-like queries.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can did do does for from had has have how
    i if in into is it its of on or that the their then there these this to was
    were what when where which who why will with you your
    """.split()
)


def normalize(text: str) -> str:
    """Fold typographic punctuation and collapse whitespace runs."""
    replacements = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - "}
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Lowercase, split on non-alphanumerics, optionally drop stop words."""
    tokens = _TOKEN_RE.findall(normalize(text).lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def _isolate_headings(text: str) -> str:
    """Surround markdown heading lines with blank lines.

    Makes every heading a paragraph of its own so the paragraph split below
    treats it as a hard boundary.
    """
    return _HEADING_RE.sub(lambda match: "\n\n{}\n\n".format(match.group(0)), text)


def is_heading(text: str) -> bool:
    """True for a markdown heading line such as ``## Chunking``.

    A heading names a section; it does not assert anything. Retrieval still
    indexes it - the words are a useful signal about what the section covers -
    but an extractive answer built out of headings answers nothing.
    """
    return bool(_HEADING_RE.fullmatch(text.strip()))


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Blank lines and markdown headings are hard boundaries; within a paragraph,
    sentences end at ``.``, ``!`` or ``?`` followed by a capitalised token.
    """
    sentences: List[str] = []
    for paragraph in _PARAGRAPH_RE.split(_isolate_headings(normalize(text))):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph):
            sentence = " ".join(sentence.split())
            if sentence:
                sentences.append(sentence)
    return sentences


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text plus the provenance needed to cite it."""

    doc_id: str
    chunk_id: str
    text: str
    token_count: int
    sentence_start: int
    sentence_end: int
    metadata: Dict[str, str] = None  # type: ignore[assignment]

    @property
    def tokens(self) -> List[str]:
        return tokenize(self.text)


def _token_counts(sentences: Sequence[str]) -> List[int]:
    return [len(tokenize(sentence)) for sentence in sentences]


def chunk_document(
    doc_id: str,
    text: str,
    target_tokens: int = 120,
    overlap_tokens: int = 30,
) -> List[Chunk]:
    """Split ``text`` into overlapping, sentence-aligned chunks.

    A chunk grows sentence by sentence until adding the next sentence would push
    it past ``target_tokens``. The next chunk is then seeded with the trailing
    sentences of the one just emitted, up to ``overlap_tokens`` worth of text.
    A sentence longer than ``target_tokens`` becomes a chunk of its own rather
    than being cut in half.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be in [0, target_tokens)")

    sentences = split_sentences(text)
    if not sentences:
        return []

    counts = _token_counts(sentences)
    chunks: List[Chunk] = []
    start = 0
    index = 0
    current: List[int] = []
    current_tokens = 0

    def emit(first: int, last: int) -> None:
        # Newlines, not spaces: joining with a space would run a heading into
        # the sentence after it, and re-splitting the chunk would then produce
        # a sentence that appears nowhere in the source. Keeping the break
        # makes split_sentences(chunk.text) recover exactly these sentences,
        # which is what lets an answer be traced back to a passage.
        body = "\n".join(sentences[first : last + 1])
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id="{}::{}".format(doc_id, len(chunks)),
                text=body,
                token_count=sum(counts[first : last + 1]),
                sentence_start=first,
                sentence_end=last,
                metadata={},
            )
        )

    while index < len(sentences):
        cost = counts[index]
        if current and current_tokens + cost > target_tokens:
            emit(start, index - 1)
            # Seed the next chunk with the tail of the one just emitted.
            back = index - 1
            carried = 0
            while back > start and carried + counts[back] <= overlap_tokens:
                carried += counts[back]
                back -= 1
            start = min(back + 1, index - 1) if overlap_tokens else index
            if overlap_tokens == 0:
                start = index
            current = list(range(start, index))
            current_tokens = sum(counts[start:index])
        current.append(index)
        current_tokens += cost
        index += 1

    if current:
        emit(start, len(sentences) - 1)
    return chunks
