"""Character n-gram retrieval: a third view of the same text.

BM25 and TF-IDF look like two retrievers, but they are two *scoring functions*
over one representation. Both call :func:`ragkb.text.tokenize`, both index whole
words, and both are therefore blind to exactly the same thing: a term the query
spells differently from the document. To either of them ``quantise``,
``quantisation`` and ``quantised`` are three unrelated symbols, ``re-ranking``
and ``reranking`` are unrelated, and one transposed letter is a total miss.
That shared blind spot is why fusing them buys less than it looks like it
should - and why the README could not demonstrate that ``rrf_k`` does anything:
two lists drawn from the same term space agree too often for the fusion
constant to have room to matter.

This index changes the representation rather than the score. Each token is
padded with ``_`` boundary markers and cut into overlapping character n-grams,
so ``quantise`` and ``quantisation`` share ``_qua``, ``quan``, ``uant``,
``anti``, ``ntis`` and ``tise`` and match each other partially without any
stemmer, lexicon or model. The weighting is then the same sparse TF-IDF cosine
used for words, which is the point: :class:`CharNgramIndex` subclasses
:class:`ragkb.vector.TfidfIndex` and replaces only the analyzer, so the
retrieval arithmetic being compared is held fixed and the representation is the
single thing that varies.

The costs are real and worth stating before any table does. A 120-token chunk
yields roughly 700 4-grams where it yielded 120 words, so the index is denser
and larger; and because most English 4-grams are common, *everything* matches
everything a little, which blunts precision. The expectation going in is a
retriever that is worse alone and useful in the fusion - the same bargain
TF-IDF already makes against BM25, but for a different reason.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, Sequence

from .vector import TfidfIndex

#: Default n. Three is too common to discriminate (``ing``, ``the`` and ``ion``
#: appear everywhere); five stops matching short words at all once the two
#: boundary markers are counted. Four keeps a useful overlap for the eight- to
#: twelve-character technical terms this corpus is full of.
DEFAULT_N = 4

#: Word boundary marker. The tokenizer's alphabet is ``[a-z0-9'-]``, so ``_``
#: can never appear inside a token and is unambiguous as a sentinel. Marking
#: both ends means a gram that starts a word is a different feature from the
#: same letters in the middle of one, which is what keeps ``_ran`` (as in
#: ``ranking``) distinct from ``eran`` (as in ``reranking``).
BOUNDARY = "_"


def char_ngrams(tokens: Sequence[str], n: int = DEFAULT_N) -> List[str]:
    """Cut each token into overlapping character n-grams, boundaries included.

    A token shorter than ``n`` once padded is emitted whole rather than
    dropped: losing every short word would silently change what the index can
    match, and the IDF weighting already handles the fact that short words are
    common.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    grams: List[str] = []
    for token in tokens:
        padded = BOUNDARY + token + BOUNDARY
        if len(padded) <= n:
            grams.append(padded)
            continue
        for start in range(len(padded) - n + 1):
            grams.append(padded[start : start + n])
    return grams


class CharNgramIndex(TfidfIndex):
    """TF-IDF cosine over character n-grams instead of words.

    Everything inherited from :class:`~ragkb.vector.TfidfIndex` - smoothed IDF,
    sublinear term frequency, L2 normalisation, cosine scoring, the ranking
    loop and the serialisation format - is unchanged. Only the analyzer moves,
    which is what makes a comparison between the two a comparison of
    representations rather than of implementations.
    """

    def __init__(self, n: int = DEFAULT_N, sublinear_tf: bool = True) -> None:
        super().__init__(sublinear_tf=sublinear_tf)
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n

    def fit(self, documents: Sequence[Sequence[str]]) -> "CharNgramIndex":
        """Fit over the n-grams of each document's word tokens."""
        super().fit([char_ngrams(tokens, self.n) for tokens in documents])
        return self

    def vectorize_query(self, query_tokens: Sequence[str]) -> Dict[str, float]:
        """Vectorise *word* tokens, converting them to n-grams first.

        Callers hand this index the same token list they hand the other two -
        the conversion belongs here rather than at every call site, which is
        what lets :meth:`~ragkb.vector.TfidfIndex.search` be inherited as-is.
        """
        return self._vectorize(Counter(char_ngrams(list(query_tokens), self.n)))

    def vectorize_weights(self, weights: Mapping[str, float]) -> Dict[str, float]:
        """Spread each word's weight across the n-grams that spell it.

        A weighted query model (what pseudo-relevance feedback produces) says
        "this *word* is worth 0.3". A gram appearing in several weighted words
        accumulates their weights, which is the behaviour the word index gives
        for a repeated term, and it keeps expansion meaningful over this
        representation instead of silently ignored.
        """
        gram_weights: Dict[str, float] = {}
        for term, weight in weights.items():
            if weight <= 0.0:
                continue
            for gram in char_ngrams([term], self.n):
                gram_weights[gram] = gram_weights.get(gram, 0.0) + weight
        return super().vectorize_weights(gram_weights)

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data["n"] = self.n
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "CharNgramIndex":
        index = super().from_dict(data)
        index.n = int(data.get("n", DEFAULT_N))
        return index
