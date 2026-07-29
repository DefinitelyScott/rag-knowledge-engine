# Reciprocal Rank Fusion and hybrid retrieval

Hybrid retrieval runs more than one retriever over the same corpus and merges
their outputs. The usual pairing is a lexical retriever such as BM25 with a
semantic or vector retriever, because their failure modes are close to
independent: lexical search misses paraphrases, vector search misses exact rare
identifiers such as error codes, product names or version numbers.

The hard part of merging is that the two systems produce scores on
incompatible scales. A BM25 score of 12.4 and a cosine similarity of 0.71 carry
no common unit, and the ranges shift from query to query, so a fixed weighted
sum of raw scores is fragile. Normalising scores per query, for example by
min-max scaling, helps but is sensitive to outliers and to how many candidates
each retriever returned.

Reciprocal Rank Fusion sidesteps the problem by throwing the scores away and
keeping only the ranks. Every result list contributes 1 / (k + rank) to each
document it ranks, and the contributions are summed across lists. The constant
k, conventionally 60, damps the influence of the very top ranks so that a
document ranked first by one retriever and absent from the other does not
automatically beat a document ranked third by both. RRF was introduced by
Cormack, Clarke and Buettcher at SIGIR 2009, where it outperformed Condorcet
fusion and trained rank-learning methods.

RRF has two practical virtues. It has essentially one hyperparameter, and it
requires nothing from the underlying retrievers beyond an ordered list, which
means a new retriever can be added to the ensemble without recalibrating
anything. Its main limitation is that it cannot express confidence: a retriever
that is certain about its top hit has no way to say so, because only position
survives the fusion.

Retrieval depth matters. Each retriever should return more candidates than the
number of results you finally want, typically twenty to a hundred, so that a
document ranked poorly by one retriever and well by the other still has a
chance to surface through the fusion.
