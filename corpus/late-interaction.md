# Late interaction retrieval

Bi-encoders and cross-encoders sit at opposite ends of a trade-off. A
bi-encoder squeezes a whole passage into one vector, so every document can be
embedded ahead of time and search is a fast nearest-neighbour lookup, but the
single vector blurs detail: it cannot say which part of the passage matched the
query. A cross-encoder reads the query and the passage together, token
attending to token, and is far more accurate for it, but nothing can be
precomputed, so it is only affordable as a second stage over a small candidate
pool.

Late interaction keeps a middle seat. ColBERT, introduced by Khattab and
Zaharia at SIGIR 2020, stores one small vector per token of each document
rather than one vector per document. At query time each query token is also
embedded, and the score of a passage is the sum over query tokens of each
token's maximum similarity against any token of that passage. This scoring
rule is called MaxSim. The interaction between query and document happens late,
after both have been encoded independently, which is what preserves the ability
to precompute the document side.

The design buys back most of the accuracy that pooling into a single vector
gives up: a query term such as an error code or a rare name only needs to find
its one matching token in the document, instead of hoping the pooled vector
still carries a trace of it. The price is storage. Keeping a vector for every
token multiplies the index size by the passage length, which is why follow-up
work concentrates on compression: ColBERTv2 quantises the token vectors around
cluster centroids, and the PLAID engine prunes candidate passages using only
those centroids before any full token vectors are touched, both from Santhanam
and colleagues in 2022.

Late interaction also degrades differently from a bi-encoder. When a
single-vector model misses, the passage was simply never close in the embedding
space, and no amount of re-ranking of retrieved candidates can recover it. A
token-level scorer fails softer: partial matches still accumulate score from
the query tokens that did find counterparts.

Whether the extra machinery is worth it depends on the corpus. On short, clean
passages with distinctive vocabulary, lexical search plus a strong single-vector
model already covers most queries, and the token-level index mostly adds
operating cost. The honest way to decide is the same as for every other
component: run the candidate designs against a labelled gold set and compare.
