# Approximate nearest neighbour search

Once documents are embedded as vectors, retrieval becomes a nearest-neighbour
problem: given a query vector, find the vectors closest to it. Exact brute-force
search compares the query against every stored vector. That is simple, exact,
and linear in the number of vectors, which is perfectly acceptable up to
roughly hundreds of thousands of items and unacceptable at hundreds of
millions.

Approximate nearest neighbour indexes trade a small amount of recall for a very
large speedup. Inverted file indexes, IVF, cluster the vectors with k-means and
search only the few clusters closest to the query, controlled by a parameter
that sets how many clusters to probe. Product quantisation compresses each
vector into a short code by splitting it into sub-vectors and quantising each
one separately, which shrinks memory dramatically at some cost in accuracy.

Hierarchical Navigable Small World graphs, described by Malkov and Yashunin,
build a multi-layer proximity graph. Search starts at a sparse top layer, greedily
walks toward the query, and descends layer by layer, refining the candidate set.
HNSW gives excellent recall-latency trade-offs and is the default index in many
vector databases. Its costs are memory, since the graph edges must be held
alongside the vectors, and awkward support for deletion, which is often
implemented as tombstoning plus periodic rebuilds.

Every ANN index exposes a recall knob: how many clusters to probe, how large to
make the candidate list during graph search. Turning it up costs latency and
recovers accuracy. Because the index is approximate, a retrieval system built
on it has a recall ceiling that no amount of downstream re-ranking can lift,
which is why the setting should be measured against a gold set rather than
guessed.

For corpora of a few thousand passages, none of this is necessary. Exact search
over sparse vectors is faster than the engineering effort of maintaining an ANN
index, and it removes approximation as a possible source of evaluation error.
