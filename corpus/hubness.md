# Hubness in high-dimensional search

Hubness is a property of nearest-neighbour search in high-dimensional spaces:
a small number of points turn up in the nearest-neighbour lists of a very large
number of other points, while many points turn up in almost none. The effect
was described for music similarity by Aucouturier and Pachet, who called the
frequent offenders hubs, and was analysed in general terms by Milos Radovanovic,
Alexandros Nanopoulos and Mirjana Ivanovic around 2010. It matters for dense
retrieval because a retrieval index is exactly a nearest-neighbour structure.

The standard way to measure it is the k-occurrence. For each indexed vector,
count how many times it appears among the k nearest neighbours of a set of
query points. In a low-dimensional space this count is roughly symmetric
around k. As dimensionality grows, the distribution of the k-occurrence becomes
strongly right-skewed: a few hubs have counts many times k, and a growing share
of points are anti-hubs with a count of zero. The skewness of that distribution
is the usual single-number summary of how bad the problem is.

The cause is concentration of distances. When dimensionality is high, the
distances from a query to all the points in a collection become similar to
one another, and the points that happen to lie close to the centre of the data
are, on average, a little closer to everything. Hubs are therefore not
better documents. They are documents that are geometrically central, and they
win neighbour lists for that reason rather than because they answer anything.
The effect is stronger in the presence of a few dominant directions in the
embedding space, which is common for vectors produced by a trained encoder.

For a search engine the consequences are concrete. A hub is retrieved for many
unrelated queries, which looks like noise in the results. An anti-hub is
effectively unreachable by any query, even one written to match it, so it is a
document the index holds but the user can never find. This is a vector-space
version of the retrievability problem, and the two are measured in a similar
way: count how often each document is retrieved over a large sample of queries.

Several corrections exist. Centring the vectors by subtracting the collection
mean, and then re-normalising, removes the shared direction that makes some
points central. Whitening goes further and decorrelates the dimensions. Local
scaling and the closely related cross-domain similarity local scaling rescale a
similarity by the average similarity of each point to its own neighbourhood,
so that a hub, whose neighbourhood is dense and highly similar, has its
scores discounted. Mutual proximity converts a distance into the probability
that each point is among the other's neighbours. Lowering the dimensionality
with a method such as principal component analysis reduces concentration, but
at the cost of the information the dropped dimensions carried.

Hubness is also an argument for being careful about what a recall figure for an
approximate index means. An approximate nearest-neighbour method may return
exactly the true neighbours and still return hubs for most queries. Recall
against the exact neighbour list measures fidelity to the geometry, not
quality for the user. Lexical retrieval is less exposed because a sparse
vector has very high nominal dimensionality but most of its mass sits in a few
rare terms, which keeps distances from concentrating in the same way. That is
one reason hybrid systems that fuse a sparse list with a dense one often
outperform the dense list alone.
