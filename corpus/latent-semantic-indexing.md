# Latent semantic indexing: matching on meaning through a low-rank matrix

Lexical retrieval fails when the searcher and the author use different words
for the same idea. A query about an "automobile" will not match a document that
only says "car", however relevant it is. Deerwester, Dumais, Furnas, Landauer
and Harshman proposed latent semantic indexing in 1990 as a way to bridge that
vocabulary gap without any hand-built thesaurus, by letting the statistics of
the collection reveal which terms behave alike.

The method starts from the term-document matrix, with one row per term and one
column per document. It applies a truncated singular value decomposition, which
factors the matrix into term vectors, a diagonal matrix of singular values and
document vectors, and then keeps only the largest k singular values. The
rank-k approximation is the best one in the least-squares sense. Each term and
each document becomes a short dense vector, typically a few hundred
dimensions rather than tens of thousands, and the dimensions are combinations of
terms rather than terms themselves.

The effect is that words which occur in similar contexts end up close together
even if they never appear in the same document. Two documents can be similar
because they share neighbours, not because they share words. A query is treated
as a tiny document: its term vector is projected into the same k-dimensional
space, a step called folding in, and documents are ranked by cosine similarity
to it. This is the same cosine ranking used on raw term vectors, applied after
the projection, so it can retrieve documents that contain none of the query
terms.

The costs are real. Computing the decomposition of a large sparse matrix is
expensive, and the dense factors must be stored. Adding documents later by
folding them in is cheap but drifts from what a full recomputation would
produce, so the index is rebuilt periodically. The number of dimensions k is a
tuning decision with no principled answer: too small blurs distinct topics
together, too large reproduces the original matrix and the vocabulary mismatch
with it. Because the factors contain negative values they have no direct
probabilistic reading, which motivated Hofmann's probabilistic latent semantic
analysis in 1999 and later topic models such as latent Dirichlet allocation.

Results were mixed. Reported gains over plain term matching were modest and
uneven, and the compression can hurt exact matches such as rare names and
identifiers, because blurring words together loses the sharp signal that makes
them useful. In practice the approach is better regarded as a complement to
lexical scoring than a replacement. Its lasting influence is conceptual: dense
vectors that place related text near each other, learned from co-occurrence,
are the ancestors of the neural embeddings used in modern dense retrieval.
