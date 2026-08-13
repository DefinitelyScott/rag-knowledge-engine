# Vector compression and quantisation

Embeddings are usually produced as arrays of 32-bit floats, and at scale the
floats become the bill. A thousand-dimensional float vector costs four
kilobytes; a hundred million of them cost four hundred gigabytes before any
index structure is added. Vector compression attacks that bill directly, and
most of the techniques need no retraining of the embedding model at all.

The gentlest step is scalar quantisation: store each dimension as an 8-bit
integer instead of a 32-bit float, mapping the observed range of each dimension
onto 256 levels. That is an immediate four-fold saving, and because the error
added to each dimension is tiny relative to the spread of the data, retrieval
quality is usually within a hair of the float index.

Product quantisation, described by Jégou, Douze and Schmid in 2011, compresses
much harder. The vector is split into a handful of sub-vectors, a small
codebook of centroids is learned for each sub-space with k-means, and each
sub-vector is replaced by the identifier of its nearest centroid. A vector
becomes a short code of centroid ids, often eight to sixty-four bytes, and
distances are approximated by looking up precomputed centroid-to-query
distances per sub-space and summing them. Compression ratios of thirty times
and beyond are routine, which is why product quantisation underlies most
billion-scale vector indexes.

Binary quantisation is the extreme point: keep one bit per dimension, the sign,
and compare vectors with Hamming distance, which hardware counts very quickly.
Accuracy drops more visibly here, and the standard remedy applies to every
aggressive scheme: use the compressed index only to fetch a generous candidate
list, then re-score those candidates with the full-precision vectors, which can
live on disk since only a few dozen are touched per query. This rescoring pass
restores most of the lost ordering at almost no memory cost.

Compression can also happen before storage, inside the embedding itself.
Matryoshka representation learning, from Kusupati and colleagues in 2022,
trains the model so that meaning concentrates in the leading dimensions of the
vector. The same embedding can then be truncated to a half or a quarter of its
length and still rank sensibly, letting one model serve both a cheap coarse
search and a precise re-scoring pass.

Every scheme here trades recall for memory on a curve, and the curve depends on
the data. The only trustworthy way to pick an operating point is to measure
retrieval quality on a labelled gold set at each compression level, exactly as
one would tune any other retrieval knob.
