# Near-duplicate detection: keeping one copy of the same passage

Any collection assembled from more than one source accumulates copies. Mirrors,
syndicated articles, successive revisions of the same page, and boilerplate
headers repeated across a site all produce documents that are not byte-identical
but say the same thing. A passage store built for retrieval manufactures its own:
overlapping chunking deliberately emits windows that share a sentence with their
neighbours, so adjacent chunks of one document are near-duplicates by
construction.

Exact copies are a hash lookup away. Near-copies need a similarity measure, and
the obstacle is that comparing every pair is quadratic. The standard route makes
each document into a small fixed-size signature first, then compares signatures.
Broder's work in 1997 supplies the representation: treat a document as the set
of its overlapping word sequences, or shingles, and define resemblance as the
Jaccard similarity of two shingle sets. Shingle width is the first knob. Short
shingles match on common phrasing and flag unrelated documents; long ones are
specific and miss a pair that differs by a word inserted every sentence.

MinHash compresses that set. Hash every shingle, keep the minimum, and the
probability that two documents agree on that minimum is exactly their Jaccard
similarity, so a signature of a few hundred independent minima estimates the
similarity with error shrinking as the square root of the signature length.
Locality-sensitive hashing then avoids comparing even the signatures pairwise:
split each signature into bands of several rows, hash each band, and treat two
documents as candidates only if some band collides. The number of bands and
rows sets an S-shaped curve whose steep part sits near a chosen similarity
threshold, so the candidate step can be tuned to miss few true pairs while
rejecting most false ones cheaply.

SimHash takes the other trade. Charikar's 2002 construction projects a weighted
term vector onto random hyperplanes and records only the sign of each, producing
one compact fingerprint per document whose Hamming distance tracks cosine
similarity; Manku and colleagues reported it running at web scale in 2007. One
fingerprint is cheaper to store and compare than a minhash signature, at the
cost of a coarser estimate.

Deduplication and rank-time diversification address the same symptom at
different times and fail differently. Removing a duplicate at index time is
irreversible and depends on a threshold chosen once for the whole collection;
set it loosely and a genuinely distinct passage is deleted. Demoting redundancy
at rank time keeps everything and decides per query, at the cost of paying for
the copy in every index and every candidate list.

The evaluation consequence is easy to miss. If two near-identical documents
exist and only one is marked relevant, a retriever that returns the other is
scored as wrong for finding the right answer, and the measured difference
between two systems can come down to which copy each happened to prefer.
