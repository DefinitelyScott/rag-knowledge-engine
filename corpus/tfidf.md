# TF-IDF and the vector space model

The vector space model represents each document as a point in a
high-dimensional space where every dimension corresponds to a term in the
vocabulary. The value along a dimension is the weight of that term in the
document, and similarity between a query and a document becomes a geometric
question rather than a set-matching one.

TF-IDF is the classic weighting scheme for those dimensions. Term frequency
measures how often the term appears in this document, and inverse document
frequency, introduced by Karen Sparck Jones in 1972, measures how rare the term
is across the whole collection. Multiplying the two gives high weight to terms
that are frequent here and uncommon elsewhere, which is a good proxy for "this
term is what the document is about".

Raw term frequency is usually dampened. Sublinear scaling replaces the count c
with 1 + log(c), so the tenth occurrence adds far less than the second. Without
that damping, a document that repeats a word obsessively can outrank a better
one that mentions it only a few times.

Vectors are then L2-normalised, meaning each is divided by its own length so
that it lies on the unit sphere. After normalisation the cosine similarity
between a query vector and a document vector is simply their dot product. The
normalisation is what makes documents of very different lengths comparable:
similarity depends on the direction of the vector, that is, the mix of terms,
not on its magnitude.

Because most documents contain only a small slice of the vocabulary, these
vectors are extremely sparse and are stored as dictionaries from term to
weight. Computing a dot product then means iterating over the shorter of the
two dictionaries and looking up matching terms in the other.

TF-IDF cosine similarity and BM25 often disagree about ranking even though both
are lexical. Cosine similarity compares whole-document composition, while BM25
sums per-term evidence with explicit length correction. Those different
inductive biases are precisely why fusing their result lists helps.
