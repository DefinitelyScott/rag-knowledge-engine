# Learned sparse retrieval

BM25 assigns a weight to a term using nothing but counts: how often the term
appears in this document, how many documents contain it, and how long the
document is. The weights are honest but blind. They cannot tell that a passage
about "vector search" is a good answer for a query about "embedding lookup",
because the two share no surface token, and they give a term in a passage's
topic sentence exactly the weight they give the same term buried in an aside.

Learned sparse retrieval keeps the sparse vector and replaces the counting rule
with a trained one. The output is still a vector over the vocabulary with a
handful of non-zero entries, which is the property that matters: an inverted
index, postings lists, and every skipping optimisation built for them all
continue to work unchanged. Only the numbers written into the postings change.

There are two ways to make that vector better, and the strong systems use both.
The first is re-weighting. DeepCT, from Dai and Callan in 2019, runs a
transformer over the passage and predicts a term-frequency-like weight per
token from its context, so a term that carries the passage's meaning outscores
the same term used in passing. DeepImpact, from Mallia and colleagues in 2021,
pushes the idea further by learning the impact score that gets stored directly,
so retrieval is a sum of stored numbers with no scoring function on top.

The second is expansion, which attacks the vocabulary mismatch rather than the
weighting. Document expansion by query prediction, from Nogueira and colleagues
in 2019 and better known as doc2query, generates the questions a passage would
answer and appends them to the passage before indexing; the T5-based version,
docT5query, is the one usually cited. The expansion terms are ordinary tokens
in an ordinary index, so the cost lands entirely at indexing time.

SPLADE, from Formal, Piwowarski, and Clinchant in 2021, does both at once. It
reuses a masked-language-model head to project every input token onto the whole
wordpiece vocabulary, pools those projections, and so produces a weight for
terms that never appeared in the passage. Left alone this vector would be
nearly dense and useless for an inverted index, so training adds a FLOPS
regulariser that penalises the expected cost of a query against the resulting
postings, buying sparsity explicitly rather than hoping for it.

The trade-off is latency, and it is easy to miss. A learned sparse index has
the same data structure as a lexical one but not the same shape: expansion
lengthens postings lists and flattens the Zipfian skew that dynamic-pruning
methods like WAND rely on to skip work. An index that is nominally sparse can
still be several times slower to query than BM25 over the same collection,
which is why the fair comparison reports latency next to accuracy rather than
treating "sparse" as a synonym for "cheap".
