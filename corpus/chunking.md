# Chunking strategies for retrieval

A retriever does not return documents so much as passages, and the way a
document is cut into passages determines what can be found at all. Chunk too
coarsely and a single chunk covers several topics, diluting its term statistics
and burying the relevant sentence among irrelevant ones. Chunk too finely and
the passage loses the context that made it meaningful, and pronouns lose their
antecedents.

Fixed-size chunking splits on a token or character count. It is trivial to
implement and gives uniform chunks, which keeps length normalisation
well-behaved, but it happily cuts sentences and even words in half.
Sentence-aware chunking instead accumulates whole sentences until a token
budget is reached, so no chunk ever ends mid-thought. Structure-aware chunking
goes further and respects markdown headings, list boundaries or code fences, on
the theory that the author's own segmentation is meaningful.

Overlap is the standard defence against boundary loss. Each chunk repeats the
last few sentences of the previous one, so an idea that spans a boundary is
retrievable from either side. Overlap of roughly ten to twenty-five per cent of
the chunk size is common. The cost is index size and duplicate results: two
adjacent chunks sharing the same overlapping sentence will often both match the
same query, which wastes slots in the final context window unless duplicates
are collapsed or diversity re-ranking is applied.

Chunk size interacts with everything downstream. Smaller chunks raise
precision, because the retrieved text is mostly on-topic, and they let more
independent passages fit into a fixed context budget. Larger chunks raise
recall per chunk and give a generator more surrounding context to work with.
There is no universally correct size; the honest way to choose is to hold the
retriever fixed, sweep the chunk size over a labelled question set, and read the
recall and MRR curves.

Whatever the strategy, every chunk should carry provenance back to its source
document, because that identifier is what a cited answer ultimately points at.
