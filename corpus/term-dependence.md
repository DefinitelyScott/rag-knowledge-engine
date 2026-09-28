# Term dependence: when word order and proximity should count

Nearly every lexical scorer in common use treats a query as an unordered bag of
words. BM25, TF-IDF cosine and query likelihood all score each query term
independently and add the contributions up, so a document that mentions
"retrieval" in its first paragraph and "evaluation" in its last scores the same
as one containing the phrase "retrieval evaluation", provided the counts match.
For many queries that loses real evidence. Words the user typed next to each
other usually belong together, and a document where they appear close together
is more likely to be about the combined idea than one where they merely
co-occur somewhere.

The best-known way to put that evidence back is the Markov random field model
of Metzler and Croft, published in 2005. It scores a document with three kinds
of feature rather than one: the individual query terms, as usual; each pair of
adjacent query terms appearing as an exact ordered phrase; and each adjacent
pair appearing in either order within a small unordered window, typically eight
words. The variant that links only neighbouring query terms is the sequential
dependence model, and it is the one that became a baseline. Its three feature
groups are mixed with fixed weights, commonly quoted as 0.85 for the single
terms, 0.10 for the ordered phrases and 0.05 for the windows. The proximity
features are deliberately a light correction on top of a bag-of-words core,
not a replacement for it. A full dependence variant that links every pair of
query terms exists too, and it rarely pays for the extra features on short
queries.

Proximity has a cost that bag-of-words scoring does not. Deciding whether two
terms fall within a window needs their positions, so the index must store
positional postings, which are larger than frequency-only postings, and the
scorer has to intersect position lists at query time for every pair it
considers. Büttcher, Clarke and Lushman showed in 2006 that a simpler proximity
bonus could be added to BM25 directly, rewarding each pair of query terms by
how near their occurrences sit, but the positional data is still required.
Because the gain is concentrated at the top of the ranking, a common
compromise is to run a cheap bag-of-words pass first and apply proximity
features only when re-ranking the few hundred candidates it returns.

The gains are real but uneven. Longer queries with a clear phrase inside them
benefit most; single-word queries cannot benefit at all, and queries whose
terms are genuinely independent can be hurt when the phrase features reward an
accidental adjacency. Learned models absorbed much of this signal later:
cross-encoders and late-interaction models see token order directly, which is
one reason explicit dependence features now appear mainly in first-stage
lexical systems. Chunking interacts with all of it. A window feature can only
fire within one passage, so a phrase split across a chunk boundary is invisible
to it, which is one more argument for letting chunks overlap.
