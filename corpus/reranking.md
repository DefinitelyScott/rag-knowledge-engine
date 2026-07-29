# Re-ranking and result diversity

Retrieval is usually split into two stages. A cheap first-stage retriever scans
the whole corpus and returns a candidate pool of perhaps fifty to a hundred
passages. A more expensive second-stage re-ranker then scores only those
candidates and reorders them. The split exists because the accurate scorer is
too slow to run over everything.

The standard second-stage model is a cross-encoder. Unlike a bi-encoder, which
embeds the query and the passage separately, a cross-encoder concatenates them
and runs them through the model together, so every query token can attend to
every passage token. That interaction makes it markedly more accurate at
judging relevance, and also means its cost scales with the number of candidates,
since nothing can be precomputed.

Re-ranking cannot repair first-stage recall. If the correct passage is not in
the candidate pool, no reordering will surface it. The pool size is therefore a
real hyperparameter: too small and it caps quality, too large and latency
suffers.

Diversity is a separate concern. Top results are often near-duplicates,
especially with overlapping chunks, and returning five paraphrases of the same
sentence wastes a context window that could have held complementary evidence.
Maximal Marginal Relevance, proposed by Carbonell and Goldstein in 1998,
addresses this by selecting results greedily: at each step it picks the
candidate maximising a weighted trade-off between relevance to the query and
dissimilarity from what has already been selected. A parameter lambda controls
the balance, with lambda near one behaving like pure relevance ranking and
lower values pushing harder toward variety.

MMR is most valuable for questions that require several distinct facts, and
least valuable for narrow factoid questions where the single best passage is all
that is needed. Like every other retrieval choice, whether it helps is an
empirical question that a gold set can answer.
