# Passage aggregation: turning many small scores into one ranking

Splitting a document into passages solves an indexing problem and creates a
ranking problem. The retriever now scores fragments, but the thing a user
wants ranked is often the whole document, and there is no single obvious way to
collapse a handful of fragment scores into one number. The choice of rule is not
a detail; it changes which documents win, and it interacts with the splitting
decision that produced the fragments in the first place.

The simplest rule takes the first fragment and ignores the rest. That sounds
indefensible until you notice how much writing is front-loaded: a news lede, an
abstract, and an opening paragraph are all deliberate summaries, so the opening
fragment is a fair proxy for a whole article and costs one score to compute. It
fails exactly where documents are not written that way, which includes most
reference material and anything where the answer is buried in section four.

Taking the maximum instead is the rule that has held up best. Callan showed in
1994 that passage-level evidence beats whole-document scoring for long
documents, and Dai and Callan found in 2019 that when neural rankers were
applied to documents too long to score in one pass, the best single fragment was
the strongest simple aggregator available. The intuition is that relevance is
usually local: a document is worth reading because some part of it answers the
question, and averaging that part against ten paragraphs of unrelated material
buries the signal that mattered.

Summing every fragment does the opposite, and does it badly. A long document has
more fragments, so it accumulates more score simply by existing, which
reintroduces precisely the length bias that length normalisation inside the
ranking function was built to remove. Averaging avoids that but punishes the
document with one excellent paragraph surrounded by filler, which is often the
document the user needed. Pooling the best few fragments and summing only those
is the usual compromise, and it has a genuine advantage over the maximum: it
rewards corroboration, so a document that supports a claim in three places
outranks one that supports it once.

The rule and the splitting strategy cannot be chosen independently. Overlapping
fragments repeat their shared sentences, so a summing rule counts the same
evidence twice and quietly favours whatever configuration produced the most
overlap; a maximum is indifferent to that duplication. Smaller fragments raise
the maximum's variance, because a short window can match a query by accident in
a way a long one cannot.

For generation the whole framing may be wrong. A pipeline assembling context
does not need documents ranked at all, it needs a deduplicated set of passages
that fits a budget, so collapsing to document scores and then re-expanding
discards the very granularity that was useful. The deeper cost of the maximum
rule is that it is blind to corroboration by construction, which is the wrong
instinct for a question whose answer must be assembled from several places at
once.
