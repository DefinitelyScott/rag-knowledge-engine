# Score normalisation and weighted score fusion

Reciprocal rank fusion merges result lists by discarding the scores and keeping
only the positions. The alternative is to keep the numbers, put them on a
common scale, and add them. That is a real choice rather than an oversight,
because a rank says only that one document beat another, while a score says by
how much.

Min-max normalisation is the usual first attempt: subtract the smallest score
in the candidate list and divide by the range, mapping every list into zero to
one. Its weakness is that both anchors are artefacts. The minimum is wherever
the list happened to be truncated, so retrieving fifty candidates instead of
twenty changes the normalised value of a document neither run would have
ranked differently, and a single outlier at the top compresses everything
beneath it.

Theoretical min-max replaces the observed extremes with bounds the scoring
function guarantees. Cosine similarity over length-normalised vectors cannot
leave minus one to one, so those constants can stand in for the sample extremes
and the transformation stops depending on retrieval depth at all. Lexical
scores have no such natural ceiling, which is why the two sides of a hybrid
system are rarely normalised by the same recipe.

The standard score, or z-score, subtracts the mean of the candidate scores and
divides by their standard deviation. It survives a single outlier better than
min-max, but it assumes a roughly symmetric spread, and retrieval score
distributions are strongly skewed: a handful of documents score far above a
long flat tail.

Fox and Shaw catalogued the combination rules in 1994 and the names stuck.
CombSUM adds the normalised scores. CombMNZ multiplies that sum by the number
of lists that retrieved the document at all, so agreement between retrievers is
rewarded directly rather than emerging from the arithmetic. CombANZ divides by
the same count and averages instead, which suits lists of very different
lengths.

The modern form of the argument is the convex combination: one weight alpha on
the lexical score and one minus alpha on the dense score. Analyses of fusion
functions for hybrid retrieval have found that a convex combination paired with
theoretical min-max normalisation can beat reciprocal rank fusion, and that the
tuned weight transfers across collections better than the rank-only view
predicts.

The trade is calibration against confidence. Rank fusion has nothing to fit and
never has to be re-checked when a retriever changes. Weighted score fusion can
say "this list is sure about its top hit", but only if someone maintains the
normalisation and the weight that let it speak.
