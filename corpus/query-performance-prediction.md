# Query performance prediction: knowing a query went badly without being told

Every evaluation in this project needs labels. Query performance prediction is
the attempt to estimate, for a single query and with no relevance judgements at
all, how well the retriever just did. It matters because a system that can tell
a good result list from a bad one can act on the difference: refuse to answer,
fall back to a slower and more accurate pipeline, ask the user to rephrase, or
decline to expand a query it cannot trust in the first place.

Predictors split by how much they are allowed to see. Pre-retrieval predictors
use only the query and the index statistics, so they cost almost nothing and can
run before any ranking happens. The average or maximum inverse document
frequency across query terms estimates specificity, on the reasoning that a
query made entirely of common words has nothing to discriminate with. The
standard deviation of those term weights matters as much as their mean, since a
query with one rare term and three ubiquitous ones behaves quite differently
from one whose terms are uniformly middling. Query scope, the fraction of the
collection containing any query term at all, captures the opposite failure: a
query that matches half the index has not narrowed anything.

Post-retrieval predictors are allowed to look at the result list, which makes
them stronger and more expensive. The clarity score of Cronen-Townsend, Zhou and
Croft measures the divergence between a language model built from the top
retrieved documents and the language model of the whole collection; a focused
result set talks about something in particular, and a failed one looks like
background noise. Weighted information gain compares the mean score of the top
few results against the score the collection as a whole would receive, so it
asks how far above ordinary the winners actually are. Normalized query
commitment, from Shtok and colleagues, instead reads the standard deviation of
the retrieved scores: a list whose scores are all alike is one where the
retriever never committed to anything, and a steep drop after the first result
is a sign it did.

Predictors are themselves evaluated by correlation. Run a query set, compute
each query's real average precision or nDCG from the judgements, compute the
prediction, and report the rank correlation between the two columns. The honest
summary of decades of that exercise is that correlations in the region of 0.4 to
0.5 are respectable, which is useful for routing decisions in aggregate and
close to worthless as a verdict on any individual query.

Two caveats keep this from being a free lunch. Score-based predictors are not
comparable across queries until they are normalised, because raw retrieval
scores grow with query length and term rarity rather than with quality, so an
unnormalised threshold silently encodes a preference for long queries. And
predictors transfer badly: one tuned on a collection of news articles will
mis-calibrate on short technical passages, which is the same distribution-shift
problem that afflicts the retrievers they are meant to be supervising.
