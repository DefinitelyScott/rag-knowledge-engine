# Boolean retrieval: exact set logic, and why it never went away

The oldest retrieval model is also the simplest. A Boolean query combines terms
with AND, OR and NOT, and the system returns exactly the documents whose term
sets satisfy the expression. There is no score and no rank: a document either
matches or it does not, so the answer is an unordered set. The searcher gets
precise control, because every returned document provably contains the required
words, and also gets precisely the burden that control implies.

The mechanics sit directly on the inverted index. AND is an intersection of
two sorted posting lists, OR is a union and NOT is a difference, and each is a
single linear merge. Query processors reorder a conjunction so that the
shortest list is intersected first, since the result can never be larger than
the smallest operand, and skip pointers let the merge jump over stretches of
the longer list that cannot match. Nothing in this needs term weights, which is
why Boolean engines were practical on very early hardware.

The weaknesses are well documented. Results are all or nothing, and the size of
the set is hard to steer: a query with too many ANDs returns nothing, one with
too many ORs returns thousands of documents, a pattern often called feast or
famine. There is no way to say that a document matching four of five terms is
better than one matching one. Complex queries with nested parentheses are
error-prone, and most searchers write them badly. Treating every match as
equally good also throws away the evidence that term frequency and document
length supply.

Extended Boolean models try to keep the operators and add grading. Salton, Fox
and Wu proposed the p-norm model in 1983. Each term gets a weight in each
document, and a parameter p controls how strictly the operators behave: at
p equal to one AND and OR collapse into the same soft, vector-like average,
and as p grows large they approach strict Boolean logic. The result is a ranked
list that still respects the structure of the query, at the cost of a free
parameter and a slower evaluation than a plain merge.

Ranked retrieval displaced Boolean search for general use, and comparisons on
a large commercial legal collection reported that natural-language queries ranked
by a probabilistic model performed at least as well as the expert Boolean
queries written for the same information needs. Yet Boolean search is still
the working tool in several professional settings. Patent searchers, librarians
building systematic reviews and e-discovery teams value the properties a ranked
list cannot offer: a query is an auditable, reproducible statement of exactly
what was searched, a result set can be counted and reported, and an expert can
reason about which documents the query cannot have missed. When the goal is
near-complete recall rather than a good first page, that transparency matters
more than a clever ordering. Modern systems often keep both, using a Boolean
filter to define the candidate set and a ranking function to order it.
