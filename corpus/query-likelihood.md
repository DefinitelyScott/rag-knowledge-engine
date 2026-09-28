# Query likelihood: ranking by how probably a document would have produced the query

BM25 and TF-IDF start from a weighting scheme and justify it afterwards. The
language-modelling approach, introduced by Ponte and Croft in 1998, starts from
a generative story instead. Treat each document as a sample from its own small
language model, a probability distribution over words, and rank documents by
the probability that their model would have generated the query. A document
scores highly if the query looks like something its author could plausibly
have written.

Estimated naively, the document model is just relative frequency: a word's
count in the document divided by the document's length. That estimate fails on
the first query that uses a word the document happens not to contain. The
query probability is a product over query words, so a single absent word
multiplies the whole score by zero, and a document matching four of five query
words ties with one matching none. Every working query-likelihood system
therefore smooths, moving some probability mass from the words a document does
contain onto the words it does not.

The two common schemes both borrow that mass from a model of the whole
collection. Jelinek-Mercer smoothing interpolates linearly, mixing a fixed
fraction of the collection model into every document regardless of its size.
Dirichlet smoothing adds the collection probabilities as if they were a fixed
number of pseudo-counts, usually called mu and often set somewhere in the low
thousands. Because the pseudo-counts are fixed while the real counts grow with
the text, a long document ends up trusting its own statistics and a short one
leans heavily on the collection. Zhai and Lafferty's 2001 study of smoothing
found that this choice matters about as much as anything else in the model,
and that its sensitivity differs between short keyword queries and long,
verbose ones.

The same study made a point that reconnects the two traditions. Once the
smoothed score is rearranged, a word that is rare in the collection contributes
more when it matches, which is an inverse-document-frequency effect nobody
wrote in explicitly, and the smoothing term also acts as a length
normalisation. The generative model rediscovers the two ingredients the
heuristic schemes were built around; it simply derives them from one
assumption rather than tuning them as separate knobs. In head-to-head
comparisons a well-smoothed query-likelihood model and a well-tuned BM25 land
close together, which is part of why both are still standard baselines.

The framework earned its place less through the base model than through what
it made easy to add. Pseudo-relevance feedback becomes estimating a better
query model: Lavrenko and Croft's relevance model of 2001 builds that model
from the top-ranked documents and re-scores against it, with ranking framed as
the divergence between the query model and each document model. Vocabulary
mismatch can be handled by a translation model, which lets a document generate
a query word it never contains by way of a related word it does. What the
framework never removed is its central independence assumption: the query is
still a bag of words drawn one at a time, so word order and proximity play no
part in the score.
