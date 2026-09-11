# Relevance feedback and relevance models

Relevance feedback reformulates a query using evidence about which documents
were relevant to it. The evidence comes from one of three places. Explicit
feedback asks the user to mark results and is accurate and rare. Implicit
feedback infers the same thing from clicks and dwell time, which is plentiful
and biased by position. Pseudo-relevance feedback asks nobody and simply
assumes the top few results were right, which is why it is the only form that
runs unattended.

Rocchio's formulation, from the SMART system in 1971, moves the query vector
toward the centroid of the documents believed relevant and away from the
centroid of those believed not. Three weights control it: one on the original
query, one on the relevant centroid, one on the non-relevant centroid. The
third is usually small or dropped entirely, because non-relevance is diffuse.
Relevant documents cluster in one region for one reason, while irrelevant ones
are scattered in every direction, so their average points nowhere useful.

Term selection matters more than the geometry. Most words in the feedback
documents are useless as expansion terms, so candidates are ranked by a weight
that prefers terms common inside the feedback set and rare across the
collection, and only the top ten to thirty are kept. Adding hundreds slows
every posting-list traversal and dilutes the words the user actually chose.

The language-modelling version replaces the centroid with a distribution.
Relevance models, introduced by Lavrenko and Croft in 2001, estimate a term
distribution from the feedback documents, weighting each document by its
retrieval score rather than treating the top k as equally trustworthy. RM3
interpolates that estimate with the original query model, and the weight on the
original plays exactly the role Rocchio's first coefficient did: it is the
anchor that stops the reformulation from wandering off the question.

Query drift is the failure mode. When the first pass is poor the feedback terms
are drawn from off-topic documents and the second pass is worse than the first.
The damage is asymmetric, so a mean improvement across a query set can conceal
a minority of queries that degrade badly, and per-query variance is the number
worth reporting. That asymmetry is the argument for applying feedback
selectively, gated on some estimate of whether the first pass went well, rather
than always.

Measurement carries its own trap. Scoring the second pass over the same
documents that supplied the new terms credits the method for retrieving what it
was just shown. Residual ranking is the standard correction: remove the
feedback documents from the pool and evaluate only on what is left.
