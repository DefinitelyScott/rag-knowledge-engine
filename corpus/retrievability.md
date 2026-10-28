# Retrievability: measuring how reachable each document is

Effectiveness metrics ask whether a system returns the right documents for
queries someone judged. They say nothing about documents nobody asked about.
Some documents can be reached by many different queries, while others almost
never appear near the top of any ranking, however relevant they might be to
somebody. Azzopardi and Vinay proposed retrievability in 2008 to measure this
side of a retrieval system, which is sometimes called its access bias.

The definition is simple. Take a large set of queries Q. For each document d,
count how often d is retrieved at or above a rank cutoff c, optionally
weighting each query by how likely it is to be issued. The retrievability r(d)
is that count. A document with r(d) of zero is effectively invisible to this
system under this query set, even though it is sitting in the index. The
cutoff matters, because it encodes how far a user is assumed to look: a small
c such as 10 models a user who reads one page, a large c models a patient one.

Because real query logs are not available for most collections, Azzopardi and
Vinay generated the query set from the collection itself, building short
queries from terms and term pairs that occur in the documents. That makes the
measure reproducible and independent of any one user population, at the price
that the simulated queries are not the queries people really write.

To summarise the whole distribution of r(d) in one number, they borrowed the
Gini coefficient from economics. Plot documents from least to most
retrievable and compare the cumulative share of retrievability against a
perfectly even spread; this is the Lorenz curve. A Gini coefficient of zero
means every document is equally reachable. A value near one means a small
group of documents soaks up nearly all the retrievals. Different retrieval
models produce different values on the same collection and the same queries,
so the Gini coefficient lets models be compared on bias as well as accuracy.

Much of the bias comes from document length. A model that rewards term
frequency without enough normalisation tends to favour long documents, which
match more query terms by chance. Over-normalising does the opposite and lets
short documents crowd out longer ones. The length parameter b in BM25, or the
equivalent knob in other models, therefore moves a system along an access-bias
curve. Azzopardi and Vinay reported that the setting that gives the best
accuracy is not necessarily the one that gives the fairest access, so the two
goals can pull in different directions.

The measure has been put to work. Researchers have studied which retrieval
models are most and least biased, how query expansion can raise the
retrievability of documents that are rarely found, and how patent search,
where a missed document can matter legally, should care about documents that
the system hides from every query. Wilkie and Azzopardi examined the link
between bias and effectiveness and found it to be real but not tight, which
suggests that a system can improve access without giving up much accuracy.

There are caveats. The numbers depend on the query generator and the cutoff,
so Gini values from one study cannot be compared with those from another. A
low-retrievability document is not automatically a problem, because some
documents are simply obscure and rightly rarely wanted. And retrievability
counts appearances, not usefulness: a system could spread its results evenly
over documents and still be unhelpful. It is a diagnostic to read alongside
precision and recall, not a replacement for them.

This document was written from recollection of the cited papers and has not
been checked against the originals.
