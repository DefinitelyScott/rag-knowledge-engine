# Search result diversification: covering what an ambiguous query might mean

The probability ranking principle, stated by Robertson in 1977, says a system
does best by sorting documents in decreasing order of their probability of
relevance. The argument assumes that each document's relevance is judged on its
own, independently of the documents ranked above it. That assumption fails as
soon as two results say the same thing: the second copy is no more useful to a
reader who has already seen the first, and it occupies a slot that something
else could have filled. It fails harder when the query is ambiguous. A short
query such as "jaguar" may mean the animal, the car maker or an operating
system release, and a ranking built purely on estimated relevance can give the
whole first page to the most popular reading while the other readers find
nothing.

Diversification work usually separates two cases. An ambiguous query has
several unrelated interpretations, and the user wants exactly one of them. An
underspecified or faceted query has one interpretation with several aspects,
such as the history, prices and reviews of a single product, and the user may
want more than one. In both cases the ranking should hedge: spend the top few
slots so that whichever reading or aspect the user had in mind, something near
the top serves it.

Methods are grouped by whether they model the aspects at all. Implicit
methods never name them; they only penalise a candidate for resembling what is
already selected. Maximal Marginal Relevance, from Carbonell and Goldstein in
1998, is the standard example. Explicit methods first estimate the set of
aspects and then reward a candidate for covering aspects the ranking has not
yet covered. Agrawal, Gollapudi, Halverson and Ieong described IA-Select in
2009, classifying queries and documents into the categories of a taxonomy and
picking documents to maximise the probability that a user with any category in
mind finds at least one useful result. Santos, Macdonald and Ounis proposed
xQuAD in 2010, using the search engine's own query suggestions as stand-ins for
the aspects and running each one as a sub-query to judge which documents serve
which aspect. Dang and Croft's PM-2, from 2012, treats the result page as seats
to be shared out in proportion to each aspect's popularity, borrowing the
Sainte-Laguë method that some electoral systems use to allocate seats to
parties.

Selecting the best diverse set exactly is computationally hard: Agrawal and
colleagues showed their objective is NP-hard. It is, however, submodular, the
discrete analogue of diminishing returns, since adding a document covers less
new ground the more the set already covers. A greedy algorithm that repeatedly
adds the document with the largest marginal gain is therefore guaranteed to
reach at least a 1 - 1/e fraction, about 63 percent, of the optimal value, and
almost every practical diversifier is a greedy loop of this kind.

Standard measures cannot see any of this, because they score each document
independently and so reward a page of near-duplicates. Clarke and colleagues
introduced alpha-nDCG in 2008: each document is credited for the information
nuggets it contains, but a nugget already seen higher in the ranking earns
only a fraction alpha of the credit again. Agrawal and colleagues defined
intent-aware versions of familiar measures, such as nDCG-IA, that compute the
measure separately for each interpretation and average them weighted by how
likely each interpretation is. Chen and Karger's k-call asks only whether at
least k of the top n results are relevant; with k set to one it rewards a
ranking for getting anything right at all. From 2009 to 2012 the TREC Web Track
ran a diversity task over the ClueWeb09 collection, in which every topic came
with a list of subtopics and documents were judged against each subtopic
separately, which is what those measures need.
