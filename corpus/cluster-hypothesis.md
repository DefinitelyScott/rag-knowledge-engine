# The cluster hypothesis and document clustering

Retrieval systems usually treat documents as independent items to be scored
against a query. The cluster hypothesis says they are not independent. In van
Rijsbergen's 1979 formulation, closely associated documents tend to be relevant
to the same requests. If that holds, then the neighbours of a relevant document
are more likely than average to be relevant too, and grouping similar documents
together should help a searcher.

The hypothesis can be tested. Voorhees proposed a nearest-neighbour test in
1985: for each relevant document, look at its five most similar documents in the
collection and count how many are also relevant to the same query. Averaged over
queries, a high count supports the hypothesis for that collection and that
representation. Smucker and Allan later refined the test and showed that
collections differ a good deal. The result also depends on how similarity is
measured, so a hypothesis that fails with one document representation can hold
with another.

Several algorithms build the groups. Hierarchical agglomerative clustering
starts with every document alone and repeatedly merges the closest pair of
clusters. The choice of how to measure distance between two clusters matters:
single link takes the closest pair of members and tends to chain unrelated
documents into long straggling clusters, complete link takes the farthest pair
and favours compact ones, and group-average takes the mean over all pairs. All
of these need the pairwise similarities, so cost grows with the square of the
number of documents, which rules them out for a whole web collection. K-means
is far cheaper, with cost proportional to documents times clusters times
iterations, but it needs the number of clusters in advance and can land in
different solutions from different random starting points.

The first use was interactive. Cutting, Karger, Pedersen and Tukey built
Scatter/Gather in 1992. The system clusters a collection into a handful of
groups with a short summary of each. The user picks one or more groups that look
promising, the system gathers those documents and clusters them again, and the
user repeats this to narrow down. To make it fast enough to be interactive, they
used algorithms that cluster a sample first and then assign the rest. Hearst and
Pedersen reexamined the hypothesis in 1996 by clustering only the documents
returned for a query, and found that relevant documents concentrated in a few
clusters far more than they did when the whole collection was clustered in
advance. This query-specific clustering is the form that became common. Zamir
and Etzioni's suffix tree clustering, in 1998, grouped web search results by
shared phrases and gave each group a readable label, which is the hardest part of
the task, since a cluster with no good name is hard for a person to use.

Clusters can also improve ranking itself. In cluster-based retrieval, a
document's language model is smoothed with the model of the cluster it belongs
to, so a document gains score from relevant-looking neighbours. Liu and Croft,
and separately Kurland and Lee, reported gains from this in 2004. The idea is
close to pseudo-relevance feedback, since both borrow evidence from related
documents.

Clustering is also used for efficiency. Cluster pruning, described in standard
textbooks, picks about the square root of the number of documents as leaders and
attaches every other document to its nearest leader. A query is compared with
the leaders, then only with the followers of the nearest one. The answer is
approximate, because a relevant document attached to another leader is never
seen. Selective search, studied by Kulkarni and Callan, applies the same logic to
a distributed index. Documents are placed on shards by topic, and a query is sent
only to the few shards most likely to hold answers, saving most of the work that
searching every shard would cost.

The hypothesis fails in known cases. A query with several distinct aspects can
have relevant documents spread across clusters that have little in common. Clusters
are also unstable: small changes to the data or the algorithm can move the
boundaries. For the same reason, search diversification starts from the opposite
assumption, that the top of a ranking is too redundant and needs documents from
different groups.
