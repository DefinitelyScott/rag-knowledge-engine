# Federated search: querying collections you cannot index yourself

Everything else in this corpus assumes one index that the system built and
controls. Federated search, also called distributed information retrieval,
drops that assumption. The documents live behind many separate search services,
each with its own index, its own ranking function and its own idea of what a
score means, and the system can only send each one a query and read back what
it returns. Callan's survey of the field split the work into three problems:
describing what each resource holds, selecting which resources to ask, and
merging what comes back into one list.

Resource description is easy when a resource cooperates and publishes its
statistics, and hard when it does not. Query-based sampling, proposed by Callan
and Connell in 2001, handles the uncooperative case by probing: send a series
of single-word queries, download the top documents each one returns, and build
the resource's description from that sample. A few hundred sampled documents
turned out to be enough to capture most of a resource's frequent vocabulary.
A sample says nothing about how big the resource is, so size has to be
estimated separately; the sample-resample method sends a term from the sample
back to the resource and compares the hit count it reports with the term's
frequency in the sample.

Resource selection ranks the services themselves. GlOSS, from Gravano and
colleagues in 1994, estimated how many documents in each database were likely
to match. CORI, from Callan, Lu and Croft in 1995, treats each collection as
though it were one enormous document and scores it with a belief formula shaped
like tf-idf: how many of its documents contain the term takes the place of term
frequency, and how many collections contain the term takes the place of
document frequency. ReDDE, from Si and Callan in 2003, instead pools every
sample into one centralised sample index and runs the query against it. Each
highly ranked sampled document is evidence that its home collection holds
relevant material, scaled up by the ratio of the collection's estimated size to
its sample size, which keeps a small sample from a very large collection from
being overlooked.

Result merging is where federated search meets the problem of comparing scores
from different scorers. Two engines can return the same document with scores
on unrelated scales, and even two copies of the same algorithm disagree because
their collection statistics differ. CORI's merging rule reweights each
document's score by how highly its collection was selected. The
semi-supervised method of Si and Callan goes further. Some returned documents
also appear in the centralised sample index, so for those the system knows both
the resource's score and a score on one common scale. It fits a regression per
resource on those overlapping documents and uses it to map every returned
score onto the shared scale before merging.

Web search engines face a version of the same decision whenever they choose
whether to blend news, images or video into a results page. Arguello and
colleagues framed this in 2009 as vertical selection, and the evidence they
used, including query logs from each vertical and the vertical's own results,
mirrors the resource-selection signals above.
