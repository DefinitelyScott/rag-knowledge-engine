# Distributed search: splitting one index across many machines

Every other document here quietly assumes the whole index fits on one machine
and one query runs against all of it. A large search engine cannot make that
assumption. The index is too big for one machine's memory, one machine cannot
answer queries fast enough, and one machine is one failure away from an outage.
Distributed search is the engineering that splits an index the system does
control across many machines and still returns a single ranked list. It is
different from federated search, where the separate indexes belong to other
people; here every piece runs the same code and shares one notion of a score.

There are two basic ways to cut an inverted index. Document partitioning, also
called a local index, assigns each document to one partition, or shard, and
builds a complete small index over that shard's documents, with postings for
every term it contains. A query is broadcast to every shard, each shard ranks
its own documents, and a broker merges the per-shard lists. Term partitioning,
also called a global index, assigns each term to one machine, which stores that
term's complete postings list across the whole collection. A query then only
touches the machines that hold its terms, but a multi-term query has to bring
long postings lists together on one machine to intersect or score them, which
moves a great deal of data over the network. Popular terms also overload the
machines that own them. Moffat, Webber, Zobel and Baeza-Yates proposed in 2007
a pipelined form of term partitioning, passing a partially scored accumulator
set from one term's machine to the next, which reduces the transfer, but the
load imbalance remained the harder problem. Document partitioning is the layout
large web engines have described using, and it is the one assumed below.

The broker's merge is simple only because each shard ranks with the same
function. To guarantee the true top k, every shard must return its own top k,
since in the worst case all of the best documents live on one shard; asking
each shard for fewer saves work and gives up that guarantee. A subtler problem
is collection statistics. A shard that computes inverse document frequency from
its own documents gets a slightly different value for each term than the whole
collection would give, so the same document would score differently depending
on which shard it landed on. When documents are assigned to shards at random
the differences are small, because every shard is a sample of the same mix.
When shards are organised by topic they are not, and the system has to
distribute global statistics to every shard so that scores stay comparable
before they are merged.

Splitting by documents makes each shard's index smaller and so each shard
faster, but it also means every query waits for the slowest shard to answer.
With hundreds of shards, an event that delays one machine in a thousand will
delay a large fraction of all queries. Dean and Barroso described this in "The
Tail at Scale" in 2013: the latency users see is set by the tail of the
per-shard distribution, not its median. Their remedies include hedged requests,
which send a second copy of a slow request to another replica after a short
wait and take whichever answers first, and returning a result built from the
shards that did answer when a deadline passes, accepting a slightly incomplete
ranking rather than a late one.

Replication is the other axis. Partitioning decides how much of the collection
each machine holds; replication decides how many copies of each partition
exist. More partitions shorten each query, while more replicas raise how many
queries per second the system can absorb and keep a shard available when a
machine fails. Barroso, Dean and Hölzle's 2003 description of Google's cluster
architecture laid out exactly this grid: the index split into shards, each
shard served by a pool of replicas, with a load balancer choosing a replica per
request.

Not every query needs the whole collection. A tiered index places the documents
judged most likely to be useful, by quality or popularity, in a small first
tier and the rest in larger lower tiers. A query is answered from the first
tier and falls through to the next only when the first does not return enough
good results; Risvik, Aasheim and Lidal described such a multi-tier design for
web search in 2003. Selective search goes further. Kulkarni and Callan
partitioned a collection into topical shards by clustering its documents, then
used a resource-selection algorithm of the kind built for federated search to
send each query to only the few shards likely to hold relevant documents. Most
shards are never consulted, which cuts total work sharply, at the cost of
occasionally missing a relevant document that was clustered somewhere
unexpected.
