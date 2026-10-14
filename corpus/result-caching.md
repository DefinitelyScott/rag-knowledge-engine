# Caching in search engines: answering a query without evaluating it

The cheapest query is one the engine has already answered. Search engine query
streams are highly repetitive: query popularity follows a heavy-tailed,
roughly power-law distribution, so a small set of very popular queries accounts
for a large share of all traffic. Markatos studied this in 2001 on a log from
the Excite engine and showed that keeping recently returned result pages in
memory lets a sizeable fraction of incoming queries be answered without
touching the index at all. The same distribution has a long tail: a large share
of distinct queries are issued only once in a log, and no cache can help with
a query the first time it arrives. Those first occurrences, the compulsory
misses, set an upper bound on the hit rate of any result cache, however large.

A result cache stores, for a query string, the ranked list of document
identifiers and snippets it produced. Two families of policy decide what it
holds. A static cache is filled offline with the most frequent queries from a
past log and never changes until it is rebuilt. A dynamic cache admits queries
as they arrive and evicts by a rule such as least recently used, so it adapts
to bursts that a past log could not anticipate. Fagni, Perego, Silvestri and
Orlando proposed in 2006 to split the space between the two, a static part for
queries that are popular over long periods and a dynamic part for queries that
are popular for a few hours, and found the combination outperformed either
policy alone at the same size.

Many queries in the tail will never be repeated, and admitting each one into a
dynamic cache evicts something that might have been. Baeza-Yates and colleagues
studied admission policies in 2007: keep a query out of the cache unless some
cheap feature of it, such as its length in characters or terms or how often it
has been seen before, suggests it is likely to be issued again. Rejecting
probable one-off queries protects the entries that earn hits.

Results are not the only thing worth caching. A posting-list cache keeps the
inverted lists of some terms in memory, and it helps even for a query never
seen before, as long as its terms have been. In their 2007 study of caching in
search engines, Baeza-Yates and colleagues showed that a static posting-list
cache reaches a higher hit rate than a result cache of the same memory size,
because terms repeat across queries far more than whole queries do. Choosing
which lists to keep is a knapsack-like trade: a frequently queried term with a
very long list may cost more memory than several moderately popular terms with
short lists, so they ranked terms by query frequency divided by list size
rather than by query frequency alone. Long and Suel proposed in 2005 a third
level between the two, caching the intersections of postings lists for pairs of
terms that often occur together in queries, so a later query containing the
pair starts from an already intersected list. Saraiva and colleagues had earlier
combined result and posting-list caches into a two-level design.

Caching collides with freshness. When documents are added, removed or changed,
a cached result list may no longer be the list the current index would produce,
and the engine serves an answer that is quietly out of date. Invalidating every
cached entry whenever the index changes would empty the cache constantly on a
web-scale engine that updates continuously. One family of approaches predicts
which entries an update is likely to affect and invalidates only those.
Cambazoglu and colleagues argued in 2010 for a different trade: give every
entry a time-to-live after which it expires, and use the backend's idle
capacity during low-traffic hours to re-evaluate cached queries before they
expire, prioritising those most popular and most likely to have changed. With
a large cache, hit rate stops being limited by space and is limited instead by
how long results can be trusted to stay correct.
