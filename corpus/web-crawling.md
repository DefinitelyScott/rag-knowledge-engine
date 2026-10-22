# Web crawling: frontier, politeness and freshness

A search engine can only rank what it has fetched. A crawler starts from a set
of seed URLs, downloads each page, extracts the links, and adds the ones it has
not seen to a queue called the frontier. Repeating that loop is simple; doing it
at the scale of billions of pages, without harming the sites being fetched, and
while keeping the stored copies reasonably current, is where the engineering
and the research are.

Politeness comes first. A site's robots.txt file states which paths a crawler
may fetch, and well-behaved crawlers obey it and also limit how often they hit
one host. A naive single queue would fetch many URLs from the same server
back to back. The Mercator design, described by Heydon and Najork in 1999,
splits the frontier in two. Front queues order URLs by priority; back queues
each hold URLs for a single host, and a heap of release times ensures that a
host is not contacted again before a minimum delay has passed. Many fetcher
threads can then run in parallel while each server still sees a slow, steady
trickle of requests.

The crawler also has to recognise URLs it has already seen. The same page can
be written many ways, so URLs are normalised first: lowercase host, default
port removed, relative paths resolved, fragments dropped. The seen set is far
too large for an exact in-memory hash table at web scale, so it is often a
disk-backed structure or a Bloom filter, which answers membership with a small
chance of a false positive. A false positive here means a new page is wrongly
skipped, which is usually an acceptable loss. Spider traps, such as calendars
that generate an endless chain of next-month links, are handled with limits on
URL length, depth and the number of pages taken from one host.

Which page to fetch next matters because no crawl ever finishes. Cho,
Garcia-Molina and Page showed in 1998 that ordering the frontier by a measure
of importance, such as the number of known links into a page or an estimate of
its PageRank, finds the important pages sooner than a random order. Najork and
Wiener later found that plain breadth-first crawling already reaches pages of
high PageRank early, because well-linked pages tend to be discovered through
many short paths. The cheap default is therefore a surprisingly strong baseline.

Pages change, so a crawl is also a refresh problem. Cho and Garcia-Molina
defined freshness as the fraction of stored copies that still match the live
page, and age as how long a stored copy has been out of date. They modelled
page changes as a Poisson process and compared policies. The intuitive choice,
revisiting each page in proportion to how often it changes, performed worse than
visiting every page at the same rate. A page that changes every few minutes goes
stale again almost immediately after any visit, so the visits spent on it buy
little, while the same effort on a slowly changing page keeps it fresh for a long
time. Practical crawlers therefore cap the revisit rate of fast-changing pages
and spend the savings on the rest. Content behind forms or logins, the deep web,
is not reached by link-following at all, and needs separate techniques.
