# Index maintenance: keeping an inverted index current as documents change

An inverted index is built for reading. Each posting list is a sorted, compressed,
contiguous run of document identifiers, which is what makes queries fast and is
exactly what makes writes awkward. Inserting one document touches the posting
list of every distinct term it contains, and adding an entry in the middle of a
compressed list means rewriting it. A collection that never changes can be built
once. Nearly every real one changes, so engines need a strategy for absorbing
new, edited and deleted documents without rebuilding everything each time.

The simplest strategy is to rebuild: periodically reindex the whole collection
and swap the new index in. It is easy to reason about and produces the most
compact index, but it makes new documents invisible until the next build, and
the cost grows with the collection rather than with the change. At the other
extreme, in-place update appends to the existing posting lists, leaving spare
space at the end of each so that most appends need no move. This keeps documents
searchable immediately but fragments the disk, and a list that outgrows its
slack must be relocated.

The strategy most engines settle on is merge-based. New documents are indexed in
memory into a small auxiliary index. When it fills, it is written out as an
immutable *segment*, a complete small index of its own, and queries search the
main index and every segment and merge the answers. Left alone, segments would
accumulate and every query would have to consult more of them, so segments are
merged into larger ones in the background. Lester, Zobel and Williams studied
how to schedule this, and the standard answer is *geometric* or logarithmic
merging: merge segments only with others of similar size, so a segment of size n
is rewritten roughly log n times over its life rather than once per batch.
Büttcher and Clarke compared hybrids that merge short lists and update long ones
in place. The same shape appears in log-structured merge trees in databases, and
the cost has the same name there, write amplification: the same postings are
written to disk several times on their way to the largest segment.

Immutable segments make deletion and edits indirect. A deleted document is not
removed from its segment; it is recorded in a deletion list, often a bitmap, and
filtered out of results at query time. Its postings are physically dropped only
when its segment is next merged, which is why disk usage does not fall at the
moment of deletion. An edit is a deletion of the old version plus an insertion
of the new one. To make a new document visible quickly without writing a durable
segment for each one, engines offer near-real-time search: they refresh a
searchable view of the in-memory segment every second or so, and trade a short
delay in visibility for far fewer small segments.

Several things outside the posting lists depend on the collection and go stale
when it changes. Scoring functions such as BM25 use collection-wide statistics,
the document frequency behind IDF and the average document length, so adding
documents changes the scores of documents whose text was never touched; when
statistics are kept per segment, the same word can be weighted differently in
different segments until they merge. Compression schemes that renumber documents
so that similar ones sit together depend on a fixed numbering, and new documents
arrive with no place in it. Result caches must be invalidated when a refresh
could change their answers. Distributed engines face all of this once per shard.
