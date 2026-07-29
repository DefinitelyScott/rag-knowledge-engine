# Inverted indexes

An inverted index is the data structure that makes keyword search fast. Instead
of storing, for each document, the list of terms it contains, it stores for
each term the list of documents that contain it. That list is called a posting
list, and it usually carries the term frequency in each document, and sometimes
the positions at which the term occurs.

Querying then means fetching one posting list per query term and intersecting or
unioning them, rather than scanning every document. Because posting lists are
sorted by document identifier, the merge is a linear walk over sorted lists, and
skip pointers let the walk jump over long stretches that cannot match.

Positional information enables phrase queries. Storing the offsets of each
occurrence lets the engine verify that "vector database" appears as adjacent
words rather than merely somewhere in the same document, at the cost of a
substantially larger index.

Compression matters at scale. Document identifiers within a posting list are
stored as deltas, the gaps between consecutive identifiers, which are small
numbers that compress well with variable-byte or bit-packed encodings. A
well-compressed index can be a fraction of the size of the raw text it covers.

Updating an inverted index in place is awkward, because inserting a document
means touching one posting list per distinct term it contains. Production
systems typically write new documents into small in-memory segments, flush them
to immutable on-disk segments, search all segments and merge results, and
periodically compact segments in the background. Deletions are recorded as
tombstones and only physically removed during compaction.

For a corpus of a few thousand passages, dictionaries of term counts held in
memory serve the same purpose with none of the complexity, which is why a
teaching implementation can compute BM25 directly from per-document counters
and still return results in microseconds.
