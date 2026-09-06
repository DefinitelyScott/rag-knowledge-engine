# Dynamic pruning: skipping work you can prove is wasted

Scoring every document that contains any query term is enormously wasteful when
the caller only wants the best ten. Most of that work goes into documents that
were never going to place, and the useful question is whether an engine can know
that in advance rather than discovering it after paying. Dynamic pruning is the
family of algorithms that answers yes, and the important property of the good
ones is that they are rank-safe: they return exactly the top-k an exhaustive
scan would return, having merely declined to compute scores that could not have
changed it.

The trick is an upper bound. For each term the index stores the largest
contribution that term could ever make to any document's score, computed once at
index time. During a query the engine keeps a heap of the k best documents seen
so far, and the score of the worst of them is a threshold. If the sum of the
upper bounds of the terms a document actually contains is below that threshold,
the document cannot enter the heap, and nothing about its real score needs to be
known.

WAND, from Broder and colleagues in 2003, turns that observation into a
traversal. Cursors over the postings lists are kept sorted by their current
document identifier and their upper bounds are accumulated in that order until
the running total passes the threshold; the term where that happens is the
pivot, and the document it points at is the first one that could possibly
qualify. Every cursor behind the pivot is advanced straight to it, skipping
whatever lay between. MaxScore, described by Turtle and Flood in 1995, reaches
the same goal by splitting terms into an essential set that can still produce a
qualifying document on its own and a non-essential set that cannot; the
non-essential lists stop driving iteration entirely and are consulted only to
finish scoring candidates the essential ones surfaced.

A single bound per term is pessimistic, because one unusually strong document
raises the bound for the whole list. Block-Max WAND, from Ding and Suel in 2011,
stores a maximum per block of postings instead, so the bound the engine reasons
with is local and much tighter, and skipping happens at block granularity
against numbers that are actually representative.

How much any of this saves depends on skew rather than on the algorithm. Pruning
works because a few documents score far above the rest, so the threshold rises
quickly and disqualifies almost everything; it degrades when scores are flat and
no bound is ever comfortably beaten. That is the uncomfortable interaction with
expansion-based sparse retrieval, where inflating documents with extra terms
lowers the peaks that pruning depends on. Raising the threshold artificially
prunes harder still, but that buys speed by giving up rank-safety, which is a
different bargain and should be reported as one.
