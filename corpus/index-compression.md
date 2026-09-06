# Compressing a postings list

A postings list is a sorted run of document identifiers, and at web scale those
identifiers are the index. Storing them as raw 32-bit integers is both wasteful
and slow: wasteful because the numbers are highly redundant, and slow because
every query pays to move those bytes out of memory or off disk. Compression is
not a storage optimisation bolted on afterwards; it is what decides how much of
the index fits in cache, and therefore how fast the engine is.

The first move makes the numbers small. Because the list is sorted, each
identifier can be replaced by its difference from the one before it, so a run
like 1002, 1009, 1041 becomes 1002, 7, 32. These gaps are far smaller than the
values they encode, and they get smaller as the list gets longer, which is
convenient: the most common terms have the longest lists and therefore the
tightest gaps. Every scheme below is really a scheme for encoding small integers
efficiently.

Variable-byte encoding is the workhorse. Each byte spends seven bits on payload
and one on a continuation flag, so numbers below 128 cost one byte and nothing
is ever misaligned. Bit-aligned codes compress harder: the Elias gamma and delta
codes, from Elias in 1975, write a number's length in unary followed by its
remaining bits, which is close to optimal for small values but requires
bit-shifting to decode. Word-aligned schemes take the opposite bet, packing as
many equal-width integers as will fit into a machine word with a short selector
describing the packing. Frame-of-reference methods, of which PForDelta from
Zukowski and colleagues in 2006 is the best known, choose one bit width that
covers most of a block and store the handful of values that overflow it
separately as patches, so a single outlier cannot force the whole block wide.

The choice between them is not a compression-ratio contest, and reading it as
one is the standard mistake. A code that is ten percent smaller and twice as
slow to decode makes queries slower, because decoding happens on every query
while the saving is paid once. The measurement that settles it is end-to-end
query latency at a fixed index size, not bits per posting.

Two structural tricks matter more than the choice of code. Blocking the list and
storing a skip entry per block lets an intersection jump over blocks that cannot
contain a match without decoding them at all, which is why compressed lists are
usually decoded in fixed-size chunks rather than as one stream. And identifiers
can be reassigned before anything is encoded: sorting documents so that similar
ones receive adjacent numbers shrinks the gaps in every list they share, and for
web collections sorting by URL is a famously cheap way to get most of that
benefit for free.
