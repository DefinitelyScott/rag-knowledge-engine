# Pooling and judgments: where a gold set comes from, and what it leaves out

A metric computed against a gold set inherits every property of that set,
including the ones nobody intended. Judging every document against every query
is impossible at any realistic corpus size, so test collections are built by
pooling: run a set of systems, take the top `d` results from each, merge them,
and judge only the union. Everything outside the pool is scored as irrelevant
by default. That default is the assumption the whole apparatus rests on, and it
is false - it is merely cheap enough and close enough to be workable.

Pool depth controls how false. Shallow pools miss relevant documents that no
contributing system ranked highly, and the documents they miss are not a random
sample: they are exactly the ones an unusual system would find. This produces
pool bias, where a system that did not contribute to the pool is evaluated
partly on unjudged documents and scored below its true quality. Zobel's 1998
analysis measured the effect and found collections usable but demonstrably
incomplete, with the shortfall concentrated where it hurts novel approaches
most. A collection built from lexical systems will quietly punish the first
semantic one that arrives.

Metrics can be built to tolerate the gap rather than pretend it is not there.
Bpref, from Buckley and Voorhees in 2004, is computed only over judged
documents and ignores unjudged ones instead of counting them as misses, which
makes it far more stable when judgements are incomplete. Inferred AP, from
Yilmaz and Aslam in 2006, treats the pool as a sample and estimates average
precision from it. Both accept a less precise number in exchange for one that
degrades gracefully, which is the right trade when the alternative is a precise
number computed on a false assumption.

Assessors disagree, and the standard result is more reassuring than it sounds.
Voorhees found in 2000 that having different assessors judge the same
collection changed absolute scores noticeably while leaving the *relative*
ordering of systems largely intact. That is the property a benchmark actually
needs: it licenses "system A beat system B" without licensing "system A scores
0.43". Reporting an absolute figure to three decimals implies a precision the
judgements do not carry.

The modern large collections trade depth for breadth and take the opposite
risk. MS MARCO pairs each query with typically one passage marked relevant,
drawn from what a search engine surfaced, so a system retrieving a genuinely
better passage is scored as having missed. At that judgement density recall is
barely distinguishable from hit rate, and a small hand-built set with several
documents marked per question can carry information that a set a thousand times
its size does not.
