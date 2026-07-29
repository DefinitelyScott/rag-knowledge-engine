# Evaluating a retrieval system

Retrieval quality is measured against a gold set: a list of questions, each
paired with the identifiers of the passages or documents that genuinely answer
it. Building that set by hand is tedious and is also the single highest-value
activity in a RAG project, because without it every change is a guess.

Hit@k, sometimes called success@k, asks a binary question: did at least one
relevant item appear in the top k results? It is the right metric when the
generator only needs to see one good passage to produce a correct answer.
Because it is binary it is coarse, and it saturates quickly as k grows.

Recall@k measures what fraction of all the relevant items were retrieved in the
top k. It matters when an answer must synthesise several sources; a question
that needs three documents is not well served by retrieving one of them.

Precision@k is the complement: what fraction of the returned results were
relevant. In RAG it matters mainly because irrelevant passages consume context
budget and can distract the generator.

Mean reciprocal rank averages 1/rank of the first relevant result across all
queries. It rewards putting the right passage first rather than fifth, which
matters when only the top few results are passed to a generator. An MRR of 0.5
means the first relevant result sits around position two on average.

Normalised discounted cumulative gain, or nDCG, goes further by supporting
graded relevance judgements and applying a logarithmic positional discount, so
a highly relevant document at rank one is worth more than the same document at
rank five. It is the standard metric in academic IR benchmarks.

Two habits keep evaluation honest. Report metrics at several values of k rather
than the one that flatters the system, and always evaluate the ablations, that
is, each retriever on its own alongside the ensemble. A hybrid system that does
not beat both of its components is not earning its complexity.
