# Question answering: readers, retrievers and exact match

Search returns documents; question answering tries to return the answer. In the
open-domain setting the system gets a natural-language question and no
paragraph to look in, so it first has to find candidate passages in a large
collection and then read them. That split gave the field its standard shape,
the retriever-reader pipeline: a fast first stage narrows millions of passages
to a few dozen, and a slower reader model examines each one for an answer.

Most of the early benchmarks defined the reader's job as extractive. SQuAD,
released by Rajpurkar and colleagues in 2016, gave a Wikipedia paragraph and a
question written by a crowd worker, and the answer was always a contiguous span
of that paragraph. A reader then predicts two positions, where the span starts
and where it ends, and the best-scoring pair under a length limit is returned.
BERT-style encoders made this very effective: a start vector and an end vector
are learned, and each token's score is its dot product with them. SQuAD 2.0
added questions that have no answer in the paragraph, which forces the reader
to learn when to abstain rather than always returning its best span. That is
the same decision a retrieval-augmented system faces when the retrieved text
does not cover the question.

DrQA, from Chen and colleagues in 2017, is the usual reference for the
open-domain pipeline. It paired a classical retriever, TF-IDF over unigrams and
bigrams, with a recurrent neural reader, and answered questions over all of
Wikipedia. Later work replaced the first stage. Dense Passage Retrieval, by
Karpukhin and colleagues in 2020, trained two encoders so that a question and
the passage holding its answer land close together, and reported higher
top-20 passage accuracy than BM25 on several open-domain datasets, though BM25
stayed competitive on questions that hinge on rare entity names. A hybrid of
the two lists is now a common baseline for exactly that reason.

Datasets differ in how realistic their questions are. SQuAD questions were
written by people looking at the paragraph, so they often reuse its wording,
which flatters lexical matching. Natural Questions, from Kwiatkowski and
colleagues at Google in 2019, uses real queries typed into the search engine,
paired with a Wikipedia page and annotated with a long answer (a paragraph) and
a short answer (a span or yes or no), and many questions have no short answer
at all. TriviaQA collects trivia questions written independently of the
evidence documents, which makes the evidence noisier and the retrieval harder.

Answers are scored in two ways. Exact match is one when the predicted string,
after lowercasing and stripping punctuation and articles, equals one of the
gold answers. Token-level F1 gives partial credit for overlap. Both are blunt.
A correct answer written another way, such as a date in a different format or
a nickname, counts as wrong, and benchmarks that use only a handful of
reference strings underestimate good systems. In the open-domain setting some
evaluations only check whether a gold answer string appears in a retrieved
passage, which can reward a passage that contains the string by accident.

A different line of work skips retrieval at inference time. Roberts, Raffel and
Shazeer showed in 2020 that a large sequence-to-sequence model could answer
questions with the knowledge stored in its parameters, called closed-book
question answering. It is convenient but cannot cite a source, goes stale when
the world changes, and states wrong answers with the same fluency as right
ones. Generating answers from retrieved passages, as retrieval-augmented
generation does, trades that convenience for the ability to point back at the
text that supports a claim.

Two limits recur. Multi-hop questions need facts from more than one passage,
so a retriever scored on single passages can look strong while the reader still
fails. And a reader trained to always find a span will produce one even when
the evidence is missing, so an abstention rule, whether a learned
no-answer class or a threshold on retrieval and reader scores, is part of the
system, not an afterthought.
