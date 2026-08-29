# Negatives: how a dense retriever is actually trained

A bi-encoder is trained by contrast. The positive pair is given by the data,
so the only real design decision is which wrong passages the model is pushed
away from, and that decision determines what the finished retriever can
distinguish. Train against passages sampled uniformly from the corpus and the
task is trivial: almost any random passage is about a different subject
entirely, so the model learns topic separation, converges quickly, and arrives
at query time unable to choose between two passages on the same topic where
only one answers the question. That choice is the entire ranking problem.

The cheap improvement is to reuse the batch. In-batch negatives treat the other
questions' passages as negatives for this question, so a batch of size B yields
on the order of B squared comparisons while paying to encode only B passages.
Dense Passage Retrieval, from Karpukhin and colleagues in 2020, pairs this with
one genuinely hard negative per question, mined by running a lexical retriever
and taking a passage it ranks highly that nonetheless does not contain the
answer. Lexically plausible and semantically wrong is precisely the case the
encoder needs to see.

ANCE, from Xiong and colleagues in 2020, names the deeper problem: the
negatives the model trains on are not drawn from the distribution of documents
it will actually confuse at inference. It samples them instead from an
approximate nearest neighbour index built over the whole corpus in the model's
own representation space. That index is only correct for the encoder that built
it, and the encoder is changing on every step, so the index goes stale as
training proceeds and has to be rebuilt. ANCE refreshes it asynchronously,
running the index-building in parallel with training rather than stopping to
wait for it.

Mining harder negatives introduces the opposite failure. Relevance labels on
large training sets are shallow, often one marked passage per query out of
millions, so a passage the miner ranks first is frequently an unlabelled
correct answer rather than a wrong one. Training on it teaches the model to
reject something true. RocketQA, from Qu and colleagues in 2021, addresses this
with cross-batch negatives, denoised hard negatives where a cross-encoder
filters mining candidates before they are used, and data augmentation. The
cross-encoder appears here in an unfamiliar role: not as a second stage that
re-scores results at query time, but as a teacher whose judgements become
training signal for the model that will run first and alone.

None of this fails loudly. Loss curves descend on bad negatives and on good
ones alike, and a model trained against random passages looks healthy right up
until it is measured. The only diagnosis is retrieval accuracy on held-out
queries the mining process never touched.
