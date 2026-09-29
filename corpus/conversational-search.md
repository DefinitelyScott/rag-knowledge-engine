# Conversational search: retrieving for a question that depends on the last one

A search box sees each query alone. A conversation does not work that way.
After asking what causes a disease, a user asks "how is it treated?", and then
"what about in children?". The second question hides its subject behind a
pronoun, and the third leaves out almost everything, relying on ellipsis.
Neither can be answered by a retriever that sees only the words typed in that
turn, because the words that would match the right passage were said earlier.

TREC's Conversational Assistance Track, run from 2019 to 2022 by Dalton, Xiong
and Callan, turned this into a shared benchmark: sequences of related turns,
passages to retrieve for each, and relevance judged per turn. Its most
consistent finding was the gap between two kinds of runs. Systems given a
human-written, self-contained version of every turn did far better than systems
that had to interpret the raw conversational turn, and that gap became the
yardstick for everything that tried to close it.

The dominant way to close it is to rewrite the question. The CANARD dataset,
released by Elgohary, Peskov and Boyd-Graber in 2019, pairs context-dependent
questions with human rewrites that resolve every reference, and
sequence-to-sequence models such as T5 fine-tuned on those pairs became a
standard baseline. The appeal is modularity: once the turn is self-contained,
any existing retriever, lexical or dense, can be used unchanged. The cost is
that the rewriter is a single point of failure. A pronoun resolved to the wrong
earlier entity produces a fluent, confident query about the wrong thing, and
nothing downstream can tell.

The alternative is to expand rather than rewrite: pick the important terms from
earlier turns and add them to the current one. Expansion cannot hallucinate
a subject that was never mentioned, but it cannot tell which earlier topic is
still current either, so a conversation that has moved on keeps pulling the
old topic back in. Some systems run both and fuse the two result lists, on the
reasoning that the rewrite and the expansion fail in different ways.

Dense retrievers offered a third route: encode the whole conversation directly.
ConvDR, from Yu and colleagues in 2021, trains a conversational query encoder
to produce the embedding that the manual rewrite would have produced, a
teacher-student setup that needs no rewrite at query time, only the history.

Two difficulties are specific to this setting. Errors compound, because a
wrong interpretation at turn three changes the retrieved context that turn four
is read against. And the system can ask instead of guess. Aliannejadi and
colleagues showed in 2019 that choosing a good clarifying question when the
request is ambiguous, and using the answer, can improve the ranking that
follows, which turns retrieval from a single response into a short exchange.
