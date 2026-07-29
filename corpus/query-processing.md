# Query processing and expansion

The query is the weakest link in most retrieval systems. Users type two or
three words, use different vocabulary from the corpus, misspell terms, and ask
follow-up questions that only make sense given the previous turn. Query
processing is the set of techniques that repair this before retrieval runs.

Normalisation comes first: lowercasing, folding accented characters, and
splitting on punctuation so that "state-of-the-art" and "state of the art"
produce comparable tokens. Stemming and lemmatisation reduce inflected forms to
a common root, so that "retrieving", "retrieved" and "retrieval" match. Stemming
is crude but cheap; it sometimes conflates unrelated words, which is why
stemming is often skipped in favour of leaving IDF to handle the statistics.

Stop word removal drops extremely common function words. Its value is smaller
than people expect for BM25, since IDF already gives such words almost no
weight, but it matters for similarity measures that normalise by vector length,
where a pile of function words can dominate the norm.

Query expansion adds terms to the query. Pseudo-relevance feedback, the classic
unsupervised form, retrieves an initial result set, extracts the terms that are
distinctive within it, appends them to the query, and searches again. It helps
when the initial results are good and hurts when they are not, an effect known
as query drift. Synonym expansion using a thesaurus or a domain glossary is
more predictable and easier to debug.

More recent systems expand queries with a language model, either by generating
paraphrases and retrieving for each, or by generating a hypothetical answer
document and retrieving against that. Both trade latency and an extra failure
mode for better recall on badly phrased questions.

For multi-turn conversations, query rewriting is essential: "what about the
second one?" must be rewritten into a self-contained question before it is sent
to a retriever that has no memory of the conversation.
