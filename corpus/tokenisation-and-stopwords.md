# Tokenisation and stop words: deciding what counts as a term

Before any ranking function sees a document, something has to decide what the
terms are. Tokenisation splits running text into units, and a second step may
then discard the units judged not worth indexing. Both decisions look trivial
and both leak into every later stage, because a term the index never stored
can never be matched, however good the scoring is.

Splitting on whitespace and punctuation is the usual starting point and it
fails in predictable places. A hyphen can join a compound ("state-of-the-art"),
mark a line break, or separate a prefix ("co-operate"), and treating it as a
space or as nothing is wrong some of the time. An apostrophe turns "O'Neill"
into two fragments. Dots break "U.S.A." and version numbers, and punctuation
that carries meaning disappears altogether, so "C++" and "C#" both collapse to
the letter C. Case folding raises recall by letting "Search" match "search",
and costs precision where case separates senses: "Bush" the person and "bush"
the plant, or "US" the country and "us" the pronoun. Languages written without
spaces, such as Chinese and Japanese, need a word segmenter, or can avoid the
question by indexing overlapping character bigrams, which trades a larger
vocabulary for not depending on a segmentation dictionary.

Stop words are the most frequent words of a language, such as "the", "of" and
"and". Word frequencies follow Zipf's law: the r-th most common word occurs with
frequency roughly proportional to 1/r, so a few dozen words account for a large
share of all running text. Luhn argued in 1958 that a word's power to separate
documents peaks in the middle of the frequency range, since very common words
appear everywhere and very rare words appear too seldom to link anything. That
reasoning supports a stop list: drop the head of the distribution before
indexing and the postings lists shrink noticeably, and queries stop paying to
read the longest lists in the index.

The case against stop lists has grown stronger as machines have changed. Some
queries are made almost entirely of stop words, and removal destroys them:
"to be or not to be", the band "The Who", or "vitamin A". Phrase queries
suffer, because deleting a word silently changes the adjacency that the phrase
depends on. Ranking functions such as BM25 and TF-IDF already give frequent
terms a low inverse document frequency, so keeping them does little harm to
the scores; the original motivation was cost, not accuracy. Index compression
has also cut that cost sharply, because a postings list for a very common word
is dense and its document gaps are tiny and compress well. Most modern engines
therefore index everything and rely on weighting, or on a bounded stop list for
a specific purpose such as query-time pruning.

A middle path keeps the words but makes them cheap. A common-grams or
word-pair index stores an extra entry for each frequent word combined with its
neighbour, so a phrase such as "the who" is answered from one short list
instead of by intersecting the enormous list for "the" with another. The
speed-up is paid for with a larger index and more complicated query rewriting.
Whatever the policy, it must be applied identically at index time and at query
time, and its effect is best measured rather than assumed: run the same
questions with and without the stop list and compare the metrics, just as for
stemming.
