# Snippet generation: summarising a result for the query that found it

A ranked list is only as useful as the few lines shown under each title. Those
lines, the snippet, are what a searcher reads to decide whether to click, so a
snippet is a second relevance judgement made on the searcher's behalf. A poor
one hides a good document, and a persuasive one on a poor document wastes a
visit. The central idea is that a snippet is a summary of the document *for this
query*, not a summary of the document.

Tombros and Sanderson showed in 1998 that query-biased summaries help. In their
experiment, people judging whether documents were relevant did so faster and
more accurately when each result was shown with sentences chosen because they
matched the query than when it was shown with the first sentences of the text
or a static abstract. The leading sentences describe what the author wanted to
lead with; the matching sentences describe how the document touches what the
searcher actually asked.

The classic method is sentence extraction. The document is split into sentences,
each is scored, and the top few are stitched together in their original order.
The score is a weighted sum of simple features. A sentence is rewarded for
containing query terms, for containing many *distinct* query terms rather than
one repeated, and for holding the query terms close together, ideally as the
contiguous phrase. Position contributes a small bonus for sentences near the
start of the document, and Luhn's much older observation that a sentence dense
in significant words is a good candidate for an abstract still lies behind the
density feature. Titles and headings are often boosted because they are written
to be read in isolation.

Choosing the sentences is the easy half. Making it fast is the hard half, and
Turpin, Tsegay, Hawking and Williams described why in 2007. Generating a
snippet needs the original text of every document on the result page, and the
index holds only terms, not text. Fetching ten documents from disk for every
query costs ten random reads, which dominates the time the ranking took. Their
fix was to store a compressed copy of each document in a purpose-built store
and keep recently used ones in memory. They also reordered the sentences inside
each stored document so the sentences most likely to be chosen, typically the
ones with the most distinctive terms, are read first and the rest need not be
decompressed at all. The snippet cost then becomes a fraction of the ranking
cost.

Snippets have limits that a ranking metric cannot see. Showing only the
matching fragment can mislead when the sentence is lifted out of a context that
negates it. Extractive snippets also fragment awkwardly, so engines mark
omissions with an ellipsis, and may highlight the query terms inside the text,
which draws the eye but can make a document look relevant for the wrong reason.
Where the page has structured data, an answer box or a list of attributes can
replace the text excerpt entirely.

In a retrieval-augmented pipeline the same problem returns in a harder form. The
passages handed to a language model are themselves snippets, chosen to fit a
limited context window, and a passage that matches the query wording may still
omit the clause that answers it. Citation generation then has to point back from
the extracted text to the whole document it came from. Evaluating snippets is
difficult for the reason ranking evaluation is: the ground truth is a human
judgement about usefulness, and the cheap proxies such as term overlap with a
reference summary only correlate with it loosely.
