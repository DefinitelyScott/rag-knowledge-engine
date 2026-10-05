# Query auto-completion: predicting the query from its first few characters

Query auto-completion suggests whole queries while the user is still typing
them. Every keystroke sends the current prefix to a service that must return a
short ranked list, typically up to ten candidates, within the time it takes to
press the next key, which leaves a budget of tens of milliseconds. The task is
a ranking problem in its own right, but an unusual one: the input is a
fragment, often a fraction of a word, and the candidates are not documents but
other people's queries.

The standard baseline is Most Popular Completion, named by Bar-Yossef and Kraus
in 2011. Collect every query in a past log that begins with the typed prefix
and rank them by how often they were issued. It is the maximum-likelihood
answer to "which query that starts this way is most likely", and it is hard to
beat on common prefixes. It fails in two directions. On a prefix nobody has
typed before, it has nothing to offer at all. And a single global frequency
ignores who is typing and when.

Serving it fast is a data-structure problem. A trie, or prefix tree, stores the
log so that every query sharing a prefix sits below the same node, and walking
down the tree one character at a time locates the prefix in time proportional
to its length. Returning the ten best completions below a node without
scanning the whole subtree requires storing, at each node, the highest score in
its subtree, so the search can expand the most promising branches first and
stop as soon as ten complete queries have been popped. Hsu and Ottaviano
compared compressed versions of these structures in 2013, since a full query
log held in memory is large.

Context changes what the most likely completion is. Bar-Yossef and Kraus used
the queries earlier in the same session: after a search for a band, the prefix
"t" is more likely to mean that band's tour than the most popular "t" query
overall. Shokouhi and Radinsky showed in 2012 that popularity is not static
either, and forecast each candidate's near-future frequency from the time series
of its past counts, so that a seasonal or periodic query is promoted before its
peak instead of after it. Shokouhi then learned personalised rankers in 2013,
with features drawn from the user's long-term search history, location, age and
gender, and found that the same prefix should complete differently for
different groups of users.

Prefixes never seen in the log need a generative model rather than a lookup.
Park and Chiba proposed in 2017 a character-level neural language model that
writes completions one character at a time, conditioned on the prefix, so that
it can propose plausible queries that no one has issued. Generation is slower
than looking up a trie and can produce strings that are fluent but never asked,
so practical systems use it as a fallback for the rare prefixes the log cannot
cover. Typing errors are a separate problem: a prefix containing a typo matches
nothing in an exact trie. Chaudhuri and Kaushik extended auto-completion to
tolerate errors in 2009, matching the prefix within a small edit distance
rather than exactly.

Evaluation borrows from document ranking with a twist. The usual measure is the
reciprocal rank of the query the user eventually submitted, computed at each
prefix length and averaged, so a system is credited for surfacing the final
query early and high. Saved keystrokes, the characters the user did not have to
type because they accepted a suggestion, measures effort directly. Neither
measure captures how people actually look at the list. Mitra and colleagues ran
an eye-tracking study in 2014 and found strong position bias, attention
concentrated on the top suggestions, and engagement that depended on where in
the word the user was, being more likely at word boundaries than in the middle
of a word. Logged acceptances are therefore biased feedback in the same way
logged clicks on result pages are.

Suggestions are also shown before anyone asks for them, which makes them a
publishing decision as well as a ranking one. A completion learned from the
log can repeat offensive, defamatory or misleading phrasings that enough
people typed, so deployed systems filter candidates against blocklists and
classifiers before display. The survey by Cai and de Rijke in 2016 collects the
field's methods and measures.
