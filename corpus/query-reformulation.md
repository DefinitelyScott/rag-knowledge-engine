# Query reformulation in search sessions

Query reformulation is what a searcher does when the first query did not work:
they change it and try again. Studies of web search logs find that a large
fraction of queries are not the first of their session, and that the sequence
of reformulations is itself evidence about what the person wanted. A system
that treats each query as independent throws that evidence away.

Analyses of logs, notably by Huang and Efthimiadis, sort the changes into types.
A specialisation adds terms and narrows the request. A generalisation removes
terms and widens it. A parallel move replaces a term with a related one at the
same level of specificity, and a synonym substitution replaces it with a word
of the same meaning. A spelling correction repairs a typing error. Some
reformulations change the whole topic, which is how a session ends and another
begins. Telling these apart needs more than string overlap, so systems combine
edit distance, shared terms, the time gap between queries, and whether the user
clicked anything in between.

Session boundaries are the first problem. A fixed inactivity cutoff such as
thirty minutes is common and crude, since it splits long tasks and merges
unrelated quick ones. Better segmenters combine the cutoff with lexical and
semantic similarity between consecutive queries. A task segmentation goes
further and groups queries that serve one goal even if other queries are
interleaved, as happens when a person has several tabs open.

Reformulation data is useful in at least three ways. It supplies query
suggestions: the queries that most often follow a given one, weighted by how
often they ended the session with a satisfied click, are the ones worth offering.
It supplies training pairs for a rewriting model that maps a failing query to a
better one, which is the same machinery that spelling correction uses with a
broader notion of error. And it supplies an implicit label. A query that is
quickly reformulated with no click is a sign of dissatisfaction, called a good
abandonment when the answer was shown on the page and a bad abandonment when
it was not. Distinguishing the two is hard from the log alone.

In conversational systems the same problem returns in a sharper form. A
follow-up such as "and what about its memory use" cannot be understood without
the earlier turns, so a rewriter produces a standalone query by resolving
references and restoring dropped context before retrieval. That rewritten query
is a reformulation the user never typed. The risk is that the rewriter adds or
guesses a constraint the user did not intend, which narrows the results with no
visible sign, so evaluation compares the rewritten queries to human rewrites and
also measures retrieval with each.

Reformulation behaviour also confounds evaluation. A system that returns better
first-page results will see fewer reformulations, so reformulation rate is a
reasonable proxy for effort, but it falls for reasons unrelated to quality, such
as a change in interface that makes the next query harder to type.
