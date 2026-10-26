# Entity linking: from mentions to knowledge-base entries

A query or a document says "Jaguar" or "Washington" and means one particular
thing. Entity linking is the task of finding such mentions in text and mapping
each to an entry in a knowledge base, usually Wikipedia or a graph built from
it. Where plain retrieval matches strings, linking matches meanings, and the
output (a set of identifiers) can be indexed, compared and joined with
structured facts.

The pipeline has three stages. Mention detection finds the spans that could
name an entity. Candidate generation proposes, for each span, the entries it
could refer to, typically by looking the surface form up in a dictionary built
from Wikipedia titles, redirects, disambiguation pages and the anchor text of
links. Disambiguation then picks one candidate, or none. Mentions whose entity
is absent from the knowledge base are labelled NIL, and deciding when to answer
NIL is its own source of errors.

A strong baseline for disambiguation is commonness: how often an anchor text
points to each target in Wikipedia. Always choosing the commonest sense of a
mention is hard to beat on simple text, and loses only on the minority of
mentions that use a rarer sense. Milne and Witten's
2008 "Learning to link with Wikipedia" combined commonness with a relatedness
measure computed from the overlap of the pages that link to two articles, and
with the context supplied by unambiguous mentions nearby. The idea that
neighbouring entities should be coherent with each other, so that "Jordan" next
to "Chicago Bulls" resolves differently from "Jordan" next to "Amman", is
called collective disambiguation, and it turns the choice into a joint
optimisation over all mentions in the passage.

Short text is the hard case, and search queries are short. TagMe, by Ferragina
and Scaiella in 2010, was designed for it: each candidate for a mention is
scored by votes from the candidates of the other mentions, weighted by their
relatedness, and low-probability links are pruned. It is fast enough to run on
queries and snippets. More recent systems use neural encoders. BLINK, from Wu
and colleagues in 2020, retrieves candidates with a bi-encoder that embeds the
mention in context and the entity description in the same space, then reranks
the shortlist with a cross-encoder, the same two-stage shape used for passage
retrieval. Because it works from entity descriptions rather than a learned
list of known entities, it can link to entities unseen in training.

For retrieval, linked entities give several tools. A query's entities can be
used to expand it with the names and aliases of the entry, or with terms from
its description. Dalton, Dietz and Allan reported in 2014 that expanding with
features drawn from knowledge-base links improved ad hoc ranking over a
strong baseline. Documents can be indexed with their entity identifiers beside
their words, so "Apple" the company and "apple" the fruit stop colliding. And
the same machinery supports entity-oriented search, where the result is the
entity itself instead of a page about it, as in the work Balog summarised.

The method's failure modes are predictable. Errors compound: a wrong link
injects a confidently wrong expansion, which can hurt a query that plain
matching would have handled. Coverage is uneven, since Wikipedia is rich on
famous people and thin on local businesses, new products and non-English
topics. Entity meanings drift, so a dictionary built from last year's anchors
misses an entity that appeared this month. And evaluation depends on the
knowledge-base snapshot used for the gold labels, so published numbers across
datasets are not directly comparable.
