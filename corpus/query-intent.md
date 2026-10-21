# Query intent: what the searcher is actually trying to do

Two people can type the same three words and want different things, and a
retrieval system that treats every query as the same kind of request will serve
some of them badly. Query intent is the study of why a query was issued, and
the practical question is whether a system can infer it early enough to change
what it returns.

Broder's 2002 taxonomy of web search is the common starting point. A
*navigational* query aims to reach one particular site the searcher already has
in mind, such as a company's name, and exactly one result is right. An
*informational* query wants to learn something that may be spread over many
pages, so no single document settles it. A *transactional* query wants to do
something on the web, such as download a file or buy a product. Rose and
Levinson refined this in 2004 into a hierarchy of goals, splitting
informational queries into directed, undirected, advice, locate and list
requests, and replacing transactional with a broader class of resource-seeking
goals. Neither taxonomy is a clean partition: many real queries mix intents, and
labelling them by hand is itself noisy.

The classes imply different definitions of success. For a navigational query
only the rank of the one right page matters, so hit@1 or reciprocal rank is the
natural measure. For an informational query that needs several pages, recall and
nDCG matter more, and a list full of near-duplicates wastes the slots. Ambiguous
queries make the problem sharper: "jaguar" may mean an animal, a car or an
operating system, and a ranking that commits to the most popular sense fails
everyone who meant another. This is the motivation for result diversification,
which hedges across senses instead of betting on one.

Intent has to be inferred from signals, since searchers rarely state it. Words
inside the query help: "buy", "download" and "price" suggest a transactional
goal, while question words suggest an informational one. Behaviour helps more.
Lee, Liu and Cho showed in 2005 that navigational queries tend to have
concentrated click distributions, with most users choosing the same result,
while informational queries spread clicks over many results, so the entropy of
past clicks separates them. A related signal is whether the query string matches
the anchor text of links pointing at a single page. Supervised classifiers can
combine such features, and the 2005 KDD Cup query-categorisation task, which
asked entrants to label hundreds of thousands of queries with topic categories
from only a small labelled sample, showed how hard the labelling is when data
is scarce.

Intent prediction is useful only if the system acts on it. A navigational
prediction can promote an exact title match and show one confident answer. An
ambiguous prediction can switch on diversification. In retrieval-augmented
generation the same idea appears as routing: deciding whether a question needs
retrieval at all, which collection to search, or whether it is a conversational
follow-up that must first be rewritten. A wrong intent label can be worse than
none, because it applies a specialised strategy to the wrong query, so the
classifier's confidence should gate the action.
