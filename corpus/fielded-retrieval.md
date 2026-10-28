# Fielded retrieval and BM25F

Most real documents are not a single block of text. A web page has a title,
headings, body text and the anchor text of links pointing at it. A product has
a name, a description and a list of attributes. A scientific paper has an
abstract, a body and a reference list. A word in the title is usually stronger
evidence of what a document is about than the same word buried in the body, and
a retrieval model that flattens everything into one bag of words throws that
signal away.

The obvious fix is to score each field separately with BM25 and add the scores
together with a weight per field. It works, and it has a subtle flaw. BM25
saturates term frequency: the tenth occurrence of a word adds far less than the
first. Scoring fields independently applies that saturation once per field, so a
term that appears once in the title and once in the body is rewarded twice, as
though it were two separate pieces of evidence. A term repeated across many
fields can therefore outscore a document that is simply more focused.

Robertson, Zaragoza and Taylor proposed a different construction in 2004, now
called BM25F. Instead of combining the scores of the fields, it combines their
term frequencies first. Each field contributes the count of the term in that
field, multiplied by a field weight, after normalising by how long that field is
compared with the average length of the same field across the collection. The
weighted, normalised counts are summed into one pseudo term frequency, and the
usual saturation function is applied exactly once. The document is then treated
as if the title had simply been repeated several times in the body, with the
repetition count set by the weight.

Length normalisation is done per field rather than per document. A title of
five words is normally compared with an average title, not with an average
document, so the b parameter can differ between fields. A title is short by
design, which means a strong normalisation would punish ordinary titles for
being long or short by a word or two, and a small b is usual there. A long body
field behaves like plain BM25 and wants a larger b. The inverse document
frequency of a term is still computed once for the whole document, not per
field.

The parameters multiply. With k1 and the saturation, a weight and a b for every
field, a model with four fields has about nine settings to choose. Authors tune
them on a labelled set, by grid search for a few fields or by a learning-to-rank
method when there are many. This inherits a problem familiar from the rest of
tuning work: the weights that win on one collection and one query set are
partly fitted to its quirks, so they need to be checked on questions that were
not used to choose them.

Anchor text is the field that made this practical on the web. The words in the
links pointing at a page are written by other people and often describe the page
better than the page describes itself, particularly for navigational queries
that look for one particular site. Craswell, Hawking and Robertson showed in
2001 that anchor text was a strong signal for site finding. Fields also have to
cope with absence: many documents have no title or no anchors, and an empty
field should contribute nothing, not a division by a zero length.

Fielded scoring does nothing when a collection has no useful structure. A
corpus of plain paragraphs has a single field, and BM25F reduces to ordinary
BM25. The value of the model scales with how much more reliable the short,
curated fields are than the free text around them.
