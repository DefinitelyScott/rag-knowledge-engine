# Document length normalisation

Every lexical ranking function has to decide what a long document's extra words
are worth. A long document contains more occurrences of most terms and more
distinct terms, so without a correction it matches more queries and scores
higher on the ones it matches. Whether that advantage is deserved depends on why
the document is long.

Robertson and Walker named the two explanations in 1994. Under the verbosity
hypothesis, a long document says the same thing as a short one in more words, so
its higher term counts are inflation and should be divided away. Under the scope
hypothesis, a long document covers more ground, so it really is more likely to
contain what a reader is looking for and some of its advantage is earned. Real
collections mix both kinds, which is why the useful normalisations are partial:
they discount length, but not all the way.

Cosine normalisation is the full discount. Dividing a TF-IDF vector by its own
Euclidean length makes every document a unit vector, so a long document gains
nothing from being long. Singhal, Buckley and Mitra showed in 1996 that this was
not neutral. Plotting, on TREC data, the probability that a document of a given
length was relevant against the probability that it was retrieved, they found
the two curves crossed: short documents were retrieved more often than their
share of relevant documents justified, and long ones less often.

Their correction, pivoted document length normalisation, keeps the old
normalisation factor but tilts it. The point where the two curves cross is the
pivot. The new factor is a weighted mix of the pivot and the old factor, with a
slope below one. A document whose old factor is above the pivot is divided by
less than before and one below the pivot by more, which lifts long documents
and lowers short ones until retrieval tracks relevance more closely. The slope
is tuned on judged queries. The same paper also normalised by
the number of distinct terms rather than the vector length, since that count
grows less erratically with document size.

BM25 builds the same idea into its term-frequency component. The parameter b
interpolates between no length correction at b = 0 and full correction
proportional to the ratio of document length to average length at b = 1. A
value near 0.75 is the usual default, which encodes the partial discount the
scope hypothesis argues for. It is a pivot in another form: a document of exactly
average length is untouched, and others are scaled around it.

Partial correction has its own failure at the extremes. Lv and Zhai pointed out
in 2011 that for a very long document the normalised term frequency can be
driven so close to zero that matching a query term adds almost nothing, and the
document barely outscores one that does not contain the term at all. Their
remedy, often called BM25+, adds a small constant to the term-frequency
component of every matched term, putting a floor under the reward for a match.

Length is also partly a property of the system rather than the text. A system
that splits documents into passages of roughly equal size before indexing makes
most units close to the average length, so the choice of b matters less than it
does over whole documents of wildly different sizes. The question then moves to
how passage scores are combined back into a document score.
