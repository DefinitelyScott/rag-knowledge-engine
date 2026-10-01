# Spelling correction: repairing the query before it reaches the index

A lexical index can only match the strings it was given. A query term that is
misspelled matches nothing, or worse, matches a rare valid word that happens to
sit one keystroke away. Short queries make this expensive: when a user types
three words and one of them is wrong, a third of the evidence is gone. Spelling
correction tries to recover the intended word before retrieval runs, and it is
one of the oldest pieces of query processing.

Most errors are small. Damerau's 1964 study of misspellings found that roughly
eighty percent were a single insertion, deletion, substitution, or transposition
of two adjacent letters. Levenshtein's edit distance counts the minimum number
of insertions, deletions and substitutions needed to turn one string into
another; the Damerau-Levenshtein variant also counts an adjacent transposition
as one operation, which matches the way people actually mistype. Both are
computed with a dynamic-programming table whose size is the product of the two
string lengths.

Comparing a typed word against every word in a dictionary is too slow, so
correctors first generate a short list of candidates. One approach enumerates
every string within one or two edits of the typed word and keeps the ones that
exist in the vocabulary. Another, described in the Manning, Raghavan and
Schütze textbook, builds a k-gram index over the vocabulary: each word is
broken into overlapping character bigrams or trigrams, the typed word's k-grams
are looked up, and vocabulary words that share enough of them, measured with
the Jaccard coefficient, are kept as candidates for an exact edit-distance
check. Phonetic keys such as Soundex, patented by Russell and Odell in 1918,
map words that sound alike to the same code and catch errors that edit
distance scores as large.

Choosing among the candidates is a ranking problem. Kernighan, Church and Gale
framed it in 1990 as a noisy channel: the user meant word c, the channel
corrupted it into the observed word w, and the best correction is the c that
maximises the prior P(c) times the error model P(w|c). The prior comes from how
often c occurs; the error model came from confusion matrices counting which
letters are substituted for which in real typing errors. Brill and Moore
improved the error model in 2000 by allowing edits over substrings, so that a
learned rule such as "ph" becoming "f" costs one operation instead of two.

Isolated-word correction cannot catch real-word errors, where the typo is
itself a valid word, such as "form" typed for "from". Those need context.
Mays, Damerau and Mercer used a trigram language model in 1991 to pick the
candidate that best fits the surrounding words. For search, Cucerzan and Brill
showed in 2004 that query logs are a better source than a dictionary: names,
products and new terms appear in logs long before they appear in any word list,
and a misspelled query can be corrected iteratively, one step at a time, toward
a frequently issued neighbour.

Search engines rarely replace a query silently. When the corrector is confident
the engine shows results for the corrected query and offers the original;
otherwise it runs the query as typed and suggests the correction. A
character n-gram retriever is the other way to survive typos: instead of
repairing the query, it scores overlapping fragments, so a single wrong letter
removes only the few fragments that contain it and the rest still match.
