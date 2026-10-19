# Stemming and lemmatisation: matching the word, not the spelling

A searcher who types "running shoes" would usually be happy with a page about
"a shoe for runners", but an index that compares exact strings sees four unrelated
words. Morphological normalisation closes that gap by reducing inflected and
derived forms of a word to a shared representative before indexing and again
before querying. It raises recall, because more documents share a term with the
query, and it pays for that in precision, because some of the new matches are
wrong. Whether the trade is worth making is an empirical question that depends
heavily on the language.

The two families differ in how much they know. A stemmer is a set of rewrite
rules that chops suffixes off a string with no dictionary and no notion of
meaning. Lovins published the first widely used one in 1968: a single pass that
removes the longest matching ending from a list of a few hundred, followed by
recoding rules that repair the stem. Porter's 1980 algorithm works in five steps
of rules, each guarded by a condition on the *measure* of the remaining stem,
roughly the number of vowel-consonant groups it contains. The guard is what stops
the rules from stripping a word down to nothing. Porter later released Snowball,
a small language for writing stemmers, and with it versions for many languages.
A lemmatiser instead looks each word up in a dictionary, often using its part of
speech, and returns the dictionary headword, so that "better" can map to "good"
and "saw" to "see" when used as a verb. It is slower and needs a lexicon, but its
output is a real word.

Because a stemmer has no vocabulary to check against, it makes two characteristic
errors. *Overstemming* merges words that should stay apart: "university" and
"universe" can both reduce to the same stem, so a query about one retrieves the
other. *Understemming* fails to merge words that belong together: "data" and
"datum", or "alumnus" and "alumni", keep different stems because the change is
inside the word, not at its end. Making a stemmer more aggressive trades one
error for the other. Krovetz's 1993 stemmer sits between the families. It uses a
dictionary to conflate only inflectional variants such as plurals and tenses,
leaving derivational ones alone, and it returns real words, which is gentler and
easier to inspect.

The evidence that stemming helps is weaker than its popularity suggests. Harman
reported in 1991 that stemming did not reliably improve English retrieval, and
Hull's 1996 case study found average gains that were small, with individual
queries getting better and worse in both directions: a query is helped when
variants of its terms appear in the relevant documents and hurt when the merge
joins a query term to an unrelated sense. The picture changes for languages with
rich morphology. Popovic and Willett showed in 1992 that stemming mattered far
more for Slovene than it does for English, and the same reasoning applies to
Finnish, Turkish and Arabic, where one word can take dozens of surface forms.
German adds compound words, which need splitting rather than suffix removal.

Where the normalisation happens is a design decision. Stemming at index time
shrinks the vocabulary and the index, but it bakes the stemmer into the stored
data, so changing the rules means rebuilding the index. Leaving the index
unstemmed and expanding each query term into its known variants at query time
keeps the index reusable and lets rules be tuned, at the cost of longer queries.
Neighbouring steps such as case folding, removing accents and Unicode
normalisation have the same flavour. A language-independent alternative is to
index overlapping character n-grams, which match across inflections without any
language knowledge and are one reason n-gram indexes cope well with
misspellings and unfamiliar languages.
