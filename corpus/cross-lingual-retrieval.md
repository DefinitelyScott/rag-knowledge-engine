# Cross-lingual retrieval: when the question and the documents use different languages

In cross-language information retrieval the query is written in one language
and the documents are in another. It is the limiting case of vocabulary
mismatch. A monolingual query and a relevant document usually share at least
some words; a German question and an English answer may share none, so a
purely lexical scorer such as BM25 gives the relevant document a score of
exactly zero. Proper names, numbers and borrowed technical terms are the
exception, and they explain why character-level matching recovers some
cross-lingual matches between related languages that word matching misses.

There are three ways to put the two sides into a common language. The query
can be translated into the language of the documents, the documents can be
translated into the language of the query, or both can be mapped into a third
representation that neither language owns. Query translation is cheap, since
queries are short and only one needs translating at search time, but short
queries give a translator very little context to choose the right sense of a
word. Document translation has the whole text as context and can be done once
at indexing time, but it costs a translation of the entire collection for every
query language supported. McCarley compared the two in 1999 and found that
combining both directions worked better than either alone.

Early systems translated queries with bilingual dictionaries, which exposed
three problems. A word with many dictionary translations floods the translated
query with alternatives, most of them wrong for the context. Phrases translated
word by word lose their meaning. And words missing from the dictionary, most
often names and new terms, pass through untranslated or are dropped.
Ballesteros and Croft showed in 1997 that translating phrases as units and
applying pseudo-relevance feedback both before and after translation recovered
much of the loss. Pirkola's structured queries, from 1998, addressed the flood
of alternatives: all the translations of one source word are grouped and treated
as synonyms of a single term, so their frequencies are pooled, rather than
letting a word with eight translations count eight times as much as a word with
one. Darwish and Oard's probabilistic structured queries later weighted each
translation by a probability learned from aligned parallel text instead of
treating all of them equally. Names that cross a writing system, such as from
Arabic or Cyrillic script into Latin script, need transliteration rather than
translation, and the same name often has several accepted spellings.

Shared evaluation drove the field. TREC ran a cross-language track in the late
1990s, the Cross-Language Evaluation Forum, CLEF, began in 2000 with a focus on
European languages, NTCIR in Japan covered Chinese, Japanese and Korean, and
FIRE later did the same for languages of South Asia. A recurring finding across
those campaigns was that good cross-language runs could approach, but rarely
match, a monolingual run on the same topics.

Neural models changed the third option, a shared representation, from a
research idea into a default. Multilingual encoders such as multilingual BERT
and XLM-R are pretrained on text in around a hundred languages at once, and
sentence encoders such as LaBSE are trained on translation pairs so that a
sentence and its translation land close together in the embedding space. A
dense retriever built on such an encoder can be fine-tuned on English relevance
data and then applied to queries in other languages it never saw labels for,
which is called zero-shot cross-lingual transfer. It works, but it loses
accuracy relative to having labels in the target language, so one common
remedy is to machine-translate an English training set; mMARCO is a translated
version of MS MARCO built for that purpose. Mr. TyDi and its successor MIRACL
provide relevance judgments written natively in many languages, so that
systems can be measured on queries people actually wrote in those languages
rather than on translations.
