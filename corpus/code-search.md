# Code search

Code search is retrieval where the documents are source code, usually functions
or files, and the queries are either natural-language descriptions of what the
code does or fragments of code and identifiers. It differs from ordinary text
retrieval in three ways that matter for every design decision: the vocabulary
of the query and the vocabulary of the target barely overlap, the text has
internal structure that a bag of words discards, and two quite different
products hide under the same name.

The two products are exact search and semantic search. Exact search, the kind
behind a developer pressing a key to find every use of a function name, treats
code as text to be matched literally or by regular expression. The standard
index for it is a trigram index: every three-character sequence in the corpus
maps to the files that contain it, and a query or regular expression is first
reduced to the trigrams any match must contain. Those postings are
intersected to produce a short candidate list, and only the candidates are
scanned with the real pattern. Russ Cox described this design in his account of
how Google Code Search worked. It is an inverted index whose terms are
character sequences rather than words, and it is built for completeness: a
match is either there or it is not, and ranking is a minor concern.

Semantic search is the other product. The query is a sentence such as "read a
file into a string", and the system should return a function that does it
whether or not the function is called anything like that. This is the
vocabulary mismatch problem in a harsh form. A function is named
`slurp` or `readAllText`, its parameters are `p` and `enc`, and the words of the
query may appear nowhere in it. Comments and docstrings help when they exist,
but they are often missing, stale or terse.

Identifiers need preprocessing before a lexical index can use them. A name such
as `parseHttpResponseHeader` or `parse_http_response_header` is one token to a
whitespace tokenizer. Splitting on case changes, underscores and digits
produces `parse`, `http`, `response` and `header`, which can match words in a
query. Some systems index both the split parts and the original identifier,
since an exact identifier in a query is a very strong signal and a split part
is a weak one. Stemming and stop-word removal are applied with more caution
than in prose, because words such as `for`, `in` and `is` are keywords rather
than filler.

Learned approaches embed the query and the code into a shared vector space.
CodeSearchNet, a benchmark released by researchers at GitHub and Microsoft
Research in 2019, pairs about two million functions that have docstrings with
those docstrings across six programming languages, and adds a smaller set of
queries with relevance judgements collected from annotators, scored with
normalised discounted cumulative gain. CodeBERT, from Microsoft Research in
2020, pretrained a transformer on pairs of natural language and code so that a
single model reads both. The training pairs raise an evaluation worry: using
the first line of a docstring as a stand-in query is cheap, but docstrings are
written by the author who just wrote the function, so they share its
vocabulary and tend to be easier than the questions a person would actually
type. A model can score well on that proxy and still disappoint on real
queries.

Granularity is a design choice with no neutral answer. A whole file is too
coarse for a specific question and drowns the relevant function in unrelated
ones. A single function is usually the right unit for semantic search, but it
loses the class, module and imports that explain it. Chunking code by syntax
tree boundaries instead of by a fixed token window keeps a function whole,
which a sentence-based chunker built for prose cannot do. Retrieval-augmented
code assistants face the same trade-off when they decide what repository
context to place in a prompt.
