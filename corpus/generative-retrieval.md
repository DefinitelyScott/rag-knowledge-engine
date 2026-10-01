# Generative retrieval: a model that emits document identifiers

Every other retriever in this corpus keeps an explicit index, inverted lists or
a table of vectors, and scores documents against it at query time. Generative
retrieval removes the external index. A sequence-to-sequence model is trained
to read a query and generate, token by token, the identifier of a relevant
document. The corpus is stored in the model's parameters, and retrieval is
decoding.

The Differentiable Search Index, from Tay and colleagues at Google in 2022,
established the recipe. Training mixes two tasks: indexing, where the model
reads a document's text and learns to output its identifier, and retrieval,
where it reads a query and learns to output the identifier of the document
that answers it. A ranked list comes from beam search, with each beam a
candidate identifier.

How documents are named matters as much as the model. DSI compared three
schemes. Atomic identifiers give each document its own output token, which is
simple but makes the output layer grow with the corpus. Naive string
identifiers spell an arbitrary number out as tokens. Semantically structured
identifiers run hierarchical k-means over document embeddings: the first digit
names a top-level cluster, the next names a cluster within it, and so on, so
documents about similar things share prefixes and every decoding step narrows
the search to a region of the collection. The structured scheme was the
strongest in most of the original experiments, though not all. Decoding is usually constrained with a prefix tree of
the identifiers that exist, so the model can never emit one that does not.

Other systems generate text that the corpus actually contains rather than an
invented code. GENRE, from De Cao and colleagues in 2021, retrieves entities by
generating their Wikipedia titles. SEAL, from Bevilacqua and colleagues in
2022, generates n-grams that occur in passages and uses an FM-index, a
compressed full-text index, both to keep decoding on strings that exist and to
map the generated n-grams back to the passages that hold them.

The training data turned out to be the bottleneck. A document is read once
during indexing, but queries are short and phrased differently, so a model
that has only seen document text generalises poorly to questions. Later work,
including DSI-QG and the Neural Corpus Indexer, both from 2022, trained on
synthetic queries produced by a query-generation model for each document, the
same document-to-query trick used for lexical document expansion, and reported
large gains from it.

Two weaknesses limit the approach. The first is scale. Pradeep and colleagues
studied generative retrieval on the full MS MARCO passage collection of about
8.8 million passages in 2023, and found that methods which looked strong on
corpora of a hundred thousand documents lost much of their advantage, with
synthetic queries the single most important ingredient. The second is
updating. An inverted index adds a document by appending postings; a
generative retriever has to be trained further, and training on new documents
makes the model forget old ones. DSI++, from Mehta and colleagues in 2023,
measured that forgetting and proposed rehearsal of earlier material to reduce
it.
