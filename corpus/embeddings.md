# Dense retrieval and text embeddings

A dense retriever represents text as a fixed-length vector of real numbers,
typically a few hundred to a few thousand dimensions, produced by a neural
encoder. Unlike a sparse TF-IDF vector, where each dimension is a vocabulary
term, no individual dimension of an embedding has an interpretable meaning.
What matters is that texts with similar meaning land close together under
cosine similarity or inner product.

Dense Passage Retrieval, described by Karpukhin and colleagues in 2020, made
the approach practical for open-domain question answering. It uses a bi-encoder:
one encoder for questions, one for passages, each producing a vector
independently. Because passages are encoded ahead of time and only the question
is encoded at query time, retrieval reduces to a nearest-neighbour lookup and
stays fast at scale. The model is trained with a contrastive objective that
pulls a question toward its correct passage and pushes it away from negatives,
including hard negatives mined from a lexical retriever.

The strength of dense retrieval is vocabulary independence. A question phrased
as "how do I stop my program crashing on startup" can match a passage about
"initialisation failures" with no shared content words. The corresponding
weakness is precision on rare literal strings: exact identifiers, part numbers,
names and version strings are often smoothed away, because the encoder was
trained to generalise rather than to memorise surface forms.

Dense retrieval also carries operational costs that lexical retrieval does not.
It needs a model at query time, embeddings must be regenerated whenever the
model changes, and quality depends on how close the target domain is to the
training distribution. An encoder trained on web text can underperform BM25 on
specialised technical corpora.

These trade-offs are the argument for hybrid systems. Lexical retrieval handles
exact terms, dense retrieval handles paraphrase, and a fusion step lets each
cover for the other rather than forcing a choice between them.
