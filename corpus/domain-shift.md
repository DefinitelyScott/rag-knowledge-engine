# Domain shift and zero-shot retrieval

Every reported retrieval number is a number about one collection. A model
trained on web search logs and measured on web search queries has been asked an
easy question about itself. Pointing that same model at a corpus of clinical
notes, legal filings, or internal engineering documentation is not a
deployment detail; it is a different experiment, and the published score does
not carry over to it. Zero-shot here means exactly that: the retriever sees the
new collection and its queries without having been trained on either.

BEIR, from Thakur, Reimers, Rücklé, Srivastava and Gurevych in 2021, was built
to make the size of that gap visible. It assembles eighteen publicly available
datasets spanning different tasks and subject areas and runs ten retrieval
systems over all of them without per-dataset training, reporting nDCG at 10.
The headline finding is uncomfortable for anyone shopping by leaderboard: BM25
is a robust baseline. Re-ranking and late-interaction architectures achieved
the best average zero-shot scores, but at a computational cost that the cheaper
families avoid, and the dense and sparse retrievers that are efficient enough
to run as a first stage frequently underperformed the lexical baseline they
were meant to replace.

The asymmetry has a structural cause rather than a tuning one. A lexical
scorer has no parameters fitted over the term space. Its document frequencies,
lengths and averages are all recomputed from whatever collection it is handed,
so an unfamiliar vocabulary is simply a different set of counts and the method
adapts by construction. A bi-encoder's weights are frozen when training ends.
A term it never saw is still mapped somewhere in the embedding space, and the
model has no way to signal that the placement is a guess. Confidence survives
the domain change even when accuracy does not.

The adaptation that works needs no relevance labels from the target domain,
only its documents. Generative Pseudo Labeling, from Wang, Thakur, Reimers and
Gurevych in 2022, generates plausible queries for passages in the new
collection, scores each generated pair with a cross-encoder to get a soft
relevance margin rather than a hard yes or no, and trains the retriever
against those margins. On six domain-specialised datasets it improved on an
out-of-the-box dense retriever by as much as 9.3 nDCG at 10. Document-side
expansion attacks the same mismatch from the index side and costs nothing at
query time.

The practical lesson is not that lexical retrieval wins. It is that the ranking
of methods is a property of the collection, so the ranking has to be
re-established on the collection that matters. That means building a gold set
against the actual corpus, however small and however tedious, before believing
any ordering inherited from a benchmark someone else ran.
