# rag-knowledge-engine

A hybrid-retrieval RAG engine written against the Python standard library only.
No numpy, no faiss, no sentence-transformers, no LLM required. Every scoring
decision, from IDF to rank fusion to sentence selection, is plain readable
Python you can step through.

The point of the project is not to beat a vector database. It is to make the
mechanics of retrieval legible: what BM25 actually computes, why fusing two
ranked lists needs care, and how you tell whether any of it helped.

## Install and run

```bash
git clone https://github.com/DefinitelyScott/rag-knowledge-engine
cd rag-knowledge-engine

python -m ragkb.cli stats
python -m ragkb.cli search "how does reciprocal rank fusion work" -k 3
python -m ragkb.cli ask "what does the b parameter control in bm25"
```

`ask` uses the offline extractive answerer by default. Passing `--llm` switches
to the OpenAI answerer, which reads `OPENAI_API_KEY` and calls the chat
completions endpoint over `urllib`; it is optional and nothing else depends on
it.

## Architecture

```
question
   |
   +-- tokenize ------------------------------------+
   |                                                |
   v                                                v
BM25 ranked list                          TF-IDF cosine ranked list
   |                                                |
   +----------- reciprocal rank fusion -------------+
                        |
                        v
              optional MMR diversity re-rank
                        |
                        v
                 top-k chunks (with doc provenance)
                        |
                        v
        extractive answerer  |  OpenAI answerer   -> answer + citations
```

| Module | Responsibility |
| --- | --- |
| `ragkb/text.py` | tokenizer, sentence splitter, sentence-aware overlapping chunker |
| `ragkb/bm25.py` | Okapi BM25 with configurable `k1` and `b` |
| `ragkb/vector.py` | sparse TF-IDF vectors, L2-normalised, cosine similarity |
| `ragkb/retriever.py` | hybrid retriever, reciprocal rank fusion |
| `ragkb/rerank.py` | Maximal Marginal Relevance re-ranking, redundancy diagnostic |
| `ragkb/answerer.py` | extractive answerer (offline) and OpenAI answerer, both cite |
| `ragkb/engine.py` | corpus loading, chunking, indexing, question answering |
| `ragkb/metrics.py` | hit@k, recall@k, precision@k, MRR |
| `ragkb/cli.py` | `stats`, `search`, `ask`, `--mmr` |

`corpus/` holds the sample knowledge base (twelve documents on information
retrieval). `evals/gold.jsonl` holds forty-eight labelled questions with the
documents that answer them, split into an `easy` and a `hard` slice.

## Evaluation

```bash
python evals/evaluate.py
pytest -q
```

Current numbers on the bundled corpus and gold set (12 documents, 47 chunks,
48 questions, k=5, document-level relevance).

The gold set is split by difficulty. The `easy` slice is the original thirty
questions, whose wording shares vocabulary with the target document. The `hard`
slice is eighteen questions written to avoid the target document's own terms -
"if someone writes automobiles but the page only ever says cars" instead of
"vocabulary mismatch" - and eight of them are answered by two documents rather
than one.

**easy (30 questions)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| hybrid | 0.800 | 0.933 | 0.967 | 0.967 | 0.869 |
| bm25 | 0.800 | 0.933 | 0.967 | 0.967 | 0.868 |
| vector | 0.833 | 0.933 | 0.967 | 0.967 | 0.886 |

**hard (18 questions, 8 with two relevant documents)**

| method | hit@1 | hit@3 | hit@5 | recall@3 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.444 | 0.833 | 0.889 | 0.750 | 0.778 | 0.606 |
| bm25 | 0.444 | 0.778 | 0.889 | 0.694 | 0.778 | 0.596 |
| vector | 0.444 | 0.833 | 0.889 | 0.722 | 0.778 | 0.606 |

The easy slice was saturated and hid everything. Every question on it has
exactly one relevant document, which makes recall@k numerically identical to
hit@k - the column was decoration. hit@5 sat at 0.967 for all three methods, so
no retrieval change could move it in either direction.

The hard slice restores headroom: hit@1 falls from 0.800 to 0.444 and MRR from
0.869 to 0.606, and recall@k finally diverges from hit@k because a question
answered by two documents can be half-satisfied. Four questions land in exactly
that state at k=5, which is a failure mode the old set could not express at all.

Reported honestly: hybrid still does not dominate. It beats BM25 on the hard
slice at rank three (hit@3 0.833 vs 0.778, recall@3 0.750 vs 0.694) and ties
TF-IDF on MRR, so fusion buys a little ordering quality on paraphrased
questions and nothing at k=5. The corpus is still twelve documents with
distinctive per-document vocabulary; that is the remaining reason the two
retrievers agree too often, and it is a corpus problem, not a tuning problem.

### Diversity: MMR re-ranking

Accuracy metrics cannot see redundancy. A top-5 made of five paraphrases of one
passage scores exactly the same as five complementary ones, but it is worth far
less to an answerer. This corpus makes the problem concrete: the chunker
overlaps adjacent chunks on purpose, so neighbouring chunks are near-copies and
the unmodified top-5 spans only 3.5 of a possible 5 documents.

`ragkb/rerank.py` implements Maximal Marginal Relevance (Carbonell & Goldstein,
SIGIR 1998), selecting greedily on
`lambda * relevance - (1 - lambda) * max similarity to what is already chosen`.
Relevance is min-max normalised over the candidate pool so the subtraction is
meaningful for RRF scores near 1/60 and raw BM25 scores alike.

```bash
python -m ragkb.cli --mmr 0.7 search "how does BM25 saturate term frequency"
```

Sweeping lambda over all 48 questions at k=5 (`redundancy` is the mean
pairwise cosine within a result set; `distinct` is how many documents it spans):

| lambda | hit@1 | hit@5 | MRR | redundancy | distinct |
| --- | --- | --- | --- | --- | --- |
| off | 0.667 | 0.938 | 0.771 | 0.142 | 3.67 |
| 1.0 | 0.667 | 0.938 | 0.771 | 0.142 | 3.67 |
| 0.9 | 0.667 | 0.938 | 0.769 | 0.140 | 3.73 |
| 0.7 | 0.667 | 0.938 | 0.767 | 0.119 | 4.21 |
| 0.5 | 0.667 | 0.938 | 0.758 | 0.104 | 4.50 |
| 0.3 | 0.667 | 0.938 | 0.754 | 0.094 | 4.69 |

`lambda = 1.0` reproduces the unmodified ranking exactly, which is the cheapest
available correctness check on the implementation. At `lambda = 0.7` redundancy
falls 16% and the result set covers 15% more documents while hit@1 and hit@5 are
unchanged and MRR gives up 0.004. The harder questions did not change that
conclusion, which is mildly reassuring: the trade-off curve has the same shape
on a question set the retriever finds materially harder. Below that, accuracy starts paying for
diversity in a way this corpus does not justify, so re-ranking stays **off by
default** and the trade-off is a flag rather than a hidden constant.

### Faithfulness: does the answer stay inside the passages?

Retrieval metrics stop at the passage. They cannot see the failure users
actually notice: an answer that asserts something the retrieved text does not
support, or that cites a document which contributed nothing.
`ragkb/faithfulness.py` measures the last step, offline and without labels:

- **sentence grounding** - share of answer sentences found verbatim in a
  retrieved passage. The extractive answerer should score 1.000 by
  construction, so this is a regression guard, not a quality score.
- **token grounding** - share of content tokens present in the context. The
  lenient measure, and the one that still means something for a paraphrasing
  LLM answerer: it is what catches an invented name, number, or acronym.
- **citation precision** - share of cited documents that actually contributed a
  sentence. An answerer that cites everything it was handed fails here, which
  is the point: a citation list that is not selective is not evidence.
- **citation recall** - share of grounded sentences whose source is cited.

```bash
python evals/faithfulness_eval.py
```

| k | sentence grounding | token grounding | citation precision | citation recall | cited gold |
| --- | --- | --- | --- | --- | --- |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.812 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.792 |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.792 |

`cited gold` is the only column that uses the labels: the share of answers
citing a document the gold set marks relevant. It fell from 0.900 to 0.792 when
the hard questions were added, and that is the correct direction: retrieval
misses more of them, so more answers are built from the wrong passages. The
grounding and citation columns stayed at 1.000 throughout, which is the
distinction the measure exists to draw - the answerer is not less faithful, it
is being handed worse evidence.

The measure earned its place on the first run by failing. Sentence grounding
came back at 0.967 rather than 1.000, and the cause was real: markdown headings
carry no terminal punctuation, so the sentence splitter glued each heading to
the sentence after it. That pseudo-sentence existed nowhere in the corpus, and
it was being selected into answers, where it could not be traced back to any
passage. Headings are now hard sentence boundaries, chunks join their sentences
on newlines so a chunk round-trips through the splitter, and the extractive
answerer skips heading-only sentences - a heading names a section, it does not
assert anything. Retrieval metrics are unchanged; sentence grounding and
citation precision are now 1.000.

## Design notes

- **Sentence-aware chunking with overlap.** Chunks never end mid-sentence, and
  each chunk repeats the tail of the previous one so ideas that straddle a
  boundary stay retrievable from both sides.
- **Rank fusion rather than score blending.** BM25 scores and cosine
  similarities have no common scale, and their ranges move per query. RRF only
  needs the ordering.
- **An extractive answerer that cannot hallucinate.** It returns sentences
  copied verbatim from retrieved chunks, so the failure mode is an unhelpful
  answer rather than an invented one. That isolates retrieval quality from
  generation quality during evaluation.
- **Ablations by default.** `evals/evaluate.py` always prints BM25, vector and
  hybrid side by side, because a hybrid that does not beat its components is
  not earning its complexity.
- **Diversity measured, not assumed.** MMR ships with a redundancy read-out and
  a lambda sweep rather than a tuned default, because "more diverse" is only
  worth having if you can show what it cost.

## Tests

```bash
pytest -q      # 95 tests
```

The suite covers tokenisation and chunk boundaries, BM25 saturation and length
normalisation, TF-IDF normalisation, RRF scoring shape, every metric, the CLI,
MMR selection order and its redundancy invariants, answer grounding and
citation accuracy, gold-set integrity, and an end-to-end pass over the real
corpus and gold set.
