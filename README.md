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
python -m ragkb.cli --expand 5 search "what breaks when a query and a page use different words"

# index once, query many times without re-fitting
python -m ragkb.cli index --out index.json
python -m ragkb.cli --index index.json ask "what does the b parameter control in bm25"
```

`ask` uses the offline extractive answerer by default. Passing `--llm` switches
to the OpenAI answerer, which reads `OPENAI_API_KEY` and calls the chat
completions endpoint over `urllib`; it is optional and nothing else depends on
it.

## Architecture

```
question
   |
   +-- tokenize --> optional PRF expansion ---------+
   |    (borrow terms from a first retrieval pass)  |
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
| `ragkb/expansion.py` | pseudo-relevance feedback query expansion (RM3-style) |
| `ragkb/rerank.py` | Maximal Marginal Relevance re-ranking, redundancy diagnostic |
| `ragkb/answerer.py` | extractive answerer (offline) and OpenAI answerer, both cite |
| `ragkb/engine.py` | corpus loading, chunking, indexing, question answering |
| `ragkb/store.py` | index persistence: save/load with staleness fingerprinting |
| `ragkb/metrics.py` | hit@k, recall@k, precision@k, nDCG@k, MRR |
| `ragkb/cli.py` | `stats`, `search`, `ask`, `index`, `--mmr`, `--expand`, `--index` |

`corpus/` holds the sample knowledge base (fourteen documents on information
retrieval). `evals/gold.jsonl` holds fifty-six labelled questions with the
documents that answer them, split into an `easy` and a `hard` slice.

## Evaluation

```bash
python evals/evaluate.py
pytest -q
```

Current numbers on the bundled corpus and gold set (14 documents, 57 chunks,
56 questions, k=5, document-level relevance).

The gold set is split by difficulty. The `easy` slice is thirty-four
questions whose wording shares vocabulary with the target document. The `hard`
slice is twenty-two questions written to avoid the target document's own terms -
"if someone writes automobiles but the page only ever says cars" instead of
"vocabulary mismatch" - and ten of them are answered by two documents rather
than one.

**easy (34 questions)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.824 | 0.971 | 0.971 | 0.971 | 0.905 | 0.882 |
| bm25 | 0.853 | 0.941 | 0.971 | 0.971 | 0.917 | 0.900 |
| vector | 0.824 | 0.912 | 0.971 | 0.971 | 0.897 | 0.873 |

**hard (22 questions, 10 with two relevant documents)**

| method | hit@1 | hit@3 | hit@5 | recall@3 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.500 | 0.864 | 0.909 | 0.727 | 0.773 | 0.636 | 0.653 |
| bm25 | 0.500 | 0.818 | 0.864 | 0.705 | 0.727 | 0.621 | 0.645 |
| vector | 0.500 | 0.818 | 0.909 | 0.682 | 0.773 | 0.635 | 0.659 |

The easy slice was saturated and hid everything. Every question on it has
exactly one relevant document, which makes recall@k numerically identical to
hit@k - the column was decoration. hit@5 sits at 0.971 for all three methods, so
no retrieval change can move it in either direction.

The hard slice restores headroom: hit@1 falls from 0.824 to 0.500 and MRR from
0.882 to 0.653, and recall@k finally diverges from hit@k because a question
answered by two documents can be half-satisfied. Six questions land in exactly
that state at k=5, which is a failure mode the old set could not express at all.

nDCG@5 is the one column that rewards ranking the *second* relevant document
high: MRR stops looking after the first hit, and recall@5 does not care whether
that second document sits at rank 2 or rank 5. Repeated chunks of one document
are credited once, so a top-5 stuffed with paraphrases of a single passage
cannot outscore one that ranks two distinct relevant documents. On the hard
slice this is still the only metric where hybrid edges *both* components
(0.636 vs 0.635 vector, 0.621 BM25) - the margin over BM25 is real, the margin
over TF-IDF is a rounding hair - because the fusion's edge is in how it orders
the full relevant set, which MRR ties could not show.

Reported honestly: hybrid still does not dominate, and the corpus was grown to
stop letting it off easy. An earlier version of this README diagnosed the
retrievers' constant agreement as "a corpus problem, not a tuning problem";
the two newest documents were added to fix exactly that. `late-interaction.md`
sits directly beside `reranking.md` (both discuss cross-encoders and candidate
pools) and `vector-compression.md` beside `ann-search.md` (both discuss product
quantisation), so several questions now have a lexically plausible wrong
neighbour. The methods respond by disagreeing more: BM25 takes the easy slice
outright at rank one (hit@1 0.853, MRR 0.900), TF-IDF takes hard-slice MRR by
0.006, and hybrid answers on the hard slice at rank three (hit@3 0.864 against
0.818 for both components) and at hit@5/recall@5, where BM25 alone drops a
question the fusion recovers. Fusion buys ordering quality on paraphrased
questions and pays a little at rank one on literal ones.

### Query expansion: borrowing the words the question did not use

Neither retriever can match a term the query never contains, and fusing two
retrievers that both missed the same synonym changes nothing - which is why the
hard slice hurts hybrid and BM25 by almost exactly the same amount. Pseudo-
relevance feedback is the classical fix that needs no embedding model: assume
the top few results of a first pass are relevant, read the terms they actually
use, and retrieve again with those terms added.

`ragkb/expansion.py` implements the RM3 form (Lavrenko & Croft, SIGIR 2001):
`alpha * P(t|query) + (1 - alpha) * P(t|feedback)`. Candidate terms are scored
by their frequency in the feedback set multiplied by IDF, so a term has to be
frequent *there specifically* rather than frequent everywhere. Weighting a query
by repeating tokens is exactly right for BM25 and only approximately right for
TF-IDF, so both indexes gained a real weighted-query entry point
(`search_weighted`) instead of that trick.

```bash
python -m ragkb.cli --expand 5 search "if someone writes automobiles but the page only says cars"
# expanded with: b (0.045), scores (0.041), probabilistic (0.039), robertson (0.039), ...
```

Sweeping terms and `alpha` with five feedback documents, at k=5:

**easy (34 questions)**

| terms | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.824 | 0.971 | 0.971 | 0.971 | 0.882 |
| 5, a=1.0 | 0.824 | 0.971 | 0.971 | 0.971 | 0.882 |
| 3, a=0.9 | 0.735 | 0.941 | 0.971 | 0.971 | 0.831 |
| 5, a=0.8 | 0.706 | 0.941 | 0.971 | 0.971 | 0.826 |
| 5, a=0.7 | 0.618 | 0.941 | 1.000 | 1.000 | 0.783 |

**hard (22 questions, 10 with two relevant documents)**

| terms | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.500 | 0.864 | 0.909 | 0.773 | 0.653 |
| 5, a=1.0 | 0.500 | 0.864 | 0.909 | 0.773 | 0.653 |
| 3, a=0.9 | 0.500 | 0.818 | 0.909 | 0.773 | 0.659 |
| 5, a=0.8 | 0.409 | 0.864 | 0.955 | 0.818 | 0.627 |
| 5, a=0.7 | 0.318 | 0.864 | 0.955 | 0.818 | 0.573 |

`alpha = 1.0` puts all the weight on the original query, so nothing is borrowed
and the ranking has to be identical to the unexpanded one. It is, on every one
of the 56 questions - that row and the test behind it are the correctness check
on this whole path, the same job `lambda = 1.0` does for MMR.

The split is the result, and the grown corpus sharpened it. On the hard slice,
five borrowed terms at `alpha = 0.8` still buy reach - recall@5 rises from
0.773 to 0.818 and hit@5 from 0.909 to 0.955, and h18 ("why can what a system
knows be changed without any training at all?") still goes from a complete miss
at k=5 to a hit, because the feedback pass borrowed *grounding*, *faithfulness*
and *relevant* from passages the plain question could not reach. But rank one
now pays for it: hit@1 falls from 0.500 to 0.409 and MRR from 0.653 to 0.627,
where on the smaller corpus expansion used to gain MRR here. Among the
questions it newly loses at rank one is h22, one of the adjacent-topic
additions - with a lexically similar neighbour document in the corpus, borrowed
terms are likelier to pull toward the neighbour. On the easy slice the same
setting costs four questions at rank one and 0.056 of MRR while moving recall
not at all - those questions already shared vocabulary with their target, so
every borrowed term is noise.

That is the textbook pseudo-relevance feedback trade-off, and this corpus
reproduces it cleanly: expansion buys reach on paraphrased questions and pays
for it in early precision on literal ones. It is **off by default** and exposed
as `--expand`, because a mixed question set is a net loss on MRR and the right
setting depends on whether the caller wants a top-1 answer or a k-passage
context. Two cautions on reading the tables: the hard slice is 22 questions, so
a 0.045 step is one question; and the feedback documents come from the same
retriever being evaluated, so when the first pass is wrong, expansion
confidently makes it wronger.

### Diversity: MMR re-ranking

Accuracy metrics cannot see redundancy. A top-5 made of five paraphrases of one
passage scores exactly the same as five complementary ones, but it is worth far
less to an answerer. This corpus makes the problem concrete: the chunker
overlaps adjacent chunks on purpose, so neighbouring chunks are near-copies and
the unmodified top-5 spans only 3.7 of a possible 5 documents.

`ragkb/rerank.py` implements Maximal Marginal Relevance (Carbonell & Goldstein,
SIGIR 1998), selecting greedily on
`lambda * relevance - (1 - lambda) * max similarity to what is already chosen`.
Relevance is min-max normalised over the candidate pool so the subtraction is
meaningful for RRF scores near 1/60 and raw BM25 scores alike.

```bash
python -m ragkb.cli --mmr 0.7 search "how does BM25 saturate term frequency"
```

Sweeping lambda over all 56 questions at k=5 (`redundancy` is the mean
pairwise cosine within a result set; `distinct` is how many documents it spans):

| lambda | hit@1 | hit@5 | MRR | redundancy | distinct |
| --- | --- | --- | --- | --- | --- |
| off | 0.696 | 0.946 | 0.792 | 0.140 | 3.70 |
| 1.0 | 0.696 | 0.946 | 0.792 | 0.140 | 3.70 |
| 0.9 | 0.696 | 0.946 | 0.788 | 0.138 | 3.73 |
| 0.7 | 0.696 | 0.929 | 0.783 | 0.116 | 4.14 |
| 0.5 | 0.696 | 0.946 | 0.787 | 0.105 | 4.38 |
| 0.3 | 0.696 | 0.946 | 0.778 | 0.093 | 4.63 |

`lambda = 1.0` reproduces the unmodified ranking exactly, which is the cheapest
available correctness check on the implementation. Growing the corpus moved the
sweet spot. `lambda = 0.7`, previously the recommended operating point, now
drops one question at hit@5: with fourteen documents, including two written to
sit lexically close to existing ones, MMR has more plausible-but-wrong
neighbours to promote in place of a relevant chunk. `lambda = 0.5` is the
better point on this set - redundancy falls 25% and the result set covers 18%
more documents while hit@1 and hit@5 are unchanged and MRR gives up 0.005.
That the optimum shifts when the corpus changes is itself the argument for
shipping the sweep rather than a tuned constant, so re-ranking stays **off by
default** and the trade-off is a flag rather than a hidden number.

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
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.857 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.857 |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.821 |

`cited gold` is the only column that uses the labels: the share of answers
citing a document the gold set marks relevant. It fell from 0.900 to 0.792 when
the hard questions were added - the correct direction, since retrieval misses
more of them, so more answers are built from the wrong passages - and recovered
to 0.857 when the corpus grew to fourteen documents, because the eight new
questions are retrieved and cited correctly. The
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

## Persistence

Every CLI invocation used to re-read the corpus, re-chunk it and re-fit both
indexes, even though none of them had changed. `ragkb index` snapshots the
fitted state (chunks, BM25 statistics, TF-IDF vectors) into a single JSON
file, and `--index` restores it directly, skipping tokenisation and fitting.

The file records a SHA-256 fingerprint of the exact document texts plus the
chunking parameters. Loading against a corpus that no longer matches raises
`StaleIndexError` instead of serving quietly wrong rankings; query-time knobs
(`--rrf-k`, `--mmr`, `--expand`) are deliberately outside the fingerprint, so
they can be changed on a loaded index without a rebuild. Unknown format
versions are refused rather than guessed at, and the write is atomic
(temp file + `os.replace`), so a crash mid-save cannot truncate the index.

Measured on this corpus (14 documents, 57 chunks), loading is about 1.7x
faster than a cold build; on a 300-document copy of the same material the gap
grows to about 2.4x, since fitting scales with corpus size while loading is a
single JSON parse. Rankings, scores and answers are bit-for-bit identical
either way - `tests/test_store.py` asserts exactly that.

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
- **Optional machinery has an identity setting.** MMR at `lambda = 1.0` and
  expansion at `alpha = 1.0` both have to reproduce the plain ranking exactly.
  That is the cheapest correctness check available for a component whose output
  is otherwise only judged by a metric that moves for many reasons.

## Tests

```bash
pytest -q      # 141 tests
```

The suite covers tokenisation and chunk boundaries, BM25 saturation and length
normalisation, TF-IDF normalisation, RRF scoring shape, every metric, the CLI,
MMR selection order and its redundancy invariants, the query-expansion model and
its weighted-query interfaces, answer grounding and citation accuracy, gold-set
integrity, and an end-to-end pass over the real corpus and gold set.
