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

# why did this chunk score what it scored? (and why did the right one lose?)
python -m ragkb.cli explain "what does the b parameter control in bm25"
python -m ragkb.cli explain "how would I know the chunk size was right" --doc chunking

python -m ragkb.cli --expand 5 search "what breaks when a query and a page use different words"
python -m ragkb.cli --rerank 0.5 search "what does the b parameter control in bm25"
python -m ragkb.cli --ngram 4 search "how does product quantisation work" --method ngram

# decline instead of answering when the corpus does not cover the question
python -m ragkb.cli --abstain 0.28 ask "how does the raft consensus protocol elect a new leader"

# index once, query many times without re-fitting
python -m ragkb.cli index --out index.json
python -m ragkb.cli --index index.json ask "what does the b parameter control in bm25"

# click through retrieval in a browser (standard library only)
python -m ragkb.cli --index index.json serve
```

`ask` uses the offline extractive answerer by default. Passing `--llm` switches
to the OpenAI answerer, which reads `OPENAI_API_KEY` and calls the chat
completions endpoint over `urllib`; it is optional and nothing else depends on
it.

## Architecture

```
question
   |
   +-- tokenize --> optional PRF expansion ------------------------+
   |    (borrow terms from a first retrieval pass)                 |
   v                                    v                          v
BM25 ranked list          TF-IDF cosine ranked list    char n-gram list
   |    (words)                    (words)                (optional)
   +----------------- reciprocal rank fusion ----------------------+
                        |
                        v
        optional cross-encoder re-rank (2nd stage)
                        |
                        v
              optional MMR diversity re-rank
                        |
                        v
                 top-k chunks (with doc provenance)
                        |
                        v
          optional abstention gate ----> "not covered" (no citations)
                        |
                        v
        extractive answerer  |  OpenAI answerer   -> answer + citations
```

| Module | Responsibility |
| --- | --- |
| `ragkb/text.py` | tokenizer, sentence splitter, sentence-aware overlapping chunker |
| `ragkb/bm25.py` | Okapi BM25 with configurable `k1` and `b`, ranked term-at-a-time over a postings list |
| `ragkb/vector.py` | sparse TF-IDF vectors, L2-normalised, cosine similarity over the same inverted index |
| `ragkb/ngram.py` | character n-gram TF-IDF: the same scoring, a different representation |
| `ragkb/retriever.py` | hybrid retriever, reciprocal rank fusion |
| `ragkb/expansion.py` | pseudo-relevance feedback query expansion (RM3-style) |
| `ragkb/rerank.py` | Maximal Marginal Relevance re-ranking, redundancy diagnostic |
| `ragkb/crossencoder.py` | second-stage re-ranker interface + lexical implementation |
| `ragkb/answerer.py` | extractive answerer (offline) and OpenAI answerer, both cite |
| `ragkb/abstain.py` | pre-generation gate: decline when the evidence does not cover the question |
| `ragkb/engine.py` | corpus loading, chunking, indexing, question answering |
| `ragkb/store.py` | index persistence: save/load with staleness fingerprinting |
| `ragkb/attribution.py` | per-term score attribution: which words and which ranks produced one chunk's score |
| `ragkb/metrics.py` | hit@k, recall@k, precision@k, nDCG@k, MRR |
| `ragkb/perturb.py` | seeded, keyboard-aware query corruption plus the edit distance that measures it |
| `ragkb/significance.py` | paired bootstrap intervals and randomization tests for metric differences |
| `ragkb/cli.py` | `stats`, `search`, `ask`, `explain`, `index`, `--mmr`, `--expand`, `--rerank`, `--ngram`, `--abstain`, `--index` |
| `evals/tune.py` | index-time sweep: chunk size, chunk overlap, `rrf_k` (2 or 3 lists) |
| `evals/failures.py` | per-question failure analysis: where a missed document really ranks, and which lever recovers it |
| `evals/robustness.py` | the same ablation under misspelled queries: which methods degrade, and how fast |
| `evals/significance.py` | error bars on the ablation table: which of those differences survive a paired test |

`corpus/` holds the sample knowledge base (eighteen documents on information
retrieval). `evals/gold.jsonl` holds seventy-eight labelled questions with the
documents that answer them, split into an `easy` and a `hard` slice.
`evals/unanswerable.jsonl` holds eighteen questions the corpus deliberately
cannot answer, split into `off-domain` and `near-domain`, which is what the
abstention gate is measured against.

## Evaluation

```bash
python evals/evaluate.py     # ablation + query-time sweeps + the third retriever
python evals/significance.py # error bars: which of those differences are real
python evals/holdout.py      # what each sweep's top row cost by picking itself
python evals/tune.py         # index-time sweeps (chunk size, overlap, rrf_k)
python evals/tune.py --ngram 4   # the same rrf_k sweep against three lists
python evals/faithfulness_eval.py  # is the answer inside the passages it cites?
python evals/abstention_eval.py    # does it know when there is no answer?
python evals/failures.py           # why each miss missed, and what would fix it
python evals/latency.py            # what each optional stage costs per query
python evals/robustness.py         # how each method holds up when the query is misspelled
pytest -q
```

Current numbers on the bundled corpus and gold set (18 documents, 80 chunks,
78 questions, k=5, document-level relevance).

The gold set is split by difficulty. The `easy` slice is forty-six
questions whose wording shares vocabulary with the target document. The `hard`
slice is thirty-two questions written to avoid the target document's own terms -
"if someone writes automobiles but the page only ever says cars" instead of
"vocabulary mismatch" - and seventeen of them are answered by two documents
rather than one.

**easy (46 questions)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.783 | 0.913 | 0.957 | 0.957 | 0.880 | 0.854 |
| bm25 | 0.804 | 0.913 | 0.957 | 0.957 | 0.890 | 0.867 |
| vector | 0.761 | 0.913 | 0.978 | 0.978 | 0.878 | 0.845 |

**hard (32 questions, 17 with two relevant documents)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.594 | 0.781 | 0.875 | 0.750 | 0.651 | 0.704 |
| bm25 | 0.562 | 0.781 | 0.844 | 0.719 | 0.635 | 0.682 |
| vector | 0.562 | 0.781 | 0.875 | 0.750 | 0.642 | 0.689 |

The easy slice is still the less informative half. Every question on it has
exactly one relevant document, which makes recall@k numerically identical to
hit@k - the column is decoration. hit@5 sits at 0.957 or better for all three
methods, so there is almost nothing left there for a retrieval change to move.

The hard slice restores headroom: hit@1 falls from 0.783 to 0.594 and MRR from
0.854 to 0.704, and recall@k finally diverges from hit@k because a question
answered by two documents can be half-satisfied. Eight questions land in exactly
that state at k=5, which is a failure mode the easy slice cannot express at all.

nDCG@5 is the one column that rewards ranking the *second* relevant document
high: MRR stops looking after the first hit, and recall@5 does not care whether
that second document sits at rank 2 or rank 5. Repeated chunks of one document
are credited once, so a top-5 stuffed with paraphrases of a single passage
cannot outscore one that ranks two distinct relevant documents.

At eighteen documents the hard slice finally shows the result the project was
built to look for, and it is worth stating precisely because it is the first
time it has appeared. **Hybrid beats both of its components on every hard-slice
column it can**: hit@1 0.594 against 0.562 for each, hit@5 0.875 against 0.844
for BM25, recall@5 0.750 against 0.719, nDCG@5 0.651 against 0.635 and 0.642,
MRR 0.704 against 0.682 and 0.689. At sixteen documents all three methods were
tied at 0.538 on hard hit@1 and fusion could only show an edge on ordering.

That is the direction the thesis predicts, and it is still not evidence. Every
one of those leads is one or two questions wide - a hard-slice question is worth
0.031 - and `evals/significance.py` puts a 95% interval across zero on all of
them (hit@1 hybrid vs BM25: +0.031, [+0.000, +0.094], W/L/T 1/0/31). One
question decided the headline. Read the table as "the sign is finally
consistent", not as "fusion wins".

The corpus keeps being grown for exactly this reason, each new document placed
beside an existing one it shares vocabulary with so that some question gets a
lexically plausible wrong neighbour: `late-interaction.md` beside
`reranking.md`, `vector-compression.md` beside `ann-search.md`,
`sparse-neural-retrieval.md` beside `bm25.md` and `inverted-index.md`,
`multi-hop-retrieval.md` beside `query-processing.md` and `rag.md`, and the two
newest, `domain-shift.md` beside `evaluation.md` and `bm25.md` and
`negative-mining.md` beside `embeddings.md` and `reranking.md`.

The one pattern that survived the growth unchanged is on the easy slice: BM25
still takes it outright at rank one (hit@1 0.804 against 0.783, MRR 0.867
against 0.854). Fusion buys ordering quality on paraphrased questions and pays
a little at rank one on literal ones. What did *not* survive is BM25's lead on
the all slice, where its hit@1 advantage (0.734 against 0.703 at sixteen
documents) has closed to a tie at 0.705.

One documented behaviour has now moved twice, which is worth tracking rather
than quietly re-pinning. `tests/test_expansion.py` originally asserted that
pseudo-relevance feedback rescues h18 (`rag`) at the shipped weight. At sixteen
documents the feedback pool changed and it no longer did - recovering h18 needed
`original_weight=0.7`, which lost three other questions - so the test was
re-pinned to h15. At eighteen documents h18 is recovered again at the shipped
setting, landing at rank 4, and so is h08. The lever never broke; its operating
point moves with the feedback pool. h15 stays pinned because it is the one case
that has held at all three corpus sizes.

### Tuning the constants: chunk size, overlap, and `rrf_k`

Everything above sweeps a *query-time* knob against one fixed index. But that
index is built from three numbers that had been constants since the first
commit - `target_tokens=120`, `overlap_tokens=30`, `rrf_k=60` - and nothing had
ever checked them. Three unexamined constants underneath every table is the
weakest claim in this README, so `evals/tune.py` sweeps them.

The two kinds of knob need different machinery, which is why they get one script
of their own. Chunking is index-time: changing it changes what a document is cut
into, so every row is a full rebuild of both indexes, and the table carries the
resulting corpus shape next to the accuracy because the shape *is* the
mechanism. `rrf_k` is query-time: it only rescales an existing pair of ranked
lists, so one fitted engine serves the whole sweep.

```bash
python evals/tune.py
python evals/tune.py --slice hard --chunk-specs 120:0,120:30,150:40
```

**chunking, all 78 questions, k=5** (deltas against the shipped default)

| target:overlap | chunks | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 60:15 | 212 | 0.692 | 0.872 | 0.923 | 0.878 | 0.785 | 0.783 |
| 80:20 | 138 | **0.718** | 0.846 | 0.936 | 0.885 | 0.795 | 0.798 |
| 100:25 | 98 | **0.718** | **0.885** | 0.936 | 0.885 | **0.803** | **0.808** |
| 120:0 | 65 | 0.641 | 0.833 | 0.936 | 0.885 | 0.763 | 0.752 |
| 120:15 | 78 | 0.705 | 0.859 | **0.962** | **0.910** | 0.800 | 0.799 |
| **120:30** (default) | 80 | 0.705 | 0.859 | 0.923 | 0.872 | 0.786 | 0.793 |
| 120:60 | 97 | 0.692 | 0.859 | 0.936 | 0.885 | 0.789 | 0.791 |
| 150:40 | 62 | 0.692 | 0.872 | 0.923 | 0.885 | 0.793 | 0.786 |
| 180:45 | 54 | 0.692 | **0.897** | **0.962** | 0.897 | 0.800 | 0.803 |
| 200:50 | 45 | 0.679 | 0.885 | 0.910 | 0.853 | 0.770 | 0.776 |

**Overlap earns its keep, and the bigger corpus made the case much stronger.**
`120:0` is the same chunk size with the overlap switched off, and it is the
worst row in the table at early ranks by a wide margin: hit@1 falls
0.705 -> 0.641 and MRR 0.793 -> 0.752. At sixteen documents that hit@1 loss was
0.031 - two questions - and could fairly be called noise. At eighteen it is
0.064, five questions, and it is the largest single movement anywhere in the
sweep. That is the design claim confirmed: an idea straddling a boundary really
is retrievable from both sides, and dropping overlap really does cost the
questions that depend on it.

But `120:0` simultaneously *gains* reach: hit@5 rises 0.013 and recall@5 0.013.
That is not a contradiction, it is the redundancy problem measured from the
other end. Overlap manufactures near-duplicate chunks, near-duplicates crowd
each other into the top five, and a top-five spent on two views of one passage
is a top-five that did not reach a second document. The MMR section treats that
as a re-ranking problem; this table shows it is also a chunking one, and the two
dials fight over the same handful of questions.

**The default is defensible but it is a choice, not a maximum, and each corpus
growth has made that sharper.** At 14 documents nothing beat 120:30 at rank one.
At 16, `120:15` and `120:60` took hit@1 by two questions. At 18 the candidate has
moved again: `100:25` now leads hit@1, hit@3, nDCG@5 and MRR simultaneously, with
`80:20` tied at rank one, while `120:15` and `180:45` take the reach columns. The
shipped default again wins no column outright.

That the winner keeps changing is itself the finding, and it is the same one the
holdout section measures directly: a sweep's top row is partly a property of the
questions rather than of the configurations. Adopting a new default would
invalidate every other table in this README, all of which were measured at
120:30, and a one-question lead on 78 questions is inside the noise. It is
recorded as the candidate it is.

The extremes still fail as expected, which is the evidence the sweep measures
something real. `60:15` cuts the corpus into 212 chunks averaging 51 tokens and
gives up hit@5, recall@5 and MRR: chunks too short to contain a whole answer.
`200:50` is now the weakest reach row in the table. The useful range is a broad
plateau between roughly 80 and 130 target tokens, which is worth knowing
precisely because it means the exact value inside that band is not what any
other result here depends on.

**`rrf_k`, all 78 questions, k=5.** The `changed` column counts questions whose
top-5 ordering differs from the default - the measurement that separates
"equally good" from "does nothing at all".

| rrf_k | hit@1 | hit@5 | nDCG@5 | MRR | changed |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.705 | 0.923 | 0.786 | 0.793 | 8 |
| 5 | 0.705 | 0.923 | 0.786 | 0.793 | 4 |
| 10 | 0.705 | 0.923 | 0.786 | 0.793 | 1 |
| 20 | 0.705 | 0.923 | 0.786 | 0.793 | 0 |
| **60** (default) | 0.705 | 0.923 | 0.786 | 0.793 | 0 |
| 200 | 0.705 | 0.923 | 0.786 | 0.793 | 0 |
| 1000 | 0.705 | 0.923 | 0.786 | 0.793 | 0 |

`rrf_k` is inert on this corpus, and the `changed` column is what proves it:
every value at or above 20 returns byte-identical top-5 orderings on all 78
questions, and the values below it reorder a handful without moving a single
metric. This is not "tuning did not help", it is "there was nothing to tune",
and those are different claims that an accuracy table alone cannot tell apart.
The bigger corpus sharpened the diagnosis rather than the outcome: the number of
reordered questions at `rrf_k = 1` grew from 5 to 8, so the knob is touching more
orderings than it used to and still changing no metric by 0.001.

The reason is structural. `rrf_k` sets how steeply `1 / (rrf_k + rank)` decays,
which interpolates between two known fusion rules: as `rrf_k` approaches zero
the first list to rank something highly dominates, and as it grows large every
term flattens toward `1/rrf_k` and the ordering converges on the sum of ranks -
Borda count. With only *two* lists that already agree on most questions, both
ends produce the same order. At eighteen documents no value
moves any metric at all, not even `rrf_k = 1`. The knob would start to matter
with more retrievers in the fusion, or with retrievers that disagree more than
BM25 and TF-IDF do here - and the honest version of that claim is that this
corpus cannot demonstrate it.

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

**easy (46 questions)**

| terms | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.783 | 0.913 | 0.957 | 0.957 | 0.854 |
| 5, a=1.0 | 0.783 | 0.913 | 0.957 | 0.957 | 0.854 |
| 3, a=0.9 | 0.761 | 0.891 | 0.957 | 0.957 | 0.838 |
| 5, a=0.8 | 0.696 | 0.870 | 0.957 | 0.957 | 0.800 |
| 5, a=0.7 | 0.587 | 0.848 | 0.935 | 0.935 | 0.731 |

**hard (32 questions, 17 with two relevant documents)**

| terms | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.594 | 0.781 | 0.875 | 0.750 | 0.704 |
| 5, a=1.0 | 0.594 | 0.781 | 0.875 | 0.750 | 0.704 |
| 3, a=0.9 | 0.469 | **0.812** | **0.906** | **0.812** | 0.635 |
| 5, a=0.8 | 0.375 | **0.812** | 0.875 | 0.797 | 0.568 |
| 5, a=0.7 | 0.375 | 0.781 | 0.875 | 0.781 | 0.570 |

`alpha = 1.0` puts all the weight on the original query, so nothing is borrowed
and the ranking has to be identical to the unexpanded one. It is, on every one
of the 78 questions - that row and the test behind it are the correctness check
on this whole path, the same job `lambda = 1.0` does for MMR.

The split is the result, and each round of corpus growth has sharpened both
sides of it. On the hard slice, three borrowed terms at `alpha = 0.9` still buy
real reach - hit@3 rises 0.781 -> 0.812, hit@5 0.875 -> 0.906 and recall@5
0.750 -> 0.812, the best recall any configuration reaches on that slice - and
expansion remains the only lever in the repo that recovers h08, h15 and h18 from
a complete miss at k=5. But rank one pays for it, and pays more than it used to:
hit@1 falls 0.594 -> 0.469 and MRR 0.704 -> 0.635 at that same setting. On the
easy slice the same borrowing costs four questions at rank one and 0.016 of MRR
while gaining nothing at all at k=5 - those questions already shared vocabulary
with their target, so every borrowed term is noise.

The operating point has now moved in both directions across three corpus sizes,
which is worth recording because it is the kind of thing a stale README hides.
At 14 documents, `alpha = 0.8` recovered h18 ("why can what a system knows be
changed without any training at all?"). At 16 it did not: the feedback pool had
gained `sparse-neural-retrieval` and `multi-hop-retrieval`, both of which discuss
training and updating knowledge, and the terms borrowed from them pulled h18 away
from `rag`. At 18 it recovers h18 again, at rank 4, along with h08 and h15.
Nothing about the method changed across those three runs. `tests/test_expansion.py`
stays pinned to h15, the one case that has survived all three.

That is the textbook pseudo-relevance feedback trade-off, and this corpus
reproduces it cleanly, including its instability: expansion buys reach on
paraphrased questions, pays for it in early precision on literal ones, and which
specific questions land on which side is a property of the corpus rather than of
the method. It is **off by default** and exposed as `--expand`. Two cautions on
reading the tables: the hard slice is 32 questions, so a 0.031 step is one
question; and the feedback documents come from the same retriever being
evaluated, so when the first pass is wrong, expansion confidently makes it
wronger.

### Re-ranking: a second stage that reads the pair

Fusion recalls far better than it precisely ranks. On the full gold set hit@5 is
0.923 but hit@1 is only 0.705: for a fifth of the questions the answer is
already in the pool and merely sitting in the wrong place. Nothing in the first
stage can fix that, because BM25 and TF-IDF both score the query and the passage
*independently* - they see which terms matched and how often, never where those
terms sit relative to each other. A chunk saying "the b parameter" in one
paragraph and "length normalisation" three paragraphs later is indistinguishable
from one saying "the b parameter controls length normalisation" in a clause.

That is what a cross-encoder is for: it reads query and passage jointly, and
because it is too expensive to run over a collection it runs as a *second* stage
over the first stage's candidate pool. `ragkb/crossencoder.py` keeps that
architecture and swaps the transformer for three positional features the
standard library can compute - IDF-weighted **coverage** of the query terms,
**proximity** (how tight the smallest window containing them is), and **phrase**
overlap on adjacent bigrams. `CrossEncoder` is a two-method interface, so a real
model or a hosted re-ranking API drops in without touching the retriever.

The second-stage score is blended with the min-max normalised first-stage score:
`alpha * first_stage + (1 - alpha) * cross`. Sweep over the full gold set, k=5:

| alpha | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.705 | 0.859 | 0.923 | 0.786 | 0.793 |
| 1.0 | 0.705 | 0.859 | 0.923 | 0.786 | 0.793 |
| 0.8 | 0.731 | 0.872 | **0.936** | 0.804 | 0.811 |
| 0.6 | **0.744** | 0.872 | **0.936** | **0.808** | 0.820 |
| **0.5** | 0.731 | 0.872 | **0.936** | 0.803 | 0.813 |
| 0.4 | 0.731 | 0.872 | **0.936** | 0.806 | 0.814 |
| 0.2 | 0.731 | **0.897** | 0.910 | 0.797 | 0.807 |
| 0.0 | 0.705 | 0.885 | 0.923 | 0.794 | 0.802 |

And on the hard slice, where the first stage has the most room:

| alpha | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.594 | 0.781 | 0.875 | 0.651 | 0.704 |
| 0.6 | 0.625 | 0.781 | 0.906 | 0.672 | 0.726 |
| **0.5** | **0.656** | 0.781 | **0.906** | **0.683** | **0.742** |
| 0.2 | 0.625 | **0.844** | 0.875 | 0.671 | 0.721 |

This remains the largest single-increment gain the project has measured, and it
has survived two rounds of corpus growth: at alpha=0.5, hit@1 goes
0.705 -> 0.731 and MRR 0.793 -> 0.813, and on the hard slice hit@1 goes
0.594 -> 0.656 while hit@5 *rises* 0.875 -> 0.906. The mechanism is exactly the
one predicted - it converts recall the first stage already had into precision at
rank one, and the hard slice is where it converts the most.

The shipped alpha=0.5 is no longer the best row on the all slice; alpha=0.6 now
takes hit@1, nDCG@5 and MRR there by one question. It is still the best row on
the hard slice, which is the slice that has headroom, so the default is left
where it is and the alternative is recorded rather than adopted.

Read the trade-off honestly, and note that it has sharpened. Leaning harder on
the second stage (alpha <= 0.2) now *costs* hit@1 as well as hit@5
(0.936 -> 0.910): at sixteen documents the low-alpha rows still bought rank one
and paid in reach, and at eighteen they pay at both ends. A re-ranker reorders
the pool, it cannot enlarge it, so pushing it too far shuffles a correct answer
out of the top five - and with more documents in the pool there is more for it to
shuffle wrongly. The easy slice moves less at rank one (0.783 -> 0.826 at its
best) and gives back hit@5 below alpha=0.4, which is the same story: positional
features help most where the wording differs. alpha
is 0.5 by default for that reason, and the stage is **off by default** and
exposed as `--rerank`, on the same principle as `--expand`. `alpha = 1.0`
reproduces the unranked ordering exactly, which is the correctness check on the
whole path.

### A third retriever: character n-grams, and what it cost the thesis

Everything above this point fuses two ranked lists. It is worth being precise
about what those two lists are: BM25 and TF-IDF are two *scoring functions*
over one **representation**. Both call the same `tokenize`, both index whole
words, and both are therefore blind to exactly the same things - `quantise` and
`quantisation` are unrelated symbols to either of them, so is `reranking`
against `re-ranking`, and one transposed letter is a total miss. Fusing two
retrievers that share a blind spot cannot see into it, which is the structural
reason the `rrf_k` sweep found nothing to tune: the two lists agree on 0.867 of
their top-5 by chunk, and a fusion constant has no room to arbitrate lists that
already concur.

`ragkb/ngram.py` changes the representation instead of the score. Each token is
padded with `_` boundary markers and cut into overlapping character 4-grams, so
`quantise` and `quantisation` share `_qua`, `quan`, `uant`, `anti`, `ntis` and
`tise` and match partially with no stemmer, lexicon or model. Deliberately,
`CharNgramIndex` **subclasses** `TfidfIndex` and overrides only the analyzer -
same smoothed IDF, same sublinear term frequency, same L2 normalisation, same
cosine, same ranking loop - so what the table below compares is two
representations and not two implementations.

```bash
python -m ragkb.cli --ngram 4 search "how does product quantisation work" --method ngram
python evals/tune.py --ngram 4        # re-sweep rrf_k against three lists
```

Adding it drops the mean pairwise top-5 overlap between the fused lists from
0.867 to 0.669, which is the mechanism working: the third list is contributing
a genuinely different opinion rather than a paraphrase of an existing one.
`hybrid2` is the shipped two-list fusion, `ngram` is the n-gram list alone and
`hybrid3` is all three fused.

**all 78 questions, k=5**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid2 | 0.705 | 0.859 | 0.923 | 0.872 | 0.786 | 0.793 |
| **ngram** | **0.769** | **0.936** | **0.962** | **0.936** | **0.851** | **0.846** |
| hybrid3 | 0.744 | 0.885 | 0.949 | 0.910 | 0.828 | 0.827 |

**hard slice (32 questions, 17 with two relevant documents)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid2 | 0.594 | 0.781 | 0.875 | 0.750 | 0.651 | 0.704 |
| **ngram** | **0.625** | **0.844** | **0.906** | **0.844** | **0.719** | **0.734** |
| hybrid3 | **0.625** | 0.781 | **0.906** | 0.812 | 0.705 | **0.734** |

**The result is the largest this project has measured, and it is not the result
the experiment was designed to find.** The n-gram list was added as a third
input to the fusion. It beats the fusion - alone, on every column of the all
slice and on every column of the hard slice bar the two it ties `hybrid3` on.
Against `hybrid2` on the hard questions it takes hit@1 from 0.594 to 0.625 and
recall@5 from 0.750 to 0.844, which is three of thirty-two questions where the
answer was simply not in the top five before.

**And `hybrid3` lands between its inputs, which is the finding worth keeping.**
On the all slice, fusing the strong list with the two weaker ones is *worse than
the strong list alone* on every single column. On the hard slice it has caught
up at rank one and on MRR but still gives back hit@3, recall@5 and nDCG@5. That
is exactly where an unweighted average of unequal inputs should land, and it is
the limit of RRF stated plainly: `1 / (rrf_k + rank)` gives every list the same
vote regardless of how good that list is, so a third opinion helps when the
lists are comparable and dilutes when they are not. The project's working
assumption from the first commit - that adding retrievers to the fusion is how
retrieval gets better - is the thing this table refutes. What made retrieval
better was changing the representation; the fusion then gave part of it back.

Two cautions before anyone reads this as "character n-grams beat words". The
first is arithmetic: 78 questions, so the 0.064 gap in hit@1 is five questions,
and the hard slice is 32, where one question is 0.031. The second is
structural, and it matters more. The mechanism that wins here - partial credit
for shared substrings - is the same mechanism that should *lose* on a corpus of
lexically similar documents, because n-grams blur distinctions that whole words
preserve. Eighteen documents on largely different topics is still close to the
friendliest case this idea will ever get, though the pressure keeps building:
six of the eighteen were placed deliberately beside a lexical neighbour, and the
n-gram list's hard-slice margin over `hybrid2` has narrowed monotonically with
each pair added - hit@1 +0.136 at 14 documents, +0.077 at 16, +0.031 at 18. Two
more rounds of that trend would erase it. The honest claim is "on this corpus,
character 4-grams retrieve better than words", the trend is in the direction the
structural argument predicts, and the experiment that would settle it is a
corpus with far more near-neighbours in it.

It is therefore **off by default** and exposed as `--ngram N`, on the same
principle as `--expand` and `--rerank`, and for the same reason `150:40`
chunking was recorded rather than shipped: every other table in this README was
measured against the two-list fusion, and quietly changing the default would
invalidate all of them. Unlike those flags it is index-time, not query-time -
it fits a third index - so a saved index cannot gain one by flag, and `--ngram`
against an index built without it fails loudly rather than answering from a
two-list fusion nobody asked for.

The cost is real and small in absolute terms: 4,642 gram features against 1,688
word features (2.8x), and fitting the third index takes about 22 ms against
about 4 ms for TF-IDF on this corpus. Denser vectors are also why the list is
noisier per-match - most English 4-grams are common, so everything matches
everything a little - which is what makes the standalone result surprising
rather than obvious.

### Weighted RRF: acting on "the third list is the best list"

The n-gram section ends on an uncomfortable number. The character n-gram list
*alone* scores nDCG@5 0.851, while the three-list fusion that contains it
scores 0.828. Textbook RRF weights every list identically, so the fusion is
averaging its best retriever with two worse ones and landing between them.
That is not a bug in RRF; it is RRF doing exactly what it promises, on inputs
that violate the assumption behind the promise.

The generalisation is one symbol wide - `sum_i w_i / (rrf_k + rank_i)` instead
of `sum_i 1 / (rrf_k + rank_i)` - and `--fusion-weights` exposes it:

```bash
python -m ragkb.cli --ngram 4 --fusion-weights "bm25=1,vector=1,ngram=4" \
    search "how does product quantisation work"
```

Weights are rescaled to mean 1.0, which has three consequences worth stating.
Only the *ratios* matter, so `bm25=2,vector=2` is exactly `bm25=1,vector=1`.
Uniform weights reproduce plain RRF bit-for-bit, so the feature is genuinely
opt-in. And fused scores keep their scale, so `rrf_k` and the abstention
threshold - both calibrated against fused-score magnitudes - do not silently
need re-tuning when the ratios change. A list left out of the spec keeps
weight 1.0 ("no opinion"); weight it `0` to drop it.

Sweeping the weight on the n-gram list, over all 78 gold questions at k=5:

| weights | hit@1 | ndcg@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| uniform (plain RRF) | 0.744 | 0.828 | 0.910 | 0.827 |
| `ngram=2` | 0.756 | 0.836 | 0.917 | 0.836 |
| `ngram=3` | 0.782 | 0.844 | 0.904 | 0.849 |
| **`ngram=4`** | **0.808** | **0.854** | 0.904 | **0.865** |
| `ngram=6` | 0.795 | 0.847 | 0.904 | 0.854 |
| ngram only (control) | 0.769 | 0.851 | **0.936** | 0.846 |

The gain concentrates in the hard slice, which is where fusion was supposed to
earn its keep: nDCG@5 0.705 → 0.729 and hit@1 0.625 → 0.688 at `ngram=4`.

Two honest readings of that table, both of which matter more than the win:

**None of it is significant.** Run through the same paired bootstrap that
`evals/significance.py` uses, `ngram=4` against uniform is +0.026 nDCG@5 with a
95% interval of [-0.005, +0.059] and p=0.132 - won 13 questions, lost 7, tied
58. hit@1 is +0.064 with 8 wins and 3 losses, interval [-0.013, +0.141],
p=0.227. On 78 questions a handful of queries is still the entire effect, and
the interval includes zero on every metric. Fourteen more questions moved both
intervals closer to excluding zero without getting there, which is the useful
signal: the comparison is trending, and the gold set would still have to grow
considerably further before it could resolve.

**The control now loses on ordering and wins on reach, which is new.** At 16
documents `ngram only` beat every weighted fusion on both nDCG@5 and recall@5.
At 18, `ngram=4` takes nDCG@5 (0.854 against 0.851) as well as hit@1 and MRR,
while the control keeps recall@5 (0.936 against 0.904) outright and by a wide
margin. Weighting has closed the ordering gap uniform RRF opened; it has not
closed the reach gap, and the split is consistent - fusion reorders what it has
better than its best input, and still retrieves less. The defensible conclusion
is not
"weighted RRF is better" - it is that *on this corpus, at this size, fusion has
still not been shown to beat its own best input on reach, and the ordering lead
it has just taken is 0.003 with an interval across zero*. The weight knob makes
that visible rather than hiding it behind a fused default.

So the shipped default stays uniform. A non-significant delta measured against
a control that still wins is not grounds for changing what the engine does by
default; it is grounds for a knob and a table.

### `rrf_k` was not inert. It had nothing to arbitrate.

The tuning section above closed with a claim it could not test: that `rrf_k`
"would start to matter with more retrievers in the fusion, or with retrievers
that disagree more than BM25 and TF-IDF do here - and the honest version of
that claim is that this corpus cannot demonstrate it. **It can now**: adding a
third, structurally different list moves `rrf_k = 1` from 8 changed orderings
to 44, and the sweep is repeated under "`rrf_k` was not inert" below." The third list is the
experiment that demonstrates it. Same corpus, same gold set, same sweep, one
more ranked list:

| rrf_k | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR | changed (2 lists) | changed (3 lists) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.744 | **0.897** | 0.936 | 0.820 | 0.820 | 8 | **44** |
| 5 | 0.744 | 0.885 | 0.936 | 0.820 | 0.821 | 4 | 38 |
| 10 | 0.744 | 0.885 | **0.949** | 0.825 | 0.824 | 1 | 26 |
| 20 | 0.744 | 0.885 | **0.949** | **0.828** | **0.827** | 0 | 19 |
| **60** (default) | 0.744 | 0.885 | **0.949** | **0.828** | **0.827** | 0 | 0 |
| 200 | 0.744 | 0.885 | **0.949** | **0.828** | **0.827** | 0 | 2 |
| 1000 | 0.744 | 0.885 | **0.949** | **0.828** | **0.827** | 0 | 3 |

With two lists, every value at or above 20 returns byte-identical orderings on
all 78 questions and the accuracy columns never move. With three, `rrf_k = 1`
reorders 44 of 78 - well over half the gold set - and the sweep develops a
shape. The shape has also flipped since sixteen documents, which is worth
recording: there, small `rrf_k` beat the default on the reach columns; here the
default and everything above it take hit@5, nDCG@5 and MRR, and it is `rrf_k = 1`
that gives ground on reach while buying hit@3. Both readings were one question
wide, which is the real lesson. The point is not the gain, it is that the earlier
"there was nothing to tune" was a statement about the *inputs* rather than about
the knob, and it took a genuinely different list to tell those two explanations
apart. Note that the largest values still change 2 and 3 orderings rather than 0 -
there are a few questions where the ranks are far enough apart for the Borda
limit to differ from the default.

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

Sweeping lambda over all 78 questions at k=5 (`redundancy` is the mean
pairwise cosine within a result set; `distinct` is how many documents it spans):

| lambda | hit@1 | hit@5 | MRR | redundancy | distinct |
| --- | --- | --- | --- | --- | --- |
| off | 0.705 | 0.923 | 0.793 | 0.135 | 3.72 |
| 1.0 | 0.705 | 0.923 | 0.793 | 0.135 | 3.72 |
| 0.9 | 0.705 | 0.923 | 0.793 | 0.130 | 3.85 |
| 0.7 | 0.705 | 0.923 | 0.790 | 0.112 | 4.32 |
| 0.5 | 0.705 | 0.923 | 0.787 | 0.099 | 4.55 |
| 0.3 | 0.705 | 0.910 | 0.782 | 0.090 | 4.69 |

`lambda = 1.0` reproduces the unmodified ranking exactly, which is the cheapest
available correctness check on the implementation. The sweet spot has moved with
the corpus twice, and at 18 documents it has stopped moving, which is itself the
finding. At 14 documents `lambda = 0.7` dropped a question at hit@5 and 0.5 was
the better point. At 16 and again at 18 the accuracy columns are flat from 1.0
all the way down to 0.5 - hit@1 and hit@5 do not move at all, and at 0.9 not even
MRR moves - while only `lambda = 0.3` finally costs a question at hit@5. So the
whole 0.9-to-0.5 band is free: redundancy falls 27% and the result set covers 22%
more documents for 0.006 of MRR and nothing else. That the optimum shifted at all
when the corpus changed is the argument for shipping the sweep rather than a tuned
constant, so re-ranking stays **off by default** and the trade-off is a flag
rather than a hidden number.

### Failure analysis: which misses, and what would fix them

Every table above is an average. Averages are exactly the wrong instrument for
choosing between the four levers this repo now ships, because a lever that
rescues one question and breaks another lands within a rounding hair of a lever
that does nothing at all. `evals/evaluate.py` had been ending by printing the
ids it missed and stopping there. `evals/failures.py` picks each one up and asks
where the document actually is, how far that is, and whether anything already in
the codebase pulls it back.

```bash
python evals/failures.py
python evals/failures.py --method bm25 -k 3 --slice hard
```

Sixteen of the ninety-five (question, relevant document) pairs are missed at k=5.

| id | slice | document | hybrid | bm25 | vector | coverage | category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q15 | easy | evaluation | 11 | 11 | 11 | 0.44 | deep-miss |
| q26 | easy | citations | 6 | 7 | 4 | 0.64 | fusion-loss |
| h01 | hard | embeddings | 18 | 18 | 18 | 0.28 | deep-miss |
| h02 | hard | bm25 | 15 | 15 | 16 | 0.38 | deep-miss |
| h03 | hard | tfidf | 18 | 18 | 18 | 0.36 | deep-miss |
| h05 | hard | chunking | 14 | 15 | 14 | 0.39 | deep-miss |
| h05 | hard | evaluation | 13 | 13 | 13 | 0.33 | deep-miss |
| h08 | hard | reranking | 7 | 8 | 4 | 0.39 | fusion-loss |
| h15 | hard | rank-fusion | 6 | 7 | 6 | 0.62 | near-miss |
| h18 | hard | rag | 9 | 9 | 7 | 0.54 | near-miss |
| h20 | hard | reranking | 10 | 11 | 9 | 0.37 | near-miss |
| h22 | hard | reranking | 9 | 7 | 9 | 0.28 | near-miss |
| h27 | hard | domain-shift | 7 | 8 | 7 | 0.50 | near-miss |
| h27 | hard | evaluation | 13 | 15 | 10 | 0.30 | deep-miss |
| h30 | hard | embeddings | 9 | 9 | 9 | 0.38 | near-miss |
| h32 | hard | negative-mining | 8 | 6 | 9 | 0.71 | near-miss |

**Two of them are now fusion losses, where at sixteen documents there were
none.** That is the honest correction to a claim this section used to make
flatly. In q26 and h08 TF-IDF ranked the document at 4 - inside k=5 - and the
fusion pushed it out to 6 and 7 respectively, because BM25 disagreed strongly
enough to outvote it. RRF is doing what an unweighted vote does when one voter
is right and the other is confidently wrong. That is a real cost of fusion, it
was invisible on a smaller gold set, and it is exactly the kind of thing growing
the corpus is meant to expose.

The other fourteen still are not fusion's fault: in most of them the fused rank
is identical to both component ranks, and where it differs, neither component
had the document inside k either. The `coverage` column - the share of the
question's distinct terms that occur anywhere in the document - says why. It
averages 0.42 and is below 0.40 on eight of the sixteen. These are mostly
hard-slice questions written to avoid the target document's own vocabulary, and
both word retrievers index whole words, so they are two scores over one
representation that cannot see what neither can match.

Which lever recovers them is then an empirical question rather than an argument,
so the script re-runs every failing pair under all four (`*` marks a rank inside
k=5):

| id | document | expansion | rerank | mmr | hybrid3 | ngram-only |
| --- | --- | --- | --- | --- | --- | --- |
| q15 | evaluation | 13 | 12 | 11 | 9 | 1* |
| q26 | citations | 10 | 7 | 5* | 3* | 1* |
| h01 | embeddings | 18 | 18 | 18 | 18 | 18 |
| h02 | bm25 | 12 | 15 | 15 | 14 | 5* |
| h03 | tfidf | 16 | 18 | 18 | 14 | 4* |
| h05 | chunking | 11 | 14 | 13 | 15 | 15 |
| h05 | evaluation | 14 | 13 | 14 | 13 | 14 |
| h08 | reranking | 3* | 7 | 7 | 5* | 4* |
| h15 | rank-fusion | 3* | 4* | 7 | 4* | 4* |
| h18 | rag | 5* | 9 | 10 | 4* | 3* |
| h20 | reranking | 14 | 10 | 10 | 8 | 5* |
| h22 | reranking | 15 | 8 | 9 | 4* | 2* |
| h27 | domain-shift | 7 | 8 | 7 | 8 | 2* |
| h27 | evaluation | 10 | 13 | 13 | 13 | 11 |
| h30 | embeddings | 5* | 10 | 10 | 7 | 8 |
| h32 | negative-mining | 6 | 6 | 8 | 9 | 6 |
| | **recovered** | 4/16 | 1/16 | 1/16 | 5/16 | **10/16** |

**The character n-gram list alone recovers ten of the sixteen failures the word
retrievers share.** It is the only lever in the repo that changes the
*representation* rather than re-scoring or re-ordering the same one, and this is
the sharpest statement of that difference the project has produced: MMR recovers
one and re-ranking recovers one, because re-ordering a pool cannot admit a
document the pool never contained. Both were already known to be pool-bound in
principle; here it is in the only arithmetic that can show it.

Expansion is the row that keeps moving with the corpus. At 14 documents it
recovered one failure; at 16 it recovered three, one of which needed a weight
the engine does not ship; at 18 it recovers four (h08, h15, h18, h30) and all
four are at the shipped `alpha=0.8`. It is a lever with wide variance whose good
runs and bad runs are both properties of the feedback pool, not of the code.

It also explains a result from the section above that had been left as a
curiosity. The three-list table showed `ngram` beating `hybrid3` on the hard
slice's ordering columns (nDCG@5 0.719 vs 0.705, recall@5 0.844 vs 0.812), which
reads as a fusion that dilutes its best member. This table is the same fact at
question granularity: on the pairs where the word lists both fail, the n-gram
list carries ten rescues and fusion averages half of them back out against two
word lists that ranked the document fourteenth.

What survives every lever is the backlog:

```
nothing in the repo recovers these:
  h01 -> embeddings (rank 18, coverage 0.28): If someone writes automobiles but
         the page only ever says cars, what kind of search still finds it?
  h05 -> chunking (rank 14, coverage 0.39): How do I decide how much text to put
         in each piece, and how would I know the choice was right?
  h05 -> evaluation (rank 13, coverage 0.33)
  h27 -> evaluation (rank 13, coverage 0.30): Why can a system that tops a public
         leaderboard still be the wrong first choice for a specialised collection
         nobody has labelled?
  h32 -> negative-mining (rank 8, coverage 0.71)
```

`h01` is the honest one, and it is dead last of eighteen under all five
configurations. The question asks for the retrieval method that survives a
synonym, using a car/automobile analogy the corpus never uses; the answer is in
`embeddings.md`, which never says either word. Character n-grams fix
*morphology* - "controls" against "control" - not synonymy, so the one lever
that rescues the other ten cannot touch this. Nothing short of a semantic
representation will, which makes `h01` a fair statement of where a
dependency-free design actually stops rather than a bug to be tuned away. `h05`
is a different shape: one question with two relevant documents, both stuck in
the rank 13-15 band, where every lever moves it a place or two and none moves it
nine. It gets *worse* with each corpus growth, which is what a rank-based backlog
does when the collection grows and the question still matches nothing.

The two new entries are the useful ones, because they are the intended product
of this round of growth rather than survivors of the last. `h27` is a
two-document question whose second document (`evaluation`) sits at 13 with
coverage 0.30 - the retriever finds the new `domain-shift` document under
`ngram-only` but never assembles both halves. `h32` is the opposite failure and
the more interesting: coverage is 0.71, the highest of any unrecovered pair in
the table, so the words *are* there and the document still lands at rank 8. That
is not a vocabulary problem, it is a ranking one, and it is the first pair in
this backlog that a better scorer rather than a better representation could
plausibly fix.

### Attribution: which terms actually produced that score

Everything above this line is an aggregate or a rank. `evals/failures.py` will
tell you that `h18`'s gold document sits at rank 9; it cannot tell you *why*,
and "rank 9" is not something you can act on. Both scorers are sums over terms,
so the score can be decomposed back into the summands exactly - not a saliency
heuristic, the loop body with the accumulation unrolled:

- BM25 sums `weight * idf(t) * tf(t)(k1+1) / (tf(t) + norm)` over query terms;
- TF-IDF cosine on L2-normalised vectors is a dot product, so each shared term
  contributes `query_vector[t] * doc_vector[t]`;
- RRF sums `1 / (rrf_k + rank)` over lists.

`tests/test_attribution.py` asserts each of those sums against the scorer it
claims to decompose, so a decomposition that stopped adding up would fail rather
than mislead.

```bash
python -m ragkb.cli explain "what does the b parameter control in bm25"
python -m ragkb.cli explain "..." --rank 2          # the 2nd retrieved chunk
python -m ragkb.cli explain "..." --doc chunking    # a document that lost
python -m ragkb.cli explain "..." --chunk 31 --method bm25
```

Take `h18`, one of the six questions hybrid misses outright at k=5. Its gold
document is `rag.md`, and `--doc` explains a chunk whether or not it was
retrieved, which is the only way to ask why the right answer lost:

```
$ python -m ragkb.cli explain \
    "Why can what a system knows be changed without any training at all?" \
    --doc rag --terms 5

#11  [rag]  rag::1  score=0.02878

fusion (rrf_k=60):
  bm25        rank 10  1/(60+10) = 0.014286
  vector       rank 9  1/(60+9) = 0.014493
  total                          0.028778

bm25 score 5.74108 (top 5 of 5 matched terms)
  term                    qw      tf      df       idf     contrib
  system               1.000       1       8     2.254     2.12448
  be                   1.000       2      30     0.977     1.33693
  can                  1.000       3      36     0.797     1.28491
  at                   1.000       1      29     1.010     0.95186
  a                    1.000       2      78     0.031     0.04291
  missing from this chunk: all, any, training, what, why, without
  not in the corpus at all: changed, knows
```

Every point `rag.md` scored came from `system`, `be`, `can`, `at` and `a`. Not
one content word of the question contributed anything, and the two that carry
its meaning - `changed` and `knows` - are not in the collection vocabulary at
all. The document that beat it to rank one, `citations.md`, won on the same kind
of vocabulary: `system`, `without`, `what`, `can`, `be`. So this is not a ranking
that needs tuning; both documents were ranked on function words because the
question and the corpus share no content words, and the retriever had nothing
else to go on.

That distinction is why the negative space is reported in two separate lines
rather than one, and the difference is the difference between a fixable miss and
an unfixable one:

- **missing from this chunk** - the term is in the corpus but not here. A
  chunking, pooling or ranking problem; a bigger candidate pool or a different
  chunk size can move it.
- **not in the corpus at all** - there is nothing to match, and no re-ranker
  can invent it. This is the signature the n-gram list attacks when the cause is
  morphology (`control` against `controls`, and the standalone token `control`
  genuinely never appears in this corpus) and that query expansion attacks when
  the cause is a missing near-synonym. Neither touches `knows` against
  `knowledge`-free prose, which is the same wall `h01` hits.

Two details keep the trace honest rather than plausible. It calls the retriever's
own ranking pass instead of re-deriving one, including which of the two argument
paths `search` would have taken - the weighted path damps repeated query terms
differently from the token path, so explaining the wrong one would describe a
ranking the engine never produced. And the `tf` column is blank for the vector
rows because a TF-IDF cosine scores an L2-normalised weight in which the
original count is not recoverable; printing a `1` there would be a number the
scorer never used.

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
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.846 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.833 |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.833 |

`cited gold` is the only column that uses the labels: the share of answers
citing a document the gold set marks relevant. It fell from 0.900 to 0.792 when
the hard questions were added - the correct direction, since retrieval misses
more of them, so more answers are built from the wrong passages - recovered to
0.857 at fourteen documents, dropped to 0.812 at sixteen, and has recovered
again to 0.833 at eighteen. Four moves, all small, none of them caused by a
change to the answerer: this column tracks the corpus and the gold set, not the
generation code. The grounding and citation columns stayed at 1.000 throughout,
which is the distinction the measure exists to draw - the answerer is not less
faithful when this number falls, it is being handed worse evidence. Note also
that `cited gold` has started varying with k again, dropping 0.846 to 0.833
between k=3 and k=4: with eighteen documents there is once more a question where
the extra passage displaces a gold citation.

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

### Abstention: knowing when there is no answer

Everything above assumes the answer is in the corpus. Ask this engine about
Kubernetes and retrieval still returns five passages, the answerer still builds
a sentence from them, and a citation still gets attached:

```
$ python -m ragkb.cli ask "how does the raft consensus protocol elect a new leader"
It has essentially one hyperparameter, and it requires nothing from the
underlying retrievers beyond an ordered list ...

Sources: rank-fusion
```

That is the worst output the system can produce. It is fluent, it is sourced,
and it is about something else entirely - and the citation makes it *more*
convincing, not less. Faithfulness cannot catch it: the sentence really is in
`rank-fusion`, so it scores a perfect 1.000. Faithfulness asks whether the
answer matches the evidence; nothing yet asked whether the evidence matches the
question.

`ragkb/abstain.py` is that check, and it runs before generation rather than
after - which matters most for the LLM answerer, where it fires before any
tokens are paid for. Three signals, all read off indexes that are already
fitted:

- **vocab coverage** - share of the query's IDF mass whose terms appear
  anywhere in the corpus. Terms the corpus never indexed have no IDF, so they
  are charged the *maximum* observed IDF: an unseen term is by definition
  maximally rare, and charging it zero would let unknown words cost nothing.
- **match coverage** - share of that mass present in the top-ranked chunk. This
  is the signal that separates a query whose words exist in the corpus from one
  whose words exist *together* in a passage.
- **top similarity** - TF-IDF cosine against the top chunk. Cosine is length
  normalised, so unlike a raw BM25 score it is comparable across queries and
  can be thresholded at all.

Measuring this needed a second question set, since a gold set of answerable
questions can only ever measure false alarms. `evals/unanswerable.jsonl` holds
eighteen questions the corpus genuinely does not cover, split deliberately:

- **off-domain** (6) - Kubernetes, Raft, the GIL. Different subject entirely.
- **near-domain** (12) - dense-passage-retrieval training objectives, sharding a
  search cluster, click models. Information retrieval questions this corpus
  happens not to answer.

The split is the point of the file. Off-domain questions are built from words
the corpus never indexed and any threshold catches them; near-domain questions
are assembled almost entirely from vocabulary the corpus knows, so they are the
only rows that say anything. Reporting one pooled number would be flattered by
the easy half.

**This file is a maintenance liability, and growing the corpus proved it.**
`u09` used to ask how SPLADE learns sparse term weights - a textbook
near-domain question right up until `sparse-neural-retrieval.md` was added,
which answers it in a paragraph. It was still labelled unanswerable, and the
eval duly reported the gate "wrongly answering" the highest-scoring near-domain
question in the set. Nothing failed loudly: a mislabelled row just quietly makes
the abstention numbers wrong in the flattering direction. `u09` is now a click
model question, which the corpus genuinely does not cover. Every future corpus
addition has to be checked against this file, because unlike `gold.jsonl` -
where `tests/test_gold.py` asserts that every referenced document exists - there
is no test that can catch a question becoming answerable.

Adding `negative-mining.md` put `u08` - "what loss function is dense passage
retrieval trained with" - under exactly that suspicion, and it was checked
rather than assumed. The new document describes what DPR contrasts against, in
detail, but the strings `loss function`, `softmax` and `cross-entropy` appear
nowhere in the corpus, so the question is still unanswerable on its own terms.
It stays labelled unanswerable, and it is now a *harder* near-domain probe than
it was, which is the file working as designed rather than a row going stale.

```bash
python evals/abstention_eval.py
```

Mean score: **0.449** answerable, **0.262** near-domain, **0.096** off-domain.

| threshold | answer rate | abstain off | abstain near | accuracy | balanced | caught misses | lost hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 1.000 | 0.000 | 0.000 | 0.812 | 0.500 | 0.000 | 0.000 |
| 0.20 | 1.000 | 0.667 | 0.250 | 0.885 | 0.694 | 0.000 | 0.000 |
| 0.24 | 0.949 | 1.000 | 0.500 | **0.896** | **0.808** | 0.125 | 0.043 |
| **0.28** (shipped) | 0.872 | 1.000 | 0.583 | 0.844 | 0.797 | 0.500 | 0.086 |
| 0.30 | 0.833 | 1.000 | 0.667 | 0.823 | 0.806 | 0.625 | 0.114 |
| 0.36 | 0.615 | 1.000 | 0.833 | 0.667 | 0.752 | 0.875 | 0.329 |
| 0.50 | 0.346 | 1.000 | 1.000 | 0.469 | 0.673 | 1.000 | 0.614 |

**The shipped 0.28 was the balanced-accuracy optimum at sixteen documents and is
not the optimum at eighteen.** It is now beaten by 0.24 (0.808 against 0.797) and
by 0.30 (0.806), and it sits in a shallow local dip between them. This is the
same instability the chunking default and the MMR sweet spot showed, arriving one
increment later, and the cause is visible in the mean scores: the answerable mean
barely moved (0.447 to 0.449) while the off-domain mean *fell* from 0.111 to
0.096. A bigger corpus dilutes IDF mass across more documents, which pushes
genuinely foreign queries further down and lets a lower gate separate them.

The default is left at 0.28 rather than re-tuned, for the same reason as
everywhere else in this README: a 0.011 balanced-accuracy difference on 96
questions is one or two questions, and re-tuning a constant every time the
corpus grows is how a project ends up reporting its own selection noise. The
candidate is recorded; `--abstain` takes any threshold.

The last two columns are the ones worth arguing about. Eight of the 78
answerable questions are ones retrieval already misses at k=4 - it returns
chunks, none from a relevant document - and on those the ideal gate abstains
too, because answering from evidence that does not contain the answer is the
same failure whether or not the corpus *could* have answered it. At 0.28 the
gate declines 50% of those misses and 9% of the questions retrieval got right.
Push to 0.30 and it catches 63% of them, at the cost of throwing away 11% of
good retrievals. Neither is obviously correct; which one is depends on what a
wrong answer costs, which is why the gate is off by default and the threshold
is a flag.

```
$ python -m ragkb.cli --abstain 0.28 ask "how does the raft consensus protocol elect a new leader"
The knowledge base does not appear to cover this question, so no answer is
given rather than one built from passages that do not address it.

answerability: abstain (score 0.126 vs threshold 0.280; vocab 0.133, match 0.133, cosine 0.111)
terms absent from the corpus: raft, consensus, protocol, elect, leader
```

The score is printed for answered questions too. A gate you only hear from when
it fires is one you cannot calibrate.

One known weakness, pinned by a test rather than hidden: tokenisation does no
stemming, so a query saying "control" against a corpus that says "controls"
counts that term as unknown and charges it the maximum IDF. It costs the score
without sinking it - `what does the b parameter control in bm25` still scores
0.559 - but it is the same morphological blind spot the character n-gram index
exists to fix for retrieval, and the gate does not yet consult it.

The single worst survivor at 0.28 is still `u08`, *"what loss function is dense
passage retrieval trained with"*, at 0.41. Every content word in it appears in
`embeddings.md` and now `negative-mining.md`, both of which discuss dense
retrieval and its training - the corpus simply never names the loss. No lexical
signal can tell those apart, and saying so is more useful than tuning the
threshold until that one row flips. Its score barely moved when a whole document
about dense-retriever training was added (0.42 to 0.41), which is worth noting:
the gate's near-domain failures are not a coverage problem it could grow its way
out of.

### How much of any of that is real? Error bars on the ablation table

Every table above this line reports a mean over the same 78 questions, and
several of the differences in them are one question wide. That is not a
rhetorical point: at n=78 one question is worth 0.013 of any per-query metric,
so hybrid's 0.594 hit@1 against BM25's 0.562 on the hard slice is *exactly one
question*, and that slice - 32 questions - cannot resolve anything finer than
0.031. Nothing in the repo could distinguish "this lever helps" from "this lever
moved one row", which made the whole evaluation section easy to over-read.
`evals/significance.py` closes that hole:

```bash
python evals/significance.py
```

```
hard slice - 32 questions, one question is worth 0.031
metric    comparison                 A       B    delta              95% CI       p   floor       W/L/T
-------------------------------------------------------------------------------------------------------
hit@1     hybrid vs bm25         0.594   0.562    0.031    [+0.000, +0.094]   1.000   1.000      1/0/31
hit@5     hybrid vs bm25         0.875   0.844    0.031    [+0.000, +0.094]   1.000   1.000      1/0/31
recall@5  hybrid vs bm25         0.750   0.719    0.031    [+0.000, +0.094]   1.000   1.000      1/0/31
ndcg@5    hybrid vs bm25         0.651   0.635    0.017    [-0.003, +0.047]   0.500   0.250      2/1/29
mrr       hybrid vs bm25         0.704   0.682    0.022    [+0.000, +0.059]   0.500   0.500      2/0/30
```

Those five rows are the entire hard-slice result the evaluation section leads
with, and the `W/L/T` column is the honest version of it: hybrid's clean sweep
over BM25 rests on **one question** for three of the five metrics and two for the
others. The `floor` column says the rest - with one discordant pair the smallest
reachable p-value is 1.000, so those rows could not have been significant no
matter how the data fell.

Both procedures are *paired* - each method is scored on the same questions, so
each question acts as its own control. The question-to-question variance dwarfs
the difference between two methods on any one question, and an unpaired interval
would bury the signal in it. `ragkb/significance.py` implements two:

- **Paired bootstrap.** Resample the *questions* with replacement, keeping both
  methods' scores for a question together, and take percentiles of the resulting
  mean differences. Answers "how far would this delta move on a different gold
  set of the same size?"
- **Paired approximate randomization.** Flip a coin per question over which side
  its difference falls on, and count how often chance alone reaches the observed
  gap. Answers "is this delta distinguishable from noise?"

The verdict on the whole evaluation section: **none of it is significant.** Not
one comparison at any k, on any slice, has a 95% interval that excludes zero.
That is the honest headline, and it is more useful than the tables it qualifies.

The `W/L/T` column explains why, and is the number worth reading first. The
three retrievers *agree* on almost every question - the largest disagreement
anywhere in the run is 12 questions out of 78, and several slices are 0/0/32
outright. The comparison's real sample size is not 78, it is the discordant
count, and every tie is a question that carries no evidence either way. Three
rounds of gold-set growth have moved the largest discordant count 8 -> 10 -> 12,
lowering the best reachable p-value from 0.008 to 0.002 to 0.000 and changing no
verdict at any step. That is the measured rate of return on gold-set growth for
this particular claim, and it is slow: the questions being added are ones the
three methods mostly answer the same way, so they add resolution to the means
without adding evidence to the comparisons. Writing questions the retrievers
*disagree* on is a different and harder task than writing more questions.

That leads to the one genuinely non-obvious result here, which the `floor`
column reports directly. With `d` discordant pairs there are only `2 ** d` sign
assignments, at most two of which are as extreme as an all-one-direction
observation, so no p-value below `2 ** (1 - d)` is *reachable*. A row reading
`0/1/55` has a floor of 1.000: a single discordant question can never be
significant, no matter how many resamples are thrown at it, and fewer than six
can never clear 0.05 however large the gold set grows. Printing the floor next
to the p-value is what stops a reader treating `p = 0.5` from four discordant
questions as weak evidence rather than no evidence.

Building that column also caught a real bug in the first draft. The sampled
p-value scattered around the truth and could land *below* its own floor - 0.244
where the exact answer is 0.250 - which reads as slightly stronger evidence than
the data can contain. Since this project lives almost entirely in the regime
where the two methods disagree on three to five questions, `permutation_test`
now enumerates all `2 ** d` assignments exactly whenever `d <= 14` and only
samples above that. The exact path is both correct and faster: the full run
takes 0.8s.

None of this changes a single retrieval number. It changes what may be claimed
about them - and it sets the standard the next increment has to clear, which is
that a bigger gold set (specifically, one with more questions where the
retrievers *can* disagree) is now the measured prerequisite for any retrieval
claim this project wants to make.

### The other bias: every sweep above picked its own winner

`significance.py` asks whether the gap between two *named* systems is real. It
does not touch a second problem that runs through every sweep table in this
README, and that no amount of error-barring a single comparison can reach: the
top row of a sweep was **chosen by looking at the same questions it is scored
on**. The maximum of several noisy estimates is biased upward even when every
configuration is identical, so "the best fusion weighting scored 0.808 hit@1" is
not a prediction about the next 78 questions - it is the best of seven draws
from the same 78.

`evals/holdout.py` measures the size of that bias by nested cross-validation:
split the questions into folds, pick the winner on the training folds, score
that choice on the held-out fold, and repeat over twenty different splits.

```bash
python evals/holdout.py
python evals/holdout.py --slice hard --folds 4
```

```
78 questions (all slice), k=5, 5-fold x 20 splits, seed 0
one question is worth 0.013 of any per-query metric; a held-out fold is 15 questions

sweep           metric    picked          in-sample   held-out   optimism   vs default   stable
-----------------------------------------------------------------------------------------------
rrf_k           hit@1     60                  0.705      0.705     +0.000       +0.000     100%
rrf_k           ndcg@5    60                  0.786      0.786     +0.000       -0.000     100%
rrf_k           mrr       60                  0.793      0.793     -0.000       +0.000     100%
fusion weights  hit@1     ngram=4             0.808      0.795     +0.013       +0.051      65%
fusion weights  ndcg@5    ngram=4             0.854      0.836     +0.018       +0.008       0%
fusion weights  mrr       ngram=4             0.865      0.863     +0.001       +0.036      95%
mmr lambda      hit@1     off                 0.705      0.705     +0.000       +0.000     100%
mmr lambda      ndcg@5    0.9                 0.787      0.785     +0.002       -0.002       0%
mmr lambda      mrr       off                 0.793      0.792     +0.000       -0.000      95%
```

Read `optimism` first, on the `fusion weights / hit@1` row. The weighted-RRF
table earlier in this README reports `ngram=4` at 0.808 against uniform's 0.744
and invites exactly the conclusion it looks like. Scored on questions the choice
never saw, that configuration earns **0.795** - the headline shed 0.013, one
question of the five the sweep appeared to win. That is a much smaller penalty
than the +0.043 the same row carried at sixteen documents, and the `vs default`
column has moved with it: the honest estimate of the whole procedure "sweep seven
weightings, ship the winner" now beats uniform RRF by 0.051, four questions,
where at sixteen documents it beat it by 0.004. This is the first sweep row in
the project that has survived cross-validation with its margin substantially
intact.

Two controls make that number trustworthy rather than merely encouraging, and
they are why the script sweeps three different knobs instead of the one it
wanted to indict:

- **`rrf_k` reports exactly zero optimism and 100% stability** (`vs default` of
  +0.000 on all three metrics). `evals/tune.py` already established that `rrf_k`
  is inert over two lists - it changes nothing, so there is nothing for selection
  to inflate, and the harness agrees. A bias detector that printed a large number
  on a knob known to do nothing would be measuring its own arithmetic.
- **MMR reports near-zero optimism**, because on this corpus every lambda ties or
  loses to `off` on almost every split. Selection cannot inflate a sweep whose
  winner is the default.

The `stable` column is the same finding read a different way, and the two
columns are genuinely independent. `fusion weights / ndcg@5` is 0% stable with
+0.018 optimism: the winner changes with every split *and* the table presents it
as a finding, which is a sample property being reported as a setting. But
`fusion weights / mrr` is 95% stable with +0.001 optimism and +0.036 against the
default - the same knob, a different metric, and a genuinely different verdict.
Reporting one number for "does tuning the fusion weights help" would have
averaged those two into something true of neither.

The reversal worth recording is `ngram only`. At sixteen documents it was the
control the weighted fusion could not beat out of sample on nDCG@5, holding
+0.011 over uniform. At eighteen the holdout picks `ngram=4` on all three
metrics, so the control no longer wins even out of sample. Its stability had
already fallen from 45% to 0% at sixteen; the margin has now gone too. Combined
with the weighted-RRF table above - where `ngram=4` takes nDCG@5 by 0.003 and
loses recall@5 by 0.032 - the conclusion has narrowed rather than flipped: the
fusion has caught its best component on ordering and has not caught it on reach.
Three of the nine sweep/metric pairs still have a negative `vs default` - on
those, the honest estimate of tuning remains worse than shipping the constant.

The scope is limited on purpose and the module says so: folds are resampled from
one small gold set, so `optimism` is itself a noisy estimate, and this cannot
rescue a sweep from a gold set too small to distinguish anything. What it does
is price a cost that the tables had been reporting as zero. Combined with the
`floor` column above, the standard for the next retrieval claim is now explicit:
it has to survive being scored on questions that did not choose it.

### What each stage costs

Every table above answers "did it rank better?". None of them answered "what
did that cost?" - and four stages here are *optional*, which is only a real
choice once both numbers exist. `python evals/latency.py` measures the other
half: per-query wall time for each configuration, on the same 78 gold
questions, pooled over repeats so `p95` is a genuine tail over queries rather
than a mean in disguise.

```
query latency (per query, k=5)          cost against quality (k=5)
stage             median_ms   xbase       hit@1   recall@5    ndcg@5       mrr
------------------------------------------------------------------------------
hybrid (2 lists)       0.13    1.00       0.705      0.872     0.786     0.793
hybrid (3 lists)       0.30    2.38       0.744      0.910     0.828     0.827
  + mmr 0.7            1.25    9.86       0.705      0.872     0.786     0.790
  + expansion          1.36   10.65       0.564      0.891     0.742     0.704
  + rerank             1.55   12.18       0.744      0.885     0.808     0.820
  + all three          3.95   31.01       0.692      0.872     0.784     0.785
```

The ordering is the finding, and it is not the one the build order implies. The
cheapest addition in the pipeline is also the best value: the character n-gram
third list costs 2.4x a plain fused query and improves every accuracy column at
once. Re-ranking now *ties* it at rank one (0.744 both) while costing five times
as much and giving back recall - at sixteen documents re-ranking was ahead here,
and the third list has caught it. Expansion at these settings is sharply negative
at rank 1, which the sweep table already said. Stacked together the three optional
stages cost 31x and land *below* the 2.4x row on every column. If this were a
latency budget rather than a portfolio project, the recommendation would be one
line: ship the third list, keep re-ranking behind a flag, leave the rest off.

The `xbase` multipliers moved back toward the previous run (MMR 11x -> 10x, the
full stack 41x -> 31x) while the absolute milliseconds rose. That is the
denominator moving, not the numerator: a bigger corpus made the base two-list
query slower, while the fixed per-query cost of MMR and re-ranking barely
changed, so the ratio narrowed. It is a good illustration of why a ratio needs
its base stated - and why these multipliers should keep shrinking as the corpus
grows.

MMR is the one row where losing accuracy is not a failure - it is spending
relevance on coverage deliberately, and its own table measures the coverage it
bought. What is new here is the price tag on that trade.

Two things the harness surfaced that reading the code had not:

* **`method="bm25"` does not skip the other lists.** The engine records
  provenance ranks for every list on every query, so the flag chooses which
  ranking to *return*, not which work to do. The single-list rows in the table
  therefore time the index's own `search` directly - timing the engine would
  have produced a "bm25 only" row costing the same as the fusion, which is
  true of the engine and false of BM25.
* **The n-gram index dominates build time**, at ~21.6ms against ~5.9ms for both
  word indexes combined. Its query cost is proportionate, so the third list is
  a consistent tax at both ends rather than a cheap index that is expensive to
  query.

The honest caveat is scale: 18 documents and 80 chunks is small enough that
these ratios describe *this* corpus, not a curve. Some of them move predictably,
and now there is a second data point confirming it - MMR's work is O(k²) over a
fixed `k` and so barely grows, while first-stage retrieval grows with the corpus,
so MMR's multiplier was predicted to shrink and did (10.85x at sixteen documents,
9.86x at eighteen). Absolute milliseconds are this machine's; the `xbase` column
is the part that travels, and even it has a domain.

### Inverting the ranking loop: what the postings list actually bought

The timing table above priced the optional stages. It also exposed something
about the stage that is not optional: both indexes ranked by scoring *every*
document against *every* query term. Adding a postings list - `term -> the
documents containing it, with their weights` - is the textbook fix, and
measuring it split into two findings that point in opposite directions.

The obvious win is not the win. Skipping documents that contain no query term
is exact, because BM25 and a sparse cosine are both sums of per-term
contributions and those documents score precisely 0.0. It is also nearly
useless on real questions:

```
inverted index work per query set (counts, not milliseconds)
index               docs     reach        scan        taat   taat/scan
----------------------------------------------------------------------
bm25                  80     97.0%       84564       22209        0.26
tfidf                 80     97.0%       75264       22209        0.30
ngram(4)              80     99.9%      330554       84581        0.26
```

`reach` is the share of the collection a gold question touches through at least
one term: 97.0%. One common word in a question - *what*, *does*, *the* - matches
almost every chunk on its own, so the entire prize for filtering by membership
is the remaining 3.0%. For the n-gram index it is 0.1%, because a 4-gram query
reaches everything. Nor does that improve with scale - it keeps getting *worse*,
rising 96.0% at 57 chunks, 96.5% at 69, 97.0% at 80, which is the prediction
confirmed three times: *the* appears in nearly every document at 80 chunks and at
80,000. The `taat/scan` ratio, meanwhile, has not moved off 0.26 across all three
corpus sizes - a count-based number behaving like a count-based number.

The work was somewhere else. Scoring document-at-a-time costs one dictionary
lookup per (reached document, query term) pair *whether or not the document
contains the term* - 96% of the collection times every word in the question,
and most of those lookups miss. Walking postings term-at-a-time instead, and
accumulating each document's running score, costs one update per posting that
actually exists: the sum of the query terms' document frequencies. That is the
`taat/scan` column, and it is the durable number in this section because it is
a count rather than a timing - about a quarter of the lookups, on every index.

The wall clock agrees, at 10 repeats over the (then) 56 gold questions:

| stage | before | after | speedup |
|---|---|---|---|
| `bm25 index` query | 0.08 ms | 0.05 ms | 1.6x |
| `tfidf index` query | 0.07 ms | 0.03 ms | 2.3x |
| `ngram index` query | 0.22 ms | 0.08 ms | 2.8x |
| `hybrid (2 lists)` | 0.17 ms | 0.10 ms | 1.7x |
| `hybrid (3 lists)` | 0.40 ms | 0.19 ms | 2.1x |
| `fit bm25` (one-off) | 0.74 ms | 1.37 ms | 0.5x |

The last row is the price: building the postings list makes fitting BM25 about
1.8x slower, and TF-IDF slightly so. That is the trade an inverted index is -
pay once at build time, collect on every query - and the persistence layer
already existed to make the build cost a one-off. The postings list is
deliberately *not* written into the snapshot: it is a transposition of data the
snapshot already holds, so it is rebuilt on load rather than doubling the file.

Retrieval quality is unchanged, and that is a claim with evidence rather than a
hope. `evals/evaluate.py` produces byte-identical output before and after, and
`tests/test_postings.py` reconstructs the document-at-a-time scan and compares
against it on every index and every gold question. Two strengths of claim are
asserted separately, because two are true: BM25 accumulates in query-term order
either way, so its scores are compared with `==`; the TF-IDF path's
`similarity` iterates whichever vector is shorter and may sum the same products
in another order, so its ranking is compared exactly and its scores to within
floating point. Overstating the second as bit-equality would have been easy and
wrong.

What this does *not* do is change the complexity class. Term-at-a-time still
touches every posting of every query term, so a common word still costs a walk
proportional to the collection. Getting past that means not traversing postings
whose best possible contribution cannot reach the current top-`k` - WAND or
MaxScore - which is a real next step rather than a tweak, and is not
implemented here.

### Robustness: the one condition the third list was added for

Every table above this line scores queries that were typed correctly. That is a
convenient assumption and a false one, and it happens to test the character
n-gram list under precisely the condition it was never expected to help in. The
justification for that third list, written into `ragkb/retriever.py` long before
any of this was measured, is that BM25 and TF-IDF tokenise into whole words and
therefore *"both miss the same misspelling and the same morphological
variant"*. That is a falsifiable claim about noisy input. The evidence for it
was one unit test.

`ragkb/perturb.py` corrupts a query the way a hand does: the four
Damerau-Levenshtein edit types, with substitutions drawn from physically
adjacent QWERTY keys rather than from the alphabet, at a budgeted character
error rate. Three properties are load-bearing and each is pinned by a test.
Rate 0.0 is *exactly* the identity, so the clean row is a baseline rather than
a lucky draw - it reproduces the `evaluate.py` numbers to three decimals.
The stream is derived from the query text with a keyed BLAKE2b digest, not from
a counter walked across the file and not from the builtin `hash`, which is
salted per process: appending a gold question must not re-roll the noise applied
to the questions already there, or every corpus increment silently invalidates
the previous robustness numbers. And the edit count is a budget rather than
per-character coin flips, so two methods labelled "10%" were handed the same
amount of damage instead of two samples of a random variable.

```bash
python evals/robustness.py
python evals/robustness.py --rates 0.05,0.15 --seeds 15
```

Realised noise trails the knob, because tokens under four characters are never
touched and the budget rounds down; the script reports what it achieved rather
than what it asked for. At a requested 20% the realised rate is 0.129.

nDCG@5, 78 questions, averaged over 7 noise draws per rate:

| method | clean | 5% | 10% | 20% | retained at 20% |
|---|---|---|---|---|---|
| `bm25` | 0.785 | 0.724 | 0.650 | 0.500 | 63.6% |
| `vector` | 0.781 | 0.727 | 0.665 | 0.508 | 65.1% |
| `ngram` | 0.851 | 0.852 | 0.844 | 0.800 | **93.9%** |
| `hybrid2` | 0.786 | 0.736 | 0.661 | 0.509 | 64.8% |
| `hybrid3` | 0.828 | 0.774 | 0.732 | 0.595 | 71.8% |

The design claim survives contact with a measurement, and by a wide margin. The
two word-level indexes lose better than a third of their ranking quality, and
they lose it *together* - 63.6% and 65.1%, which is what "miss the same
misspelling" looks like when you put a number on it. Fusing them changes
nothing (64.8%): reciprocal rank fusion can only reorder what its inputs
retrieved, and two lists that failed for the same structural reason have no
disagreement for it to arbitrate. The character index barely notices, holding
93.9%, because a one-character typo leaves most of a word's 4-grams intact.
hit@1 tells the same story more bluntly: at 20% noise BM25 answers 36% of
questions correctly at rank one and the n-gram index answers 70%.

**And then the fusion undoes most of it.** Adding the third list lifts retention
from 64.8% to 71.8%, which reads like a win until you compare it against the
third list on its own: 93.9%, at 0.800 nDCG@5 against the fusion's 0.595. Under
noise the two brittle lists do not merely fail to help - they outvote the list
that is still working. This is the `ngram only` control that `evaluate.py`
already flags on clean text, failing here far more loudly, and it is the second
independent axis on which the three-way fusion loses to its own third list. The
script prints that comparison as a computed verdict rather than as prose, so it
cannot drift out of agreement with the table when the corpus grows.

Two things this does not establish. The seed spread (0.02-0.04 nDCG) bounds the
variance from *drawing the noise*, which is the only uncertainty this script
measures; the sampling error over 78 questions is a separate and larger one, and
`evals/significance.py` is not yet wired into this harness - a real gap rather
than a rounding detail. And "robust" here means robust to typing errors only.
Vocabulary mismatch and paraphrase are a different failure mode with a different
lever, which is what the query-expansion table above measures, and nothing here
transfers to it.

The uncomfortable reading is the useful one. On this corpus the character n-gram
index wins clean *and* wins noisy, and the fusion's job has quietly become
protecting two lists that lose to it on both axes.

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

The format is at version 2. It was bumped when the optional n-gram index
joined the payload, which was a choice worth making explicit: the new key could
have been read with a default and the version left alone, but then code
predating it would load a three-list index, find no n-gram block, and answer
from a two-list fusion while the stored config said otherwise. Refusing to load
and asking for a rebuild is the cheap failure; the silent success is the
expensive one.

Measured on this corpus (18 documents, 80 chunks), loading is faster than a cold
build, and the gap grows with corpus size since fitting scales with the
collection while loading is a single JSON parse. Rankings, scores and answers are
bit-for-bit identical either way - `tests/test_store.py` asserts exactly that.
The measured multiplier has come out at 2.4x, 1.5x and 1.7x on three successive
corpus sizes, measured the same way each time, which is worth reading as the
precision this ratio has rather than as a trend: all three are single
measurements of a build that takes about ten milliseconds.

### Growing the index: what "incremental" is actually worth

Persistence removes the cost of rebuilding an index that did not change. The
other half of the same problem is an index that changed *slightly* - one new
document arriving at a process that already holds the other seventeen. Until now
that was a full rebuild, because the engine had exactly two ways to exist: built
from a corpus, or restored from a snapshot.

`RAGEngine.add_document` adds a third. It is worth stating up front that it is
only half incremental, and that the half which is not is the expensive half.

BM25's state is additive. Term-frequency counters and document lengths are
per-document and append; document frequency is a running count; the postings
list gains one entry per (new chunk, term) pair and keeps its ascending order
because new chunks take the highest indices. Two quantities are
collection-wide, and neither needs the old documents to update: `avg_doc_length`
is a sum already being maintained, and IDF is a *scoring coefficient* applied at
query time, so refreshing all of it is one pass over the vocabulary.
`BM25Index.add_documents` therefore never re-tokenises, re-counts or re-scans a
single indexed document.

TF-IDF cannot do this, and the reason is structural rather than an omission.
Its IDF is not a coefficient applied later - it is baked into the stored data.
Each vector holds `tf * idf` and is then L2-normalised, so one new document
changes `doc_count`, changes every IDF, and therefore changes both the weights
*and* the length of every vector already stored, including vectors for documents
sharing no term with the new one. Rebuilding them is the same traversal as
fitting them. An `add_documents` on `TfidfIndex` could only ever save the
tokenisation, at the cost of keeping a second copy of the raw counters - so
there deliberately is not one. An API named for an incremental update that
performs a full refit is a promise about cost the implementation does not keep.

`python evals/incremental.py` measures the asymmetry instead of asserting it.
Holding out the document that sorts last and adding it back:

```
stage                           median_ms      p95_ms     mean_ms           n     xbase
---------------------------------------------------------------------------------------
cold: build all 18 docs              9.09       10.37        9.27           5      1.00
cold: build 17 then add 1           13.52       14.89       13.67           5      1.49
warm: add 1 to a live index          5.85        8.65        6.31           5      0.64
```

That is one run, not an average of several. Across repeated runs on this machine
the cold-incremental multiplier lands between 1.3x and 1.6x and the warm one
between 0.59x and 0.68x, which is the precision these numbers actually have -
a build this short is a handful of milliseconds sitting on top of scheduler
noise. The sign of each result is solid; the second decimal place is not. Both
multipliers have now held their ranges across three corpus sizes - 14, 16 and 18
documents, landing at 1.49x and 0.63-0.64x each time - which is more than a
stability check this cheap had any right to establish.

Two different answers, and the difference is what you start from.

From cold the incremental path **loses, by about half again**. Reaching eighteen
documents by way of seventeen pays the TF-IDF fit twice - once at seventeen, once
at eighteen - and that refit is the largest single stage in the build (3.41 ms
of it, against 1.90 ms for BM25 and 1.91 ms for chunking). If you do not already
have an index, build one; there is no version of this where assembling a corpus
document by document is the fast way.

From warm - a running process holding an index and handed one new document,
which is the only case the method exists for - it costs roughly **0.64x a full
rebuild**, against a corpus rebuild that is itself only ~9 ms. The saving is
the tokenising and counting of the seventeen documents already indexed, and it is
real but bounded, because the TF-IDF refit stays in the bill.

That framing is also the honest scaling statement, and it does not flatter the
feature: the saving is a *fixed* cost removed (one pass over the documents
already indexed, which is what add avoids re-doing) set against a cost that
*grows* with the collection (rebuilding every vector in it). The 0.64x is not a
floor that improves with scale - it is a number that gets worse as the corpus
grows, until the refit dominates and the warm row converges on 1.00x. Three
corpus sizes is still not a curve, and the multiplier has not visibly moved
across them, which is what the argument predicts at this scale: at 18 documents
the fixed saving is still large relative to the growing refit. The claim is about
which way the curve points, not about where the bend is.

The correctness claim is the part that has to be exact, and it is checked
against real from-scratch builds rather than against a re-derivation of the same
arithmetic. All 78 gold questions, ranked by both a grown engine and a whole
one, across every method, are identical position for position, and
`tests/test_incremental.py` pins the index state itself - counters, lengths,
document frequencies, IDF and postings - equal to what a single `fit` produces.
Eval numbers are unchanged, as they must be: the index is the same index.

Two limitations are deliberate rather than pending.

**Chunks are appended, not merged into corpus order.** That is the whole point.
Keeping the chunk list sorted by document id would renumber the existing chunks,
and every posting in every index refers to a chunk by its index, so a sorted
insert is a refit wearing a different name. The visible consequence is that a
grown engine holds its chunks in insertion order. Scores are unaffected, but
ranking ties break on the chunk index, so a tie between two equally scored
chunks can resolve differently than in a from-scratch build over the same
documents. The tests hold the added document to the one that sorts last, where
the two orders coincide and the rankings are required to match exactly - which
is a way of saying the tie-break caveat is a real caveat and not a hedge.

**There is no `remove_document` or `replace_document`.** Removal renumbers every
chunk after the removed one, which no postings list here survives. Growing the
index is supported; editing it is a rebuild. `add_document` raises on a
duplicate id rather than replacing, because the alternative is a method that
silently indexes one document twice under one name.

## Browser demo

`ragkb serve` starts a small read-only HTTP demo so the engine can be poked at
without the CLI:

```bash
python -m ragkb.cli index --out index.json
python -m ragkb.cli --index index.json serve          # http://127.0.0.1:8000
```

| route | returns |
| --- | --- |
| `GET /` | a single static HTML page: query box, answer, citations, ranked passages |
| `GET /api/ask?q=&k=&method=` | the extractive answer, its citations, and the abstention report when the gate is on |
| `GET /api/search?q=&k=&method=` | the ranking, each result's fused score and per-list ranks, and the redundancy of the set |
| `GET /api/stats` | document, chunk and vocabulary counts |

The obvious way to build this is FastAPI or Streamlit, and the obvious reason
not to is that the repository's whole claim is that it has no dependencies. A
demo that makes `pip install` a precondition for running the project quietly
retracts that claim, so it is written on `http.server` instead. That trade is
worth stating rather than hiding: `ThreadingHTTPServer` gives one thread per
request with no worker pool, no authentication and no rate limiting, which is
why it binds to `127.0.0.1`, routes only `GET`, and caps `k` at 25. It is a way
to see retrieval behave, not a serving tier.

Two details are deliberate. The search endpoint returns *provenance*, not just
a ranking — each result carries the ranks it held in the BM25, TF-IDF, n-gram,
MMR and re-ranking stages, because the interesting thing to show is why hybrid
put something first, and stages that did not run are omitted rather than
returned as nulls. And the HTML page is entirely static: every dynamic value is
inserted with `textContent`, so no user input is ever interpolated into markup.
`tests/test_serve.py` asserts that (`innerHTML` appears nowhere in the page)
rather than leaving it as a comment.

The engine is fitted once at startup and shared across request threads.
Retrieval only reads fitted state, so the sharing is safe, and a request costs
a query rather than a re-index — pair it with `--index` and startup is a JSON
parse.

## Design notes

- **Sentence-aware chunking with overlap.** Chunks never end mid-sentence, and
  each chunk repeats the tail of the previous one so ideas that straddle a
  boundary stay retrievable from both sides.
- **Rank fusion rather than score blending.** BM25 scores and cosine
  similarities have no common scale, and their ranges move per query. RRF only
  needs the ordering. What it does *not* have is any notion of which list is
  better: every list gets one vote, so fusing a strong list with weak ones
  gives back part of the strong one. The n-gram table is that limitation
  measured rather than asserted.
- **Two scores over one representation are not two retrievers.** BM25 and
  TF-IDF share a tokenizer, so they share a blind spot, and no amount of fusing
  reaches into it. The single largest accuracy gain in the project came from
  changing what a document *is* - characters instead of words - rather than
  from changing how it was scored or how the scores were combined.
- **An extractive answerer that cannot hallucinate.** It returns sentences
  copied verbatim from retrieved chunks, so the failure mode is an unhelpful
  answer rather than an invented one. That isolates retrieval quality from
  generation quality during evaluation.
- **Ablations by default.** `evals/evaluate.py` always prints BM25, vector and
  hybrid side by side, because a hybrid that does not beat its components is
  not earning its complexity.
- **No unexamined constants.** Chunk size, overlap and `rrf_k` sat unswept
  underneath every other table until `evals/tune.py` measured them. Two of the
  three turned out to be defensible and one turned out to be inert; both are
  results, and neither was knowable by inspection.
- **Diversity measured, not assumed.** MMR ships with a redundancy read-out and
  a lambda sweep rather than a tuned default, because "more diverse" is only
  worth having if you can show what it cost.
- **Re-ranking is a stage, not a scorer.** `CrossEncoder` is a two-method
  interface over the candidate pool, so the lexical implementation here can be
  swapped for a trained model or a hosted API without the retriever noticing.
  The pipeline order - fuse, expand, re-rank, then diversify - is deliberate:
  MMR trades relevance for coverage, and it can only do that honestly if the
  relevance it is handed is the best one available.
- **Optional machinery has an identity setting.** MMR at `lambda = 1.0`,
  expansion at `alpha = 1.0`, re-ranking at `alpha = 1.0` and the n-gram list
  at `ngram = None` all have to reproduce the plain ranking exactly.
  That is the cheapest correctness check available for a component whose output
  is otherwise only judged by a metric that moves for many reasons.
- **A sweep that picks its own winner has to pay for it.** Error bars on a
  comparison between two named systems are not error bars on the *choice* of
  which system to name. `ragkb/holdout.py` scores the procedure rather than the
  configuration - pick on the training folds, score on the held-out ones - and
  the price it puts on the largest sweep in the README fell from 0.043 hit@1 at
  sixteen documents to 0.013 at eighteen, as the gold set grew enough for the
  effect to outlast the selection. The two knobs where nothing is being selected
  report ~0.000 at every corpus size, which is how the number earns belief.
- **A score you cannot decompose is a score you cannot debug.** Both retrievers
  and the fusion are additive, so `ragkb/attribution.py` reconstructs any
  single score from its summands exactly and the tests assert that identity.
  Aggregates say a change helped; only the decomposition says which word did it,
  and whether the miss was fixable at all.

## Tests

```bash
pytest -q      # 453 tests
```

The suite covers tokenisation and chunk boundaries, BM25 saturation and length
normalisation, TF-IDF normalisation, RRF scoring shape, every metric, the CLI,
MMR selection order and its redundancy invariants, the query-expansion model and
its weighted-query interfaces, the cross-encoder features and blending
(including that proximity separates arrangements bag-of-words cannot),
answer grounding and citation accuracy, gold-set integrity, the index-time
sweep (that its default row reproduces the shipped engine exactly, that changing
the chunk target really rebuilds the index, and that changing `rrf_k` really
does not), the character n-gram analyzer (boundary markers, short tokens, and
the case that justifies the whole module - a morphological variant scoring
above zero where the word index provably scores exactly zero), that leaving the
n-gram list out fuses exactly two components, that a three-list index survives
a save/load round trip as a three-list index, the failure analysis (that a
question the method answers is never reported as a failure, that a missed
document is reported at its true rank rather than a truncated one, and that a
lever is credited only when it actually reaches inside k), the paired
significance tests (that identical systems are never called different, that a
system winning every question is, that a seed reproduces an interval exactly,
that the pairing is genuinely used rather than the two vectors being treated as
independent samples, and - the regression guard for the bug that produced the
exact-enumeration path - that a small comparison's p-value lands exactly on its
own floor rather than below it), the timing harness (asserted on its arithmetic
and its failure modes, never on a duration - a benchmark that fails when the
machine is busy is a flaky test, not a measurement), the score attribution
(that every decomposition sums back to the scorer it decomposes, that `missing`
and `not in the corpus` stay distinct diagnoses, that the reported rank is the
one `search` actually returned, and that an unretrieved chunk still reports its
real position rather than a blank), the inverted index (that ranking
term-at-a-time over postings reproduces the document-at-a-time scan it
replaced - bit-for-bit for BM25, exactly in rank order for the cosine paths -
on every index and every gold question, that postings are rebuilt rather than
persisted, and a pinned measurement of the unflattering half of that result:
that filtering by membership alone would have saved almost nothing), the
cross-validated selection harness (that the folds are a genuine partition rather
than a sample and are shuffled rather than sliced, that identical configurations
produce *exactly* zero optimism and a uniformly dominant one produces zero while
winning every fold, that ties resolve towards the caller's first configuration
so a default never moves for free, that the held-out mean covers every question
once rather than averaging unequal folds, and - the test the module exists to
pass - that twelve configurations of pure coin flips are detected as an inflated
in-sample maximum with unstable winners), incremental indexing (that extending
BM25 reproduces a single `fit` term for term and posting for posting, that
adding one document changes the IDF of terms it does not contain, that an engine
grown one document at a time returns rankings identical to a from-scratch build
over the same corpus - with and without the n-gram list - and that the grown
index still round-trips through a save/load), query corruption and the
robustness harness (that rate 0.0 is byte-for-byte the identity so the clean row
is a baseline rather than a lucky draw, that a query's corruption depends on its
text and seed alone so appending a gold question cannot re-roll the noise on the
ones already there, that the stream is a keyed digest rather than the
process-salted builtin `hash`, that damage lands only inside long alphabetic
runs and never deletes a word out of existence, that each operator alone moves
length in the one direction it can, that a substitution stays on a physically
adjacent key, that a requested rate is never overshot, and - the test that
caught a real measurement bug - that a transposition costs one edit rather than
the two plain Levenshtein would charge, which had been inflating every realised
noise rate the harness reported), and an end-to-end pass over the real corpus
and gold set.

One of those tests exists because a bug shipped into a draft: `--ngram` was
first declared with argparse's `nargs="?"`, which quietly consumes the
*subcommand* as its value, so `--ngram search ...` died with "invalid int
value: 'search'" and the flag only worked when it happened to be last on the
line. `tests/test_cli.py` now asserts the flag survives having a subcommand
after it.
