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
| `ragkb/significance.py` | paired bootstrap intervals and randomization tests for metric differences |
| `ragkb/cli.py` | `stats`, `search`, `ask`, `explain`, `index`, `--mmr`, `--expand`, `--rerank`, `--ngram`, `--abstain`, `--index` |
| `evals/tune.py` | index-time sweep: chunk size, chunk overlap, `rrf_k` (2 or 3 lists) |
| `evals/failures.py` | per-question failure analysis: where a missed document really ranks, and which lever recovers it |
| `evals/significance.py` | error bars on the ablation table: which of those differences survive a paired test |

`corpus/` holds the sample knowledge base (fourteen documents on information
retrieval). `evals/gold.jsonl` holds fifty-six labelled questions with the
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

**chunking, all 56 questions, k=5** (deltas against the shipped default)

| target:overlap | chunks | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 60:15 | 150 | 0.643 | 0.875 | 0.964 | 0.920 | 0.788 | 0.767 |
| 80:20 | 91 | 0.679 | 0.875 | 0.946 | 0.893 | 0.793 | 0.782 |
| 100:25 | 68 | 0.661 | 0.929 | **0.982** | **0.929** | 0.805 | 0.791 |
| 120:0 | 46 | 0.625 | 0.839 | **0.982** | **0.929** | 0.784 | 0.754 |
| 120:15 | 55 | **0.696** | 0.929 | 0.946 | 0.893 | 0.803 | **0.798** |
| **120:30** (default) | 57 | **0.696** | 0.929 | 0.946 | 0.893 | 0.799 | 0.792 |
| 120:60 | 69 | 0.679 | 0.893 | 0.964 | 0.911 | 0.801 | 0.790 |
| 150:40 | 44 | 0.679 | 0.911 | **0.982** | **0.929** | **0.814** | 0.796 |
| 180:45 | 38 | 0.661 | 0.911 | 0.964 | 0.911 | 0.795 | 0.783 |
| 200:50 | 31 | 0.643 | 0.929 | 0.946 | 0.902 | 0.785 | 0.772 |

**Overlap earns its keep, and the reason is not the one the design note gave.**
`120:0` is the same chunk size with the overlap switched off, and it is the
worst row in the table at rank one: hit@1 falls 0.696 -> 0.625 and hit@3 0.929
-> 0.839, the largest early-ranking loss any setting produces. That is the
design claim confirmed - an idea straddling a boundary really is retrievable
from both sides, and dropping overlap really does cost the questions that
depend on it.

But `120:0` simultaneously *gains* reach: hit@5 and recall@5 both rise 0.036, to
the best values in the table. That is not a contradiction, it is the redundancy
problem measured from the other end. Overlap manufactures near-duplicate chunks,
near-duplicates crowd each other into the top five, and a top-five spent on two
views of one passage is a top-five that did not reach a second document. The MMR
section treats that as a re-ranking problem; this table shows it is also a
chunking one, and the two dials fight over the same 0.036.

**The default is defensible but it is a choice, not a maximum.** Nothing beats
120:30 at rank one - 120:15 ties it - which is the column it was implicitly
tuned for. It is beaten on every reach column. `150:40` is the strongest
alternative: it matches the best hit@5 and recall@5 in the table, takes nDCG@5
outright (0.814 vs 0.799), edges MRR, and gives up only hit@1 and hit@3. In
absolute terms that trade is two questions won at k=5 against one lost at rank
one, on 56 questions - too thin to call a win, and adopting it would invalidate
every other table in this README, all of which were measured at 120:30. It is
recorded here as the candidate it is rather than quietly shipped.

The extremes fail as expected and confirm the sweep is measuring something real.
`60:15` cuts the corpus into 150 chunks averaging 50 tokens and loses 0.054 of
hit@1: chunks too short to contain a whole answer. `200:50` gives up 0.054 of
hit@1 in the other direction, where a 173-token chunk dilutes the matched terms
with surrounding text. The useful range is a broad plateau between roughly 100
and 150 target tokens, which is worth knowing precisely because it means the
exact value inside that band is not what any other result here depends on.

**`rrf_k`, all 56 questions, k=5.** The `changed` column counts questions whose
top-5 ordering differs from the default - the measurement that separates
"equally good" from "does nothing at all".

| rrf_k | hit@1 | hit@5 | nDCG@5 | MRR | changed |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.696 | 0.946 | 0.801 | 0.795 | 5 |
| 5 | 0.696 | 0.946 | 0.799 | 0.792 | 2 |
| 10 | 0.696 | 0.946 | 0.799 | 0.792 | 1 |
| 20 | 0.696 | 0.946 | 0.799 | 0.792 | 0 |
| **60** (default) | 0.696 | 0.946 | 0.799 | 0.792 | 0 |
| 200 | 0.696 | 0.946 | 0.799 | 0.792 | 0 |
| 1000 | 0.696 | 0.946 | 0.799 | 0.792 | 0 |

`rrf_k` is inert on this corpus, and the `changed` column is what proves it:
every value at or above 20 returns byte-identical top-5 orderings on all 56
questions. This is not "tuning did not help", it is "there was nothing to tune",
and those are different claims that an accuracy table alone cannot tell apart.

The reason is structural. `rrf_k` sets how steeply `1 / (rrf_k + rank)` decays,
which interpolates between two known fusion rules: as `rrf_k` approaches zero
the first list to rank something highly dominates, and as it grows large every
term flattens toward `1/rrf_k` and the ordering converges on the sum of ranks -
Borda count. With only *two* lists that already agree on most questions, both
ends produce the same order. `rrf_k = 1` is the only value that moves anything,
touching 5 questions for +0.003 MRR, which on 56 questions is noise. The knob
would start to matter with more retrievers in the fusion, or with retrievers
that disagree more than BM25 and TF-IDF do here - and the honest version of that
claim is that this corpus cannot demonstrate it.

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

### Re-ranking: a second stage that reads the pair

Fusion recalls far better than it precisely ranks. On the full gold set hit@5 is
0.946 but hit@1 is only 0.696: for a quarter of the questions the answer is
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
| off | 0.696 | 0.929 | 0.946 | 0.799 | 0.792 |
| 1.0 | 0.696 | 0.929 | 0.946 | 0.799 | 0.792 |
| 0.8 | 0.696 | 0.911 | 0.946 | 0.801 | 0.795 |
| 0.6 | 0.714 | 0.893 | 0.946 | 0.808 | 0.808 |
| **0.5** | **0.750** | 0.911 | **0.946** | **0.822** | **0.826** |
| 0.4 | 0.750 | 0.893 | 0.946 | 0.820 | 0.824 |
| 0.2 | 0.768 | 0.893 | 0.929 | 0.823 | 0.838 |
| 0.0 | 0.750 | 0.911 | 0.946 | 0.822 | 0.835 |

And on the hard slice, where the first stage has the most room:

| alpha | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.500 | 0.864 | 0.909 | 0.636 | 0.653 |
| 0.5 | 0.636 | 0.864 | 0.909 | 0.692 | 0.739 |
| 0.2 | 0.682 | 0.864 | 0.909 | 0.719 | 0.782 |

This is the largest single-increment gain the project has measured: at
alpha=0.5, hit@1 goes 0.696 -> 0.750 and MRR 0.792 -> 0.826 with hit@5
completely unchanged, and on the hard slice hit@1 goes 0.500 -> 0.636. The
mechanism is exactly the one predicted - it converts recall the first stage
already had into precision at rank one.

Read the trade-off honestly. Leaning harder on the second stage (alpha <= 0.2)
buys another 0.018 hit@1 and gives back hit@5 (0.946 -> 0.929): a re-ranker
reorders the pool, it cannot enlarge it, so pushing it too far shuffles a
correct answer out of the top five. The easy slice barely moves at rank one
(0.824 throughout - it is saturated) and loses a little at hit@3/hit@5, which is
the same story: positional features help most where the wording differs. alpha
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
reason the `rrf_k` sweep found nothing to tune: the two lists agree on 0.854 of
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
0.854 to 0.693, which is the mechanism working: the third list is contributing
a genuinely different opinion rather than a paraphrase of an existing one.
`hybrid2` is the shipped two-list fusion, `ngram` is the n-gram list alone and
`hybrid3` is all three fused.

**all 56 questions, k=5**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid2 | 0.696 | 0.929 | 0.946 | 0.893 | 0.799 | 0.792 |
| **ngram** | **0.768** | **0.964** | **0.982** | **0.973** | **0.868** | **0.849** |
| hybrid3 | 0.714 | 0.911 | 0.982 | 0.938 | 0.834 | 0.821 |

**hard slice (22 questions, 10 with two relevant documents)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid2 | 0.500 | 0.864 | 0.909 | 0.773 | 0.636 | 0.653 |
| **ngram** | **0.636** | **0.909** | **0.955** | **0.932** | **0.765** | **0.752** |
| hybrid3 | 0.545 | 0.818 | 0.955 | 0.841 | 0.698 | 0.708 |

**The result is the largest this project has measured, and it is not the result
the experiment was designed to find.** The n-gram list was added as a third
input to the fusion. It beats the fusion - alone, on every column, on every
slice. Against `hybrid2` on the hard questions it takes hit@1 from 0.500 to
0.636 and recall@5 from 0.773 to 0.932, which is three and a half of
twenty-two questions where the answer was simply not in the top five before.

**And `hybrid3` lands between its inputs, which is the finding worth keeping.**
Fusing the strong list with the two weaker ones is *worse than the strong list
alone* on hit@1, hit@3, recall@5, nDCG@5 and MRR - it only ties on hit@5. That
is exactly where an unweighted average of unequal inputs should land, and it is
the limit of RRF stated plainly: `1 / (rrf_k + rank)` gives every list the same
vote regardless of how good that list is, so a third opinion helps when the
lists are comparable and dilutes when they are not. The project's working
assumption from the first commit - that adding retrievers to the fusion is how
retrieval gets better - is the thing this table refutes. What made retrieval
better was changing the representation; the fusion then gave part of it back.

Two cautions before anyone reads this as "character n-grams beat words". The
first is arithmetic: 56 questions, so the 0.072 gap in hit@1 is four questions,
and the hard slice is 22, where one question is 0.045. The second is
structural, and it matters more. The mechanism that wins here - partial credit
for shared substrings - is the same mechanism that should *lose* on a corpus of
lexically similar documents, because n-grams blur distinctions that whole words
preserve. Fourteen documents on visibly different topics is close to the
friendliest case this idea will ever get, and the two adjacent-topic documents
added earlier (`late-interaction.md` beside `reranking.md`) are the only
pressure on it in the whole corpus. The honest claim is "on this corpus,
character 4-grams retrieve better than words", and the experiment that would
promote it to something stronger is a corpus with far more near-neighbours in
it.

It is therefore **off by default** and exposed as `--ngram N`, on the same
principle as `--expand` and `--rerank`, and for the same reason `150:40`
chunking was recorded rather than shipped: every other table in this README was
measured against the two-list fusion, and quietly changing the default would
invalidate all of them. Unlike those flags it is index-time, not query-time -
it fits a third index - so a saved index cannot gain one by flag, and `--ngram`
against an index built without it fails loudly rather than answering from a
two-list fusion nobody asked for.

The cost is real and small in absolute terms: 3,924 gram features against 1,345
word features (2.9x), a 0.4 MB index file becoming 1.0 MB, and fitting slowing
from 0.004s to 0.014s on this corpus. Denser vectors are also why the list is
noisier per-match - most English 4-grams are common, so everything matches
everything a little - which is what makes the standalone result surprising
rather than obvious.

### Weighted RRF: acting on "the third list is the best list"

The n-gram section ends on an uncomfortable number. The character n-gram list
*alone* scores nDCG@5 0.868, while the three-list fusion that contains it
scores 0.834. Textbook RRF weights every list identically, so the fusion is
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

Sweeping the weight on the n-gram list, over all 56 gold questions at k=5:

| weights | hit@1 | ndcg@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| uniform (plain RRF) | 0.714 | 0.834 | 0.938 | 0.821 |
| `ngram=2` | 0.714 | 0.838 | 0.938 | 0.827 |
| `ngram=3` | 0.750 | 0.847 | 0.938 | 0.844 |
| **`ngram=4`** | **0.768** | **0.858** | **0.946** | **0.853** |
| `ngram=6` | 0.768 | 0.855 | 0.946 | 0.849 |
| ngram only (control) | 0.768 | 0.868 | 0.973 | 0.849 |

The gain concentrates in the hard slice, which is where fusion was supposed to
earn its keep: nDCG@5 0.698 → 0.736 and hit@1 0.545 → 0.636 at `ngram=4`.

Two honest readings of that table, both of which matter more than the win:

**None of it is significant.** Run through `evals/significance.py`, `ngram=4`
against uniform is +0.025 nDCG@5 with a 95% interval of [-0.010, +0.064] and
p=0.205 - won 9 questions, lost 5, tied 42. hit@1 is +0.054 with 5 wins and 2
losses. On 56 questions a handful of queries is the entire effect, and the
interval includes zero on every metric. This is the same lesson the error-bars
section drew about the ablation table, and it applies to the feature that
section motivated.

**The control still wins.** `ngram only` beats every weighted fusion on nDCG@5
(0.868) and recall@5 (0.973). Weighting recovers most of the gap uniform RRF
opened, but it does not close it. The defensible conclusion is not "weighted
RRF is better" - it is that *on this corpus, at this size, fusion has not been
shown to beat its own best input*, and the weight knob makes that visible
rather than hiding it behind a fused default.

So the shipped default stays uniform. A non-significant delta measured against
a control that still wins is not grounds for changing what the engine does by
default; it is grounds for a knob and a table.

### `rrf_k` was not inert. It had nothing to arbitrate.

The tuning section above closed with a claim it could not test: that `rrf_k`
"would start to matter with more retrievers in the fusion, or with retrievers
that disagree more than BM25 and TF-IDF do here - and the honest version of
that claim is that this corpus cannot demonstrate it. **It can now**: adding a
third, structurally different list moves `rrf_k = 1` from 5 changed orderings
to 26, and the sweep is repeated under "`rrf_k` was not inert" below." The third list is the
experiment that demonstrates it. Same corpus, same gold set, same sweep, one
more ranked list:

| rrf_k | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR | changed (2 lists) | changed (3 lists) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.714 | 0.893 | 0.964 | 0.822 | 0.812 | 5 | **26** |
| 5 | 0.714 | 0.911 | 0.964 | 0.826 | 0.817 | 2 | 21 |
| 10 | 0.714 | **0.929** | 0.982 | **0.835** | **0.823** | 1 | 12 |
| 20 | 0.714 | **0.929** | 0.982 | **0.835** | **0.823** | 0 | 8 |
| **60** (default) | 0.714 | 0.911 | 0.982 | 0.834 | 0.821 | 0 | 0 |
| 200 | 0.714 | 0.911 | 0.982 | 0.834 | 0.821 | 0 | 0 |
| 1000 | 0.714 | 0.911 | 0.982 | 0.834 | 0.821 | 0 | 0 |

With two lists, every value at or above 20 returned byte-identical orderings on
all 56 questions and the accuracy columns never moved. With three, `rrf_k = 1`
reorders 26 of 56 - half the gold set - and the sweep develops a shape: low
values over-reward whichever list ranked something first and cost hit@3, and
`rrf_k` of 10 to 20 beats the default on hit@3 (0.929 against 0.911) and edges
nDCG@5 and MRR. The gains are one question wide and not worth changing a
default over. The point is not the gain, it is that the earlier "there was
nothing to tune" was a statement about the *inputs* rather than about the knob,
and it took a genuinely different list to tell those two explanations apart.

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

Ten of the sixty-six (question, relevant document) pairs are missed at k=5.

| id | slice | document | hybrid | bm25 | vector | coverage | category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q15 | easy | evaluation | 8 | 8 | 6 | 0.44 | near-miss |
| h01 | hard | embeddings | 14 | 14 | 14 | 0.28 | deep-miss |
| h02 | hard | bm25 | 11 | 11 | 11 | 0.38 | deep-miss |
| h03 | hard | tfidf | 14 | 14 | 14 | 0.36 | deep-miss |
| h05 | hard | chunking | 10 | 10 | 10 | 0.39 | near-miss |
| h05 | hard | evaluation | 9 | 9 | 9 | 0.33 | near-miss |
| h08 | hard | reranking | 8 | 8 | 7 | 0.39 | near-miss |
| h18 | hard | rag | 7 | 7 | 7 | 0.54 | near-miss |
| h20 | hard | reranking | 8 | 8 | 8 | 0.37 | near-miss |
| h22 | hard | reranking | 7 | 7 | 7 | 0.28 | near-miss |

**Not one of them is a fusion loss**, and the column that proves it is the one
that would otherwise look like padding. In eight of the ten rows the fused rank
is *identical* to both component ranks; in the other two (q15, h08) TF-IDF had
it one or two places higher and the fusion still landed on BM25's rank, which is
a fusion failing to *improve* on a component, not one discarding what a
component found - and neither component had it inside k either. Nothing here is
RRF throwing away a document BM25 or TF-IDF had
already found, so the `fusion-loss` count is zero on every slice. That is worth
stating plainly because the fusion is the newest large moving part and the
obvious thing to blame, and the obvious thing is wrong.

The `coverage` column - the share of the question's distinct terms that occur
anywhere in the document - says why. It never exceeds 0.54 and averages 0.38.
These are the hard-slice questions written to avoid the target document's own
vocabulary, and both retrievers index whole words, so they are two scores over
one representation that cannot see what neither can match. Blaming the fusion
for agreeing is blaming it for arithmetic.

Which lever recovers them is then an empirical question rather than an argument,
so the script re-runs every failing pair under all four (`*` marks a rank inside
k=5):

| id | document | expansion | rerank | mmr | hybrid3 | ngram-only |
| --- | --- | --- | --- | --- | --- | --- |
| q15 | evaluation | 7 | 9 | 8 | 4* | 1* |
| h01 | embeddings | 14 | 14 | 14 | 14 | 14 |
| h02 | bm25 | 10 | 11 | 11 | 10 | 5* |
| h03 | tfidf | 13 | 14 | 14 | 14 | 5* |
| h05 | chunking | 8 | 10 | 8 | 11 | 11 |
| h05 | evaluation | 11 | 9 | 11 | 10 | 10 |
| h08 | reranking | 11 | 8 | 8 | 6 | 4* |
| h18 | rag | 5* | 7 | 9 | 4* | 3* |
| h20 | reranking | 11 | 8 | 9 | 6 | 5* |
| h22 | reranking | 12 | 7 | 7 | 3* | 2* |
| | **recovered** | 1/10 | 0/10 | 0/10 | 3/10 | **7/10** |

**The character n-gram list alone recovers seven of the ten failures the word
retrievers share.** It is the only lever in the repo that changes the
*representation* rather than re-scoring or re-ordering the same one, and this is
the sharpest statement of that difference the project has produced: re-ranking
recovers nothing, because re-ordering a pool cannot admit a document the pool
never contained, and MMR recovers nothing for the same structural reason. Both
were already known to be pool-bound in principle; here it is in the only
arithmetic that can show it.

It also explains a result from the section above that had been left as a
curiosity. The three-list table showed `ngram` beating `hybrid3` outright on the
hard slice (hit@1 0.636 vs 0.545, MRR 0.752 vs 0.708), which reads as a fusion
that dilutes its best member. This table is the same fact at question
granularity: on the pairs where the word lists both fail, the n-gram list is
carrying seven rescues and fusion averages three of them back out against two
word lists that ranked the document tenth.

What survives every lever is the backlog:

```
nothing in the repo recovers these:
  h01 -> embeddings (rank 14, coverage 0.28): If someone writes automobiles but
         the page only ever says cars, what kind of search still finds it?
  h05 -> chunking (rank 10, coverage 0.39): How do I decide how much text to put
         in each piece, and how would I know the choice was right?
  h05 -> evaluation (rank 9, coverage 0.33)
```

`h01` is the honest one, and it is dead last of fourteen under all five
configurations. The question asks for the retrieval method that survives a
synonym, using a car/automobile analogy the corpus never uses; the answer is in
`embeddings.md`, which never says either word. Character n-grams fix
*morphology* - "controls" against "control" - not synonymy, so the one lever
that rescues the other seven cannot touch this. Nothing short of a semantic
representation will, which makes `h01` a fair statement of where a
dependency-free design actually stops rather than a bug to be tuned away. `h05`
is a different shape: one question with two relevant documents, both stuck in
the rank 9-11 band, where every lever moves it a place or two and none moves it
five.

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

Take `h18`, one of the three questions hybrid misses outright at k=5. Its gold
document is `rag.md`, and `--doc` explains a chunk whether or not it was
retrieved, which is the only way to ask why the right answer lost:

```
$ python -m ragkb.cli explain \
    "Why can what a system knows be changed without any training at all?" \
    --doc rag --terms 5

#9  [rag]  rag::1  score=0.02920

fusion (rrf_k=60):
  bm25         rank 8  1/(60+8) = 0.014706
  vector       rank 9  1/(60+9) = 0.014493
  total                          0.029199

bm25 score 5.54818 (top 5 of 5 matched terms)
  term                    qw      tf      df       idf     contrib
  system               1.000       1       7     2.046     1.91415
  be                   1.000       2      19     1.090     1.48440
  can                  1.000       3      28     0.711     1.14073
  at                   1.000       1      20     1.040     0.97322
  a                    1.000       2      56     0.026     0.03568
  missing from this chunk: all, any, training, what, why, without
  not in the corpus at all: changed, knows
```

Every point `rag.md` scored came from `system`, `be`, `can`, `at` and `a`. Not
one content word of the question contributed anything, and the two that carry
its meaning - `changed` and `knows` - are not in the collection vocabulary at
all. The document that beat it, `citations.md`, won on the same kind of
vocabulary: `system`, `without`, `what`, `be`, `can`. So this is not a ranking
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
- **near-domain** (12) - SPLADE, dense-passage-retrieval training objectives,
  sharding a search cluster. Information retrieval questions this corpus
  happens not to answer.

The split is the point of the file. Off-domain questions are built from words
the corpus never indexed and any threshold catches them; near-domain questions
are assembled almost entirely from vocabulary the corpus knows, so they are the
only rows that say anything. Reporting one pooled number would be flattered by
the easy half.

```bash
python evals/abstention_eval.py
```

Mean score: **0.455** answerable, **0.256** near-domain, **0.111** off-domain.

| threshold | answer rate | abstain off | abstain near | accuracy | balanced | caught misses | lost hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 1.000 | 0.000 | 0.000 | 0.757 | 0.500 | 0.000 | 0.000 |
| 0.20 | 0.982 | 0.667 | 0.250 | 0.838 | 0.686 | 0.250 | 0.000 |
| 0.24 | 0.946 | 1.000 | 0.500 | 0.878 | 0.807 | 0.500 | 0.019 |
| **0.28** | **0.857** | **1.000** | **0.667** | 0.838 | **0.817** | 0.500 | 0.115 |
| 0.30 | 0.768 | 1.000 | 0.750 | 0.784 | 0.801 | 1.000 | 0.173 |
| 0.36 | 0.625 | 1.000 | 0.917 | 0.703 | 0.785 | 1.000 | 0.327 |
| 0.50 | 0.393 | 1.000 | 0.917 | 0.527 | 0.669 | 1.000 | 0.577 |

The shipped default is **0.28**, chosen on balanced accuracy rather than pooled
accuracy. Pooled accuracy peaks at 0.878 with a gate that declines only half
the near-domain questions, and it would peak higher still on a set with more
answerable questions in it - there are three times as many here, so the pooled
column mostly rewards answering everything. Balanced accuracy weights the two
error types equally and picks 0.28: every off-domain question declined, two
thirds of the near-domain ones, and 86% of answerable questions still answered.

The last two columns are the ones worth arguing about. Four of the 56
answerable questions are ones retrieval already misses at k=4 - it returns
chunks, none from a relevant document - and on those the ideal gate abstains
too, because answering from evidence that does not contain the answer is the
same failure whether or not the corpus *could* have answered it. At 0.28 the
gate declines half of those misses and 12% of the questions retrieval got
right. Push to 0.30 and it catches all four, at the cost of throwing away 17%
of good retrievals. Neither is obviously correct; which one is depends on what
a wrong answer costs, which is why the gate is off by default and the threshold
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
0.571 - but it is the same morphological blind spot the character n-gram index
exists to fix for retrieval, and the gate does not yet consult it.

The single worst survivor at 0.28 is `u08`, *"what loss function is dense
passage retrieval trained with"*, at 0.51. Every content word in it appears in
`embeddings.md`, which discusses dense retrieval and mentions training - the
corpus simply never names the loss. No lexical signal can tell those apart, and
saying so is more useful than tuning the threshold until that one row flips.

### How much of any of that is real? Error bars on the ablation table

Every table above this line reports a mean over the same 56 questions, and
several of the differences in them are one question wide. That is not a
rhetorical point: at n=56 one question is worth 0.018 of any per-query metric,
so hybrid's 0.696 hit@1 against BM25's 0.714 is *exactly one question*, and the
hard slice - 22 questions - cannot resolve anything finer than 0.045. Nothing in
the repo could distinguish "this lever helps" from "this lever moved one row",
which made the whole evaluation section easy to over-read. `evals/significance.py`
closes that hole:

```bash
python evals/significance.py
```

```
all slice - 56 questions, one question is worth 0.018
metric    comparison                 A       B    delta              95% CI       p   floor       W/L/T
-------------------------------------------------------------------------------------------------------
hit@1     hybrid vs bm25         0.696   0.714   -0.018    [-0.054, +0.000]   1.000   1.000      0/1/55
hit@5     hybrid vs bm25         0.946   0.929    0.018    [+0.000, +0.054]   1.000   1.000      1/0/55
ndcg@5    hybrid vs bm25         0.799   0.801   -0.002    [-0.021, +0.019]   0.875   0.031      3/3/50
ndcg@5    bm25 vs vector         0.801   0.794    0.007    [-0.019, +0.034]   0.648   0.008      5/3/48
mrr       hybrid vs bm25         0.792   0.800   -0.007    [-0.030, +0.011]   0.625   0.031      3/3/50
```

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
anywhere in the run is 8 questions out of 56, and most rows are 0/1/55. The
comparison's real sample size is not 56, it is the discordant count, and every
tie is a question that carries no evidence either way.

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
configuration is identical, so "the best fusion weighting scored 0.768 hit@1" is
not a prediction about the next 56 questions - it is the best of seven draws
from the same 56.

`evals/holdout.py` measures the size of that bias by nested cross-validation:
split the questions into folds, pick the winner on the training folds, score
that choice on the held-out fold, and repeat over twenty different splits.

```bash
python evals/holdout.py
python evals/holdout.py --slice hard --folds 4
```

```
56 questions (all slice), k=5, 5-fold x 20 splits, seed 0

sweep           metric    picked          in-sample   held-out   optimism   vs default   stable
-----------------------------------------------------------------------------------------------
rrf_k           hit@1     60                  0.696      0.696     +0.000       +0.000     100%
rrf_k           ndcg@5    1                   0.801      0.799     +0.001       +0.000       0%
rrf_k           mrr       1                   0.795      0.792     +0.003       -0.000       0%
fusion weights  hit@1     ngram=4             0.768      0.720     +0.048       +0.005       0%
fusion weights  ndcg@5    ngram only          0.868      0.860     +0.008       +0.026      45%
fusion weights  mrr       ngram=4             0.853      0.828     +0.025       +0.007       0%
mmr lambda      hit@1     off                 0.696      0.696     +0.000       +0.000     100%
mmr lambda      ndcg@5    off                 0.799      0.799     -0.000       +0.000     100%
mmr lambda      mrr       off                 0.792      0.791     +0.001       -0.001      80%
```

Read `optimism` first, on the `fusion weights / hit@1` row. The weighted-RRF
table earlier in this README reports `ngram=4` at 0.768 against uniform's 0.714
and invites exactly the conclusion it looks like. Scored on questions the choice
never saw, that configuration earns **0.720** - the headline shed 0.048, which
on a 56-question set is between two and three questions of the four the sweep
appeared to win. The `vs default` column finishes the demolition: the honest
estimate of the whole procedure "sweep seven weightings, ship the winner" beats
uniform RRF by 0.005, one-tenth of one question. On the hard slice the same row
loses 0.070 to selection.

Two controls make that number trustworthy rather than merely alarming, and they
are why the script sweeps three different knobs instead of the one it wanted to
indict:

- **`rrf_k` reports near-zero optimism.** `evals/tune.py` already established
  that `rrf_k` is inert over two lists - it changes nothing, so there is nothing
  for selection to inflate, and the harness agrees. A bias detector that printed
  a large number on a knob known to do nothing would be measuring its own
  arithmetic.
- **MMR reports near-zero optimism and 100% stability**, because on this corpus
  every lambda loses to `off` on every split. Selection cannot inflate a sweep
  whose winner is the default.

The `stable` column is the same finding read a different way, and the two
columns are genuinely independent. `rrf_k / ndcg@5` is 0% stable with +0.001
optimism: the winner changes with every split, and it does not matter, because
the configurations are interchangeable. `fusion weights / hit@1` is 0% stable
with +0.048 optimism: there the instability is expensive, because the table
presents its winner as a finding. A sweep row that is unstable *and* optimistic
is a sample property being reported as a setting.

One row survives the treatment, and it is the one the earlier section already
flagged as the control it could not beat: `ngram only` on nDCG@5 holds +0.026
over uniform out of sample, at 45% stability - and on the hard slice, +0.065 at
95%. The three-list fusion is not being under-weighted; it is losing to its own
best component, and that conclusion is the only one in this section that is
stronger after cross-validation than before it.

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
half: per-query wall time for each configuration, on the same 56 gold
questions, pooled over repeats so `p95` is a genuine tail over queries rather
than a mean in disguise.

```
query latency (per query, k=5)          cost against quality (k=5)
stage             median_ms   xbase       hit@1   recall@5    ndcg@5       mrr
------------------------------------------------------------------------------
hybrid (2 lists)       0.18    1.00       0.696      0.893     0.799     0.792
hybrid (3 lists)       0.42    2.38       0.714      0.938     0.834     0.821
  + mmr 0.7            1.22    6.95       0.696      0.875     0.784     0.783
  + expansion          1.29    7.35       0.589      0.911     0.773     0.748
  + rerank             1.37    7.80       0.714      0.893     0.808     0.808
  + all three          3.58   20.41       0.679      0.893     0.792     0.786
```

The ordering is the finding, and it is not the one the build order implies. The
cheapest addition in the pipeline is also the best one: the character n-gram
third list costs 2.4x a plain fused query and is the only row that improves
every accuracy column at once. The three stages that came after it each cost
about 7x and buy either a little (re-ranking: +0.018 hit@1, +0.009 nDCG@5) or
nothing (expansion at these settings is negative at rank 1, which the sweep
table already said). Stacked together they cost 20x and land *below* the
2.4x row. If this were a latency budget rather than a portfolio project, the
recommendation would be one line: ship the third list, keep re-ranking behind a
flag, leave the rest off.

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
* **The n-gram index dominates build time**, at ~10ms against ~2.6ms for both
  word indexes combined. Its query cost is proportionate, so the third list is
  a consistent ~2.4x tax at both ends rather than a cheap index that is
  expensive to query.

The honest caveat is scale: 14 documents and 57 chunks is small enough that
these ratios describe *this* corpus, not a curve. Some of them should move
predictably - MMR's work is O(k²) over a fixed `k` and so barely grows, while
first-stage retrieval grows with the corpus, which means MMR's 7x is a
small-corpus artefact that should shrink. Absolute milliseconds are this
machine's; the `xbase` column is the part that travels, and even it has a
domain.

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
bm25                  57     96.0%       40150       10592        0.26
tfidf                 57     96.0%       35258       10592        0.30
ngram(4)              57     99.9%      153673       40174        0.26
```

`reach` is the share of the collection a gold question touches through at least
one term: 96%. One common word in a question - *what*, *does*, *the* - matches
almost every chunk on its own, so the entire prize for filtering by membership
is the remaining 4%. For the n-gram index it is 0.1%, because a 4-gram query
reaches everything. Nor does that improve with scale: *the* appears in nearly
every document at 57 chunks and at 57,000.

The work was somewhere else. Scoring document-at-a-time costs one dictionary
lookup per (reached document, query term) pair *whether or not the document
contains the term* - 96% of the collection times every word in the question,
and most of those lookups miss. Walking postings term-at-a-time instead, and
accumulating each document's running score, costs one update per posting that
actually exists: the sum of the query terms' document frequencies. That is the
`taat/scan` column, and it is the durable number in this section because it is
a count rather than a timing - about a quarter of the lookups, on every index.

The wall clock agrees, at 10 repeats over the 56 gold questions:

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

Measured on this corpus (14 documents, 57 chunks), loading is about 1.7x
faster than a cold build; on a 300-document copy of the same material the gap
grows to about 2.4x, since fitting scales with corpus size while loading is a
single JSON parse. Rankings, scores and answers are bit-for-bit identical
either way - `tests/test_store.py` asserts exactly that.

### Growing the index: what "incremental" is actually worth

Persistence removes the cost of rebuilding an index that did not change. The
other half of the same problem is an index that changed *slightly* - one new
document arriving at a process that already holds the other thirteen. Until now
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
cold: build all 14 docs              5.88        7.98        6.17          25      1.00
cold: build 13 then add 1            9.10       10.75        9.26          25      1.55
warm: add 1 to a live index          3.77        4.27        4.16          25      0.64
```

That is one run, not an average of several. Across repeated runs on this machine
the cold-incremental multiplier lands between 1.3x and 1.6x and the warm one
between 0.59x and 0.68x, which is the precision these numbers actually have -
a build this short is a handful of milliseconds sitting on top of scheduler
noise. The sign of each result is solid; the second decimal place is not.

Two different answers, and the difference is what you start from.

From cold the incremental path **loses, by about half again**. Reaching fourteen
documents by way of thirteen pays the TF-IDF fit twice - once at thirteen, once
at fourteen - and that refit is the largest single stage in the build (2.24 ms
of it, against 1.22 ms for BM25 and 1.29 ms for chunking). If you do not already
have an index, build one; there is no version of this where assembling a corpus
document by document is the fast way.

From warm - a running process holding an index and handed one new document,
which is the only case the method exists for - it costs roughly **0.6x a full
rebuild**, against a corpus rebuild that is itself only ~6 ms. The saving is
the tokenising and counting of the thirteen documents already indexed, and it is
real but bounded, because the TF-IDF refit stays in the bill.

That framing is also the honest scaling statement, and it does not flatter the
feature: the saving is a *fixed* cost removed (one pass over the documents
already indexed, which is what add avoids re-doing) set against a cost that
*grows* with the collection (rebuilding every vector in it). The 0.65x is not a
floor that improves with scale - it is a number that gets worse as the corpus
grows, until the refit dominates and the warm row converges on 1.00x. Nothing
here measures that curve; one corpus size cannot. What it does establish is
which way the curve points and why.

The correctness claim is the part that has to be exact, and it is checked
against real from-scratch builds rather than against a re-derivation of the same
arithmetic. All 56 gold questions, ranked by both a grown engine and a whole
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
  the price it puts on the largest sweep in the README is 0.048 hit@1, most of
  the improvement that sweep appeared to find. The two knobs where nothing is
  being selected report ~0.000, which is how the number earns belief.
- **A score you cannot decompose is a score you cannot debug.** Both retrievers
  and the fusion are additive, so `ragkb/attribution.py` reconstructs any
  single score from its summands exactly and the tests assert that identity.
  Aggregates say a change helped; only the decomposition says which word did it,
  and whether the miss was fixable at all.

## Tests

```bash
pytest -q      # 415 tests
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
index still round-trips through a save/load), and an end-to-end pass over the
real corpus and gold set.

One of those tests exists because a bug shipped into a draft: `--ngram` was
first declared with argparse's `nargs="?"`, which quietly consumes the
*subcommand* as its value, so `--ngram search ...` died with "invalid int
value: 'search'" and the flag only worked when it happened to be last on the
line. `tests/test_cli.py` now asserts the flag survives having a subcommand
after it.
