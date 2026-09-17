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
| `ragkb/significance.py` | paired bootstrap intervals, randomization tests, and Holm familywise correction |
| `ragkb/power.py` | exact sign-test power, required gold-set size, exact binomial intervals, the paired McNemar test, and Fisher's exact test |
| `ragkb/difficulty.py` | measured question difficulty: query-document containment, IDF-weighted, and the bands the diagnostics stratify along |
| `ragkb/cli.py` | `stats`, `search`, `ask`, `explain`, `index`, `--mmr`, `--expand`, `--rerank`, `--ngram`, `--abstain`, `--index` |
| `evals/tune.py` | index-time sweep: chunk size, chunk overlap, `rrf_k` (2 or 3 lists) |
| `evals/failures.py` | per-question failure analysis: where a missed document really ranks, and which lever recovers it |
| `evals/robustness.py` | the same ablation under misspelled queries: which methods degrade, and how fast |
| `evals/significance.py` | error bars on the ablation table: which of those differences survive a paired test and a familywise correction across all five methods |
| `evals/power.py` | sample-size analysis: the smallest effect this gold set could detect, and what each claim would cost to establish |
| `evals/corpus_lever.py` | which half of an increment moved the discordance rate: the new documents, or the new questions |
| `evals/difficulty.py` | audits the hand-assigned `easy`/`hard` label against measured containment, then stratifies the ablation along the measured axis with question length as a negative control |

`corpus/` holds the sample knowledge base (twenty-six documents on information
retrieval). `evals/gold.jsonl` holds one hundred and thirty-two labelled questions with
the documents that answer them, split into an `easy` and a `hard` slice - a
hand-assigned split, audited against a measured one in
[Is the "hard slice" a real slice?](#is-the-hard-slice-a-real-slice-auditing-the-partition-before-the-deltas),
which is also where the confound it turned out to carry is recorded.
`evals/unanswerable.jsonl` holds twenty-two questions the corpus deliberately
cannot answer, split into `off-domain` and `near-domain`, which is what the
abstention gate is measured against.

## Evaluation

```bash
python evals/evaluate.py     # ablation + query-time sweeps + the third retriever
python evals/significance.py # error bars: which of those differences are real
python evals/power.py        # how many questions the differences would need to be real
python evals/corpus_lever.py  # did the new documents or the new questions move the rate?
python evals/difficulty.py   # is the hard slice a real slice? and where the n-gram gain lives
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

Current numbers on the bundled corpus and gold set (26 documents, 126 chunks,
132 questions, k=5, document-level relevance).

> **Provenance.** Recomputed at twenty-six documents and 132 questions: the
> ablation tables, the third-retriever tables, the chunking sweep, `rrf_k` over
> two lists and three, the weighted-RRF weight sweep, the MMR lambda sweep, the
> whole significance section, difficulty, holdout and power. That closes the two
> sections the previous revision named as its next increment and left marked
> stale, so **the 106-question and 92-question markers on the ablation and
> significance sections are gone rather than re-dated.**
>
> **Still stale, and marked at the point of use:** latency, robustness and
> persistence remain at the twenty-document, ninety-two-question figures, and
> the expansion and re-ranking sweeps remain at ninety-two.
>
> **One correction that the stale markers could not have caught.** The
> significance section carried two mutually contradictory counts of how many
> comparisons survive correction - a table saying 14 raw and 0 surviving, and a
> paragraph saying 29 and 1. Re-running the previous commit's configuration
> gives 44 and 8, so both were wrong. `tests/test_readme_provenance.py` checks
> table *captions* against the shipped gold set; it does not read the numbers
> inside a table, which is where these sat. The test is doing its job and its
> job is narrower than it looks.

The gold set is split by difficulty. The `easy` slice is seventy-three
questions whose wording shares vocabulary with the target document. The `hard`
slice is fifty-nine questions written to avoid the target document's own terms -
"if someone writes automobiles but the page only ever says cars" instead of
"vocabulary mismatch" - and thirty-seven of them are answered by more than one
document rather than one. Those counts are no longer a description of intent:
`evals/difficulty.py` confirms the split does separate on measured
query-document overlap (0.614 against 0.337 IDF-weighted, p < 5e-05), and also
that all thirty-seven multi-document questions sit in the `hard` slice and none
in `easy`, so the two axes coincide exactly and a hard-slice figure carries
both. [The audit is
here](#is-the-hard-slice-a-real-slice-auditing-the-partition-before-the-deltas).

### The ablation: hybrid against the two lists it fuses

This subsection did not previously have a heading of its own, which is part of
why an earlier drift went unnoticed: the tables read as a continuation of the
gold-set description above them rather than as a result with its own provenance.
They were marked stale at 106 questions for two revisions and are current again
here.

**all (132 questions, 37 with more than one relevant document)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.727 | 0.864 | 0.924 | 0.850 | 0.777 | 0.802 |
| bm25 | 0.720 | **0.879** | 0.917 | 0.846 | 0.772 | 0.795 |
| vector | 0.727 | 0.871 | 0.924 | 0.850 | 0.777 | 0.802 |

**easy (73 questions)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | **0.863** | **0.945** | 0.986 | 0.986 | **0.927** | **0.907** |
| bm25 | 0.849 | 0.932 | 0.986 | 0.986 | 0.920 | 0.898 |
| vector | 0.849 | 0.932 | 0.973 | 0.973 | 0.915 | 0.895 |

**hard (59 questions, 37 with more than one relevant document)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.559 | 0.763 | 0.847 | 0.681 | 0.591 | 0.672 |
| bm25 | 0.559 | **0.814** | 0.831 | 0.672 | 0.589 | 0.667 |
| vector | **0.576** | 0.797 | **0.864** | **0.698** | **0.606** | **0.687** |

The easy slice is still the less informative half. Every question on it has
exactly one relevant document, which makes recall@k numerically identical to
hit@k - the column is decoration. hit@5 sits at 0.973 or better for all three
methods, so there is almost nothing left there for a retrieval change to move.

The hard slice restores headroom: hit@1 falls from 0.863 to 0.559 and MRR from
0.907 to 0.672, and recall@k finally diverges from hit@k because a question
answered by several documents can be half-satisfied. Twenty questions land in
exactly that state at k=5, which is a failure mode the easy slice cannot express
at all.

nDCG@5 is the one column that rewards ranking the *second* relevant document
high: MRR stops looking after the first hit, and recall@5 does not care whether
that second document sits at rank 2 or rank 5. Repeated chunks of one document
are credited once, so a top-5 stuffed with paraphrases of a single passage
cannot outscore one that ranks two distinct relevant documents.

**The headline this section carried at eighteen documents did not survive
twenty, and it has not come back.** That revision reported hybrid beating both
of its components on every hard-slice column it could - hit@1 0.594 against
0.562 for each, recall@5 0.750 against 0.719, MRR 0.704 against 0.682 - and
called it "the result the project was built to look for". At twenty-six
documents the hard slice belongs to **plain TF-IDF**, which takes hit@1, hit@5,
recall@5, nDCG@5 and MRR outright while BM25 takes hit@3 and the fusion takes
nothing at all. On the all slice the fusion and TF-IDF are identical to three
decimals on all six columns. The lead did not shrink; it inverted, and it has
now stayed inverted across three corpus sizes.

Nothing in the retriever changed across those tables. What changed is that the
questions deciding the result were replaced by more questions, and the
eighteen-document revision said in as many words what would happen if they were:
"Every one of those leads is one or two questions wide... One question decided
the headline. Read the table as 'the sign is finally consistent', not as 'fusion
wins'." That caveat is now a measurement rather than a hedge, and it is worth
more than the win it was hedging. A one-question lead on a 78-question gold set
is indistinguishable from noise, and the cheapest way to demonstrate that is to
add questions and watch the sign move.

So the standing claim remains the weaker and older one: **on this corpus, fusing
BM25 with TF-IDF has still never been shown to beat BM25 alone.** The
significance section tests exactly that comparison and reports NOT SHOWN
(nDCG@5, +0.004, 9W/9L/114T) - eighteen discordant questions splitting exactly
nine each way, which is as close to a measured zero as this gold set can
produce.

Worth separating from that, because the two are easy to run together: the
*three-list* fusion is a different claim and it does now clear correction. Every
comparison surviving Holm in the current significance run has `hybrid3` on one
side. Fusing BM25 with TF-IDF buys nothing here; adding a structurally different
third list to that fusion is the one retrieval change this project has
demonstrated.

The corpus keeps being grown for exactly this reason, each new document placed
beside an existing one it shares vocabulary with so that some question gets a
lexically plausible wrong neighbour: `late-interaction.md` beside
`reranking.md`, `vector-compression.md` beside `ann-search.md`,
`sparse-neural-retrieval.md` beside `bm25.md` and `inverted-index.md`,
`multi-hop-retrieval.md` beside `query-processing.md` and `rag.md`,
`domain-shift.md` beside `evaluation.md` and `bm25.md`, `negative-mining.md`
beside `embeddings.md` and `reranking.md`, `index-compression.md` and
`dynamic-pruning.md` both against `inverted-index.md` because it was the one
document covering two large topics in a paragraph each, and the two newest:
`passage-aggregation.md` beside `chunking.md`, which already described how a
document is cut up but never how the resulting scores are put back together,
and `query-performance-prediction.md` beside `evaluation.md` and `abstain`'s
own subject matter, since both are about judging a result - one with labels and
one without.

The one pattern that survived the growth unchanged is on the easy slice: BM25
still takes it outright at rank one (hit@1 0.850 against 0.833, MRR 0.893
against 0.885). Fusion buys ordering quality on paraphrased questions and pays
a little at rank one on literal ones. On the all slice BM25's hit@1 lead has
closed to a tie again - 0.736 against 0.736 - having reopened at twenty
documents, which is the same one-question churn read from the other
end.

One documented behaviour has now moved twice, which is worth tracking rather
than quietly re-pinning. `tests/test_expansion.py` originally asserted that
pseudo-relevance feedback rescues h18 (`rag`) at the shipped weight. At sixteen
documents the feedback pool changed and it no longer did - recovering h18 needed
`original_weight=0.7`, which lost three other questions - so the test was
re-pinned to h15. At eighteen documents h18 was recovered again at the shipped
setting, landing at rank 4. At twenty, expansion does not change h18's top-5 at
all. The lever never broke; its operating point moves with the feedback pool.
h15 stays pinned because it is the one case that has held at every corpus size
so far.

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

**chunking, all 132 questions, k=5.** One question is worth 0.008 of any
column here, so a 0.030 gap is four questions and a 0.008 gap is one.

| target:overlap | chunks | mean tok | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60:15 | 336 | 52.3 | 0.727 | 0.871 | 0.902 | 0.823 | 0.767 | 0.798 |
| 80:20 | 220 | 66.5 | **0.750** | 0.856 | 0.879 | 0.801 | 0.763 | 0.805 |
| 100:25 | 156 | 83.8 | 0.735 | 0.886 | 0.917 | 0.842 | 0.781 | 0.809 |
| 120:0 | 103 | 100.2 | 0.689 | 0.841 | 0.902 | 0.827 | 0.752 | 0.773 |
| 120:15 | 124 | 99.8 | **0.750** | 0.864 | 0.924 | 0.851 | 0.782 | **0.814** |
| **120:30** (default) | 126 | 99.8 | 0.727 | 0.864 | 0.924 | 0.850 | 0.777 | 0.802 |
| 120:60 | 152 | 104.5 | 0.742 | 0.886 | 0.924 | 0.854 | **0.788** | **0.814** |
| 150:40 | 98 | 125.8 | 0.712 | **0.894** | **0.932** | **0.857** | 0.787 | 0.807 |
| 180:45 | 83 | 147.1 | 0.689 | 0.879 | **0.932** | **0.857** | 0.772 | 0.787 |
| 200:50 | 71 | 169.6 | 0.705 | 0.879 | **0.932** | **0.857** | 0.780 | 0.796 |

**Overlap earns its keep, and the claim has now held at five corpus sizes.**
`120:0` is the same chunk size with the overlap switched off, and it is the
worst row in the table at rank one and the worst row outright at MRR: hit@1
falls 0.727 -> 0.689 and MRR 0.802 -> 0.773. That hit@1 loss has gone 0.031 at
sixteen documents - two questions, fairly called noise at the time - then 0.064,
0.043, 0.051 and now 0.038. It is the largest MRR movement anywhere in the sweep
and it ties `180:45` for the largest hit@1 movement, so the "by a wide margin"
phrasing earlier revisions used has to go even though the finding has not. This
remains the only claim in this table that five corpus sizes have left standing.
An idea straddling a chunk boundary really is retrievable from both sides, and
dropping overlap really does cost the questions that depend on it.

**One claim beside it came back, and one that was corrected last increment has
now reversed a second time.**

The returning one is the depth cost of dropping overlap. Two revisions ago the
table was read as `120:0` buying back at depth what it loses at rank one;
at twenty-four documents that gain shrank to +0.004 and the sentence was
retracted. At twenty-six documents `120:0` loses at depth as well - recall@5
-0.023 and hit@5 -0.022 against the default - so the row is now simply worse
everywhere rather than trading rank one for depth. That is a *stronger* result
for overlap than the original claim was, which is why it is worth flagging that
it arrived by reversal rather than by confirmation: a quantity that has read
+0.011, +0.004 and -0.023 across three corpus sizes is a quantity this sweep
cannot pin down, and the honest summary is that overlap's effect at depth is
unresolved while its effect at rank one is not.

The extremes still fail asymmetrically rather than in the expected U shape.
`60:15` costs hit@5 (-0.023) and recall@5 (-0.027) as chunks too short to hold a
whole answer, while gaining hit@3 and sitting flat at rank one. The long end
still shows no penalty at all: `150:40`, `180:45` and `200:50` all tie for the
best hit@5 and recall@5 in the table. The plateau claim survives in the same
weakened form as before - everything from 100 tokens up is within about four
questions of everything else, and the exact value inside that band is still not
what any other result here depends on.

**And the claim this section made most confidently last increment is gone.** At
twenty-four documents `120:60` took or tied five of six columns, by margins of
three to four questions, and the prose said flatly that the older "winner
scattered across four rows" description "is no longer accurate and should not be
recycled." Two documents and fourteen questions later `120:60` takes nDCG@5
outright, ties MRR with `120:15`, and wins nothing else; the best hit@1 is now a
tie between `80:20` and `120:15`, and depth is swept by the three longest rows.
The winner is scattered across four rows again. The description that was retired
was the correct one, and the confident replacement lasted exactly one increment.

That is the most useful thing this table has produced, so it should not be
buried in a row of decimals: **an in-sample sweep winner chosen on 118 questions
did not survive the addition of fourteen more.** It is the third such reversal in
this file and the first one this README had explicitly predicted for itself, in
the paragraph immediately below, which is reproduced unchanged because it now
reads as a forecast rather than a caveat.

`120:60` was never adopted, for two reasons worth separating. The weak one is
bookkeeping: every other table in this README was measured at 120:30, so moving
the default would invalidate them all in one edit. The real one is that this is
exactly the pick the holdout section prices, and it prices it as the most
expensive kind - the chunking sweep is not among the three knobs `holdout.py`
cross-validates, because each row is a full index rebuild, so `120:60` had an
in-sample margin and *no* out-of-sample estimate at all. A four-question lead
chosen by looking at the same questions it is scored on is the pattern that had
already reversed twice in this file. It has now reversed a third time, on the
row that prompted the warning. Extending the holdout harness to cover index-time
knobs is still the increment that would settle it; until then no row here is a
candidate for anything.

**`rrf_k`, all 132 questions, k=5.** The `changed` column counts questions whose
top-5 ordering differs from the default - the measurement that separates
"equally good" from "does nothing at all".

| rrf_k | hit@1 | hit@5 | nDCG@5 | MRR | changed |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.727 | 0.924 | 0.777 | 0.803 | 14 |
| 5 | 0.727 | 0.924 | 0.776 | 0.802 | 3 |
| 10 | 0.727 | 0.924 | 0.777 | 0.802 | 2 |
| 20 | 0.727 | 0.924 | 0.777 | 0.802 | 1 |
| **60** (default) | 0.727 | 0.924 | 0.777 | 0.802 | 0 |
| 200 | 0.727 | 0.924 | 0.777 | 0.802 | 0 |
| 1000 | 0.727 | 0.924 | 0.777 | 0.802 | 0 |

`rrf_k` is inert on this corpus, and the `changed` column is what proves it:
every value at or above 60 returns byte-identical top-5 orderings on all 132
questions, and the values below it reorder a handful. This is not "tuning did
not help", it is "there was nothing to tune", and those are different claims
that an accuracy table alone cannot tell apart.

Three details in that paragraph have needed correcting as the corpus grew, and
all three are worth keeping visible because each one narrows the claim slightly.

The threshold moved. Earlier revisions wrote "every value at or above 20", and
at twenty-six documents `rrf_k = 20` reorders one question, so the sentence
above says 60. One question is the smallest possible non-zero in that column and
the finding is untouched, but the boundary was quoted as exact and is no longer
where it was.

The reordering count at `rrf_k = 1` had gone 5 -> 8 -> 16 and was described as
"doubling again"; it then read 13 and now reads **14**, so it did not keep
doubling and the trend was read out of three points. Across five corpus sizes
that count has tracked corpus size loosely and nothing more.

And the older sentence "not one metric moves by 0.001" has now been false in two
different places. At twenty-four documents `rrf_k = 1` cost 0.001 of nDCG@5 and
0.001 of MRR. At twenty-six the cost has moved to a different row and changed
sign: `rrf_k = 1` now *gains* 0.001 of MRR and is flat on nDCG@5, while
`rrf_k = 5` costs 0.001 of nDCG@5. A quantity that small relocating between rows
is exactly what one question's worth of ordering looks like, and it is the
reason the `changed` column is the honest one to read here - hit@1 and hit@5 are
still flat to three decimals across the whole sweep.

The reason is structural. `rrf_k` sets how steeply `1 / (rrf_k + rank)` decays,
which interpolates between two known fusion rules: as `rrf_k` approaches zero
the first list to rank something highly dominates, and as it grows large every
term flattens toward `1/rrf_k` and the ordering converges on the sum of ranks -
Borda count. With only *two* lists that already agree on most questions, both
ends produce the same order. At eighteen and twenty documents no value moved any
metric at all; at twenty-four the extreme end finally moves two of them by 0.001
apiece, which is the smallest non-zero this table can print. The knob would start to matter
with more retrievers in the fusion, or with retrievers that disagree more than
BM25 and TF-IDF do here - and the honest version of that claim is that this
corpus cannot demonstrate it.

### Query expansion: borrowing the words the question did not use

> **Stale (92 questions).** The sweep tables below are at 54 easy and 38 hard,
> two corpus increments behind the 66 and 52 the gold set now holds. This
> section was not on the previous revision's list of known-stale sections; the
> README-provenance test added in `tests/test_readme_provenance.py` is what
> surfaced it, which is the case for having the test. `python evals/evaluate.py`
> prints the current sweep.

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

**easy (54 questions)**

| terms | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.796 | 0.926 | 0.963 | 0.963 | 0.866 |
| 5, a=1.0 | 0.796 | 0.926 | 0.963 | 0.963 | 0.866 |
| 3, a=0.9 | 0.759 | 0.907 | **0.981** | **0.981** | 0.848 |
| 5, a=0.8 | 0.704 | 0.907 | **0.981** | **0.981** | 0.819 |
| 5, a=0.7 | 0.611 | 0.907 | 0.963 | 0.963 | 0.766 |

**hard (38 questions, 21 with two relevant documents)**

| terms | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.526 | 0.816 | 0.868 | 0.697 | 0.674 |
| 5, a=1.0 | 0.526 | 0.816 | 0.868 | 0.697 | 0.674 |
| 3, a=0.9 | 0.395 | 0.816 | **0.895** | **0.750** | 0.589 |
| 5, a=0.8 | 0.395 | 0.816 | 0.868 | 0.737 | 0.583 |
| 5, a=0.7 | 0.342 | 0.816 | **0.895** | **0.750** | 0.558 |

`alpha = 1.0` puts all the weight on the original query, so nothing is borrowed
and the ranking has to be identical to the unexpanded one. It is, on every one
of the 92 questions - that row and the test behind it are the correctness check
on this whole path, the same job `lambda = 1.0` does for MMR.

The split is the result, and each round of corpus growth has sharpened both
sides of it. On the hard slice, three borrowed terms at `alpha = 0.9` still buy
real reach - hit@5 rises 0.868 -> 0.895 and recall@5 0.697 -> 0.750, the best
recall any configuration reaches on that slice - and expansion remains the lever
that recovers h06, h15, h18, h30 and h32 from a complete miss at k=5. But rank
one pays for it, and pays more than it used to: hit@1 falls 0.526 -> 0.395 and
MRR 0.674 -> 0.589 at that same setting. On the easy slice the same borrowing
costs two questions at rank one and 0.018 of MRR while buying one question at
k=5 - those questions already shared vocabulary with their target, so almost
every borrowed term is noise.

The operating point has now moved in three directions across four corpus sizes,
which is worth recording because it is the kind of thing a stale README hides.
At 14 documents, `alpha = 0.8` recovered h18 ("why can what a system knows be
changed without any training at all?"). At 16 it did not: the feedback pool had
gained `sparse-neural-retrieval` and `multi-hop-retrieval`, both of which discuss
training and updating knowledge, and the terms borrowed from them pulled h18 away
from `rag`. At 18 it recovered h18 again, at rank 4. At 20 expansion does not
merely fail to recover h18 - it leaves h18's top-5 *byte-identical* to the
unexpanded ranking, so the lever does not fire on that question at all. Nothing
about the method changed across those four runs.

That instability had quietly been load-bearing in the test suite.
`test_engine_config_expansion_is_applied` probed exactly h18 to prove the
engine-level config was plumbed through, so at twenty documents it failed while
the config was being applied perfectly - a claim about the corpus wearing the
costume of a claim about the code. It now asserts the invariant that actually
belongs there and holds at any corpus size: an engine carrying the config ranks
identically to a plain engine handed the config per call, on every gold
question, with a separate assertion that expansion moves at least one ranking so
the equivalence cannot pass vacuously. `test_expansion_recovers_a_question...`
stays pinned to h15, the one case that has held at every corpus size so far.

That is the textbook pseudo-relevance feedback trade-off, and this corpus
reproduces it cleanly, including its instability: expansion buys reach on
paraphrased questions, pays for it in early precision on literal ones, and which
specific questions land on which side is a property of the corpus rather than of
the method. It is **off by default** and exposed as `--expand`. Two cautions on
reading the tables: the hard slice is 38 questions, so a 0.026 step is one
question; and the feedback documents come from the same retriever being
evaluated, so when the first pass is wrong, expansion confidently makes it
wronger.

### Re-ranking: a second stage that reads the pair

Fusion recalls far better than it precisely ranks. On the full gold set hit@5 is
0.924 but hit@1 is only 0.685: for a quarter of the questions the answer is
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
| off | 0.685 | 0.880 | 0.924 | 0.772 | 0.787 |
| 1.0 | 0.685 | 0.880 | 0.924 | 0.772 | 0.787 |
| 0.8 | 0.717 | 0.880 | 0.924 | 0.785 | 0.807 |
| 0.6 | **0.761** | 0.891 | **0.935** | 0.804 | **0.830** |
| **0.5** | 0.750 | 0.891 | **0.935** | 0.804 | 0.829 |
| 0.4 | 0.750 | 0.891 | **0.935** | 0.804 | 0.829 |
| 0.2 | 0.728 | **0.902** | **0.935** | **0.806** | 0.819 |
| 0.0 | 0.696 | 0.891 | **0.935** | 0.797 | 0.804 |

And on the hard slice, where the first stage has the most room:

| alpha | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.526 | 0.816 | 0.868 | 0.603 | 0.674 |
| 0.6 | **0.632** | 0.816 | 0.868 | 0.638 | **0.731** |
| **0.5** | 0.605 | 0.816 | 0.895 | 0.642 | 0.725 |
| 0.2 | 0.579 | **0.842** | 0.895 | **0.660** | 0.719 |
| 0.0 | 0.526 | 0.816 | **0.921** | 0.656 | 0.696 |

This remains the largest single-increment gain the project has measured, and it
has now survived three rounds of corpus growth: at alpha=0.5, hit@1 goes
0.685 -> 0.750 and MRR 0.787 -> 0.829, and on the hard slice hit@1 goes
0.526 -> 0.605 while hit@5 *rises* 0.868 -> 0.895. It is also the only lever
here whose measured effect has *grown* with the corpus rather than drifting: the
all-slice hit@1 gain was 0.026 at eighteen documents and is 0.065 at twenty. The
mechanism is exactly the one predicted - it converts recall the first stage
already had into precision at rank one, and the hard slice is where it converts
the most.

The shipped alpha=0.5 is no longer the best row on either slice; alpha=0.6 now
takes hit@1 and MRR on both, by one or two questions. The default is left where
it is and the alternative recorded rather than adopted, for the same reason as
everywhere else in this README: adopting it would invalidate every table
measured at 0.5, to buy a lead the significance section cannot separate from
noise.

Read the trade-off honestly, and note that it has *unsharpened*. At eighteen
documents leaning harder on the second stage (alpha <= 0.2) cost hit@1 and
hit@5 both, and this section said so. At twenty, alpha=0.2 still costs rank one
but now takes hit@3 and nDCG@5 on both slices, and alpha=0.0 takes hard-slice
hit@5 outright at 0.921. The claim that low alpha "pays at both ends" was true
of eighteen documents and is not true of twenty. What remains true across every
size is the structural half: a re-ranker reorders the pool, it cannot enlarge
it, so it can only ever move a document that was already retrieved. alpha is 0.5
by default, the stage is **off by default** and exposed as `--rerank`, on the
same principle as `--expand`, and `alpha = 1.0` reproduces the unranked ordering
exactly, which is the correctness check on the whole path.

### A third retriever: character n-grams, and what it cost the thesis

Everything above this point fuses two ranked lists. It is worth being precise
about what those two lists are: BM25 and TF-IDF are two *scoring functions*
over one **representation**. Both call the same `tokenize`, both index whole
words, and both are therefore blind to exactly the same things - `quantise` and
`quantisation` are unrelated symbols to either of them, so is `reranking`
against `re-ranking`, and one transposed letter is a total miss. Fusing two
retrievers that share a blind spot cannot see into it, which is the structural
reason the `rrf_k` sweep found nothing to tune: the two lists agree on 0.861 of
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
0.838 to 0.648, which is the mechanism working: the third list is contributing
a genuinely different opinion rather than a paraphrase of an existing one.
`hybrid2` is the shipped two-list fusion, `ngram` is the n-gram list alone and
`hybrid3` is all three fused.

**all 132 questions, k=5**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid2 | 0.727 | 0.864 | 0.924 | 0.850 | 0.777 | 0.802 |
| **ngram** | **0.780** | **0.939** | **0.962** | **0.907** | **0.838** | **0.858** |
| hybrid3 | 0.773 | 0.902 | 0.932 | 0.865 | 0.811 | 0.840 |

**hard slice (59 questions, 37 with more than one relevant document)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid2 | 0.559 | 0.763 | 0.847 | 0.681 | 0.591 | 0.672 |
| **ngram** | **0.661** | **0.881** | **0.915** | **0.791** | **0.701** | **0.768** |
| hybrid3 | 0.627 | 0.831 | 0.864 | 0.715 | 0.648 | 0.729 |

**The result is the largest this project has measured, and it is not the result
the experiment was designed to find.** The n-gram list was added as a third
input to the fusion. It beats the fusion - alone, on every column of both
slices. Against `hybrid2` on the hard questions it takes hit@1 from 0.559 to
0.661 and recall@5 from 0.681 to 0.791, and its hit@5 of 0.915 against 0.847 is
four of fifty-nine questions where the answer was simply not in the top five
before.

**And `hybrid3` lands between its inputs, which is the finding worth keeping.**
Fusing the strong list with the two weaker ones is *worse than the strong list
alone* on every single column of both tables above - at eighteen documents it had
at least tied at rank one on the hard slice, and from twenty onwards it no longer
does. That is exactly where an unweighted average of unequal inputs should land,
and it is the limit of RRF stated plainly: `1 / (rrf_k + rank)` gives every list
the same vote regardless of how good that list is, so a third opinion helps when
the lists are comparable and dilutes when they are not. The project's working
assumption from the first commit - that adding retrievers to the fusion is how
retrieval gets better - is the thing this table refutes. What made retrieval
better was changing the representation; the fusion then gave part of it back.

**One cell now contradicts that, and it is the first time any has.** The easy
slice is not printed above, and at twenty-six documents it is the one place
where `hybrid3` beats *both* of its inputs: hit@1 of 0.890 against 0.877 for
`ngram` alone and 0.863 for `hybrid2`. At twenty-four documents the same cell
read 0.864 / 0.879 / 0.833 and the fusion sat between them as everywhere else.
So the exception is new, and it is exactly one question wide - 0.013 across 73
easy questions - which is far too little to carry the claim its position would
otherwise make. It is recorded because "fusion never beats its best input" was
stated here without qualification and is now false in one cell, not because one
question has overturned anything. Every other cell in all three slices still has
`hybrid3` between its inputs.

Two cautions before anyone reads this as "character n-grams beat words". The
first is arithmetic: 132 questions, so the 0.053 gap in hit@1 is seven questions,
and the hard slice is 59, where one question is 0.017. The second is structural,
and it matters more. The mechanism that wins here - partial credit for shared
substrings - is the same mechanism that should *lose* on a corpus of lexically
similar documents, because n-grams blur distinctions that whole words preserve.

**That prediction was tested directly at twenty documents, and it failed.** The
eighteen-document revision of this section recorded a monotonically narrowing
hard-slice hit@1 margin over `hybrid2` - +0.136 at 14 documents, +0.077 at 16,
+0.031 at 18 - and stated that "two more rounds of that trend would erase it"
and that "the experiment that would settle it is a corpus with far more
near-neighbours in it". The two documents added at twenty were chosen for
exactly that: `index-compression.md` shares the word "compression" with
`vector-compression.md` and the words "index", "postings" and "identifier" with
`inverted-index.md`; `dynamic-pruning.md` shares postings-list vocabulary with
both. The margin did not narrow. It went back up, to **+0.079**, and the trend
line is broken.

The reason is a distinction the original prediction blurred, and it is the more
interesting result. Those two documents are near-neighbours at the *word* level,
which is what hurts BM25 and TF-IDF - and it did, hard: hybrid2's hard-slice
hit@1 fell 0.594 -> 0.526. They are not near-neighbours at the *substring*
level, because their distinctive content words (`variable-byte`, `PForDelta`,
`WAND`, `MaxScore`, `Block-Max`, `pivot`) are morphologically unlike anything
else in the collection, so the n-gram list barely noticed them, falling only
0.625 -> 0.605. Adding topically-adjacent documents therefore *widens* the
n-gram lead rather than closing it. The corpus that would actually stress
character n-grams is one with shared morphology and divergent meaning - many
inflections and compounds of the same stems across documents that answer
different questions - which is a different and harder document set to write than
"two more pages on adjacent topics". The honest claim remains "on this corpus,
character 4-grams retrieve better than words", and the experiment that would
settle it is now specified more precisely than it was.

It is therefore **off by default** and exposed as `--ngram N`, on the same
principle as `--expand` and `--rerank`, and for the same reason `150:40`
chunking was recorded rather than shipped: every other table in this README was
measured against the two-list fusion, and quietly changing the default would
invalidate all of them. Unlike those flags it is index-time, not query-time -
it fits a third index - so a saved index cannot gain one by flag, and `--ngram`
against an index built without it fails loudly rather than answering from a
two-list fusion nobody asked for.

The cost is real and small in absolute terms: 4,929 gram features against 1,864
word features (2.6x), and fitting the third index takes about 22 ms against
about 4 ms for TF-IDF on this corpus. Denser vectors are also why the list is
noisier per-match - most English 4-grams are common, so everything matches
everything a little - which is what makes the standalone result surprising
rather than obvious.

### Weighted RRF: acting on "the third list is the best list"

The n-gram section ends on an uncomfortable number. The character n-gram list
*alone* scores nDCG@5 0.853, while the three-list fusion that contains it
scores 0.814. Textbook RRF weights every list identically, so the fusion is
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

Sweeping the weight on the n-gram list, over all 132 gold questions at k=5:

| weights | hit@1 | ndcg@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| uniform (plain RRF) | 0.773 | 0.811 | 0.865 | 0.840 |
| `ngram=2` | 0.803 | 0.826 | 0.876 | 0.855 |
| `ngram=3` | 0.811 | 0.831 | 0.880 | 0.863 |
| **`ngram=4`** | **0.818** | 0.833 | 0.880 | **0.866** |
| `ngram=6` | 0.811 | 0.832 | 0.884 | 0.865 |
| ngram only (control) | 0.780 | **0.838** | **0.907** | 0.858 |

The gain concentrates in the hard slice, which is where fusion was supposed to
earn its keep: nDCG@5 0.648 → 0.694 and hit@1 0.627 → 0.729 at `ngram=4`.

Two honest readings of that table, both of which matter more than the win:

**The control holds the ordering and reach columns, and has now done so at five
corpus sizes out of six.** At 16 documents `ngram only` beat every weighted
fusion on both nDCG@5 and recall@5. At 18, `ngram=4` took nDCG@5 by 0.003 - a
lead this section correctly refused to call a win, noting the interval crossed
zero. At 20 and 24 the control led again on nDCG@5, and at 26 it still does
(0.838 against 0.833) while holding recall@5 outright and by a wide margin
(0.907 against 0.880). Weighting buys rank one and MRR; it has never been shown
to buy either ordering or reach against the list it is weighting up. The split
is consistent across five corpus sizes - fusion reorders what it has better than
its best input at rank one, and retrieves less - and the 0.003 nDCG@5 lead at
eighteen documents is now visibly the noise it was flagged as.

The two margins are quoted rather than asserted, because a claim that has
survived five corpus sizes deserves checking against the possibility that it is
shrinking slowly rather than holding. Against the immediately preceding
measurement at twenty-four documents, the control's nDCG@5 lead over `ngram=4`
went 0.006 to 0.005 and its recall@5 lead went 0.021 to 0.027. One narrowed by
less than a question, the other widened by less than a question, and the reading
is unchanged: on this evidence the fusion does not catch the control on either
column, and neither does it fall further behind.

**One comparison in this repo now survives multiple-comparison correction, and
it is not this one.** See the significance section below: across 150 tests,
`hybrid3` beating `hybrid2` on nDCG@5 is the single result that clears Holm at
alpha=0.05. That is "adding the third list to the fusion helps", not "weighting
it up helps" and not "the fusion beats the third list alone" - both of those are
still NOT SHOWN. So the shipped default stays uniform. A non-significant delta
measured against a control that wins two of four columns is not grounds for
changing what the engine does by default; it is grounds for a knob and a table.

### `rrf_k` was not inert. It had nothing to arbitrate.

The tuning section above closed with a claim it could not test: that `rrf_k`
"would start to matter with more retrievers in the fusion, or with retrievers
that disagree more than BM25 and TF-IDF do here - and the honest version of
that claim is that this corpus cannot demonstrate it. **It can now**: adding a
third, structurally different list moves `rrf_k = 1` from 14 changed orderings
to 81. The third list is the
experiment that demonstrates it. Same corpus, same gold set, same sweep, one
more ranked list:

| rrf_k | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR | changed (2 lists) | changed (3 lists) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.750 | **0.917** | **0.939** | 0.809 | 0.831 | 14 | **81** |
| 5 | 0.765 | 0.902 | 0.932 | **0.811** | 0.835 | 3 | 58 |
| 10 | 0.765 | 0.902 | 0.932 | **0.811** | 0.835 | 2 | 38 |
| 20 | **0.773** | 0.902 | 0.932 | **0.811** | **0.840** | 1 | 22 |
| **60** (default) | **0.773** | 0.902 | 0.932 | **0.811** | **0.840** | 0 | 0 |
| 200 | **0.773** | 0.902 | 0.932 | **0.811** | **0.840** | 0 | 9 |
| 1000 | **0.773** | 0.902 | 0.932 | **0.811** | **0.840** | 0 | 9 |

With two lists, every value at or above 60 returns byte-identical orderings on
all 132 questions and the accuracy columns move by at most 0.001. With three,
`rrf_k = 1` reorders **81 of 132** - again about three fifths of the gold set -
and the sweep develops a shape. The contrast is the whole point of running both,
and it has widened again with the corpus: 14 against 81 is the sharpest
separation this comparison has shown.

The shape itself keeps flipping between corpus sizes, which is the second
lesson. At sixteen documents small `rrf_k` beat the default on reach; at
eighteen the default and everything above it took hit@5, nDCG@5 and MRR; at
twenty `rrf_k = 1` took hit@3 and hit@5 while giving up rank one; at twenty-four
it kept hit@5 alone and gave up both hit@1 and hit@3. At twenty-six it has
swung back to taking hit@3 **and** hit@5 while still giving up rank one (-0.023,
three questions) and MRR (-0.009), and the default now ties the top of the
nDCG@5 and MRR columns rather than conceding nDCG@5 by 0.001. Five corpus sizes,
five different shapes, every reading one to four questions wide. The stable part
across all five is the direction of the trade at the extreme end - small
`rrf_k` buys depth and sells rank one - and nothing finer than that has held.

The point is not the gain, it is that the earlier "there was nothing to tune"
was a statement about the *inputs* rather than about the knob, and it took a
genuinely different list to tell those two explanations apart. Note that the
largest values still change orderings rather than none - 9 here, against 3 at
twenty-four documents - because on a few questions the ranks are far enough
apart for the Borda limit to differ from the default, and adding documents adds
such questions.

### Diversity: MMR re-ranking

Accuracy metrics cannot see redundancy. A top-5 made of five paraphrases of one
passage scores exactly the same as five complementary ones, but it is worth far
less to an answerer. This corpus makes the problem concrete: the chunker
overlaps adjacent chunks on purpose, so neighbouring chunks are near-copies and
the unmodified top-5 spans only 3.8 of a possible 5 documents.

`ragkb/rerank.py` implements Maximal Marginal Relevance (Carbonell & Goldstein,
SIGIR 1998), selecting greedily on
`lambda * relevance - (1 - lambda) * max similarity to what is already chosen`.
Relevance is min-max normalised over the candidate pool so the subtraction is
meaningful for RRF scores near 1/60 and raw BM25 scores alike.

```bash
python -m ragkb.cli --mmr 0.7 search "how does BM25 saturate term frequency"
```

Sweeping lambda over all 132 questions at k=5 (`redundancy` is the mean
pairwise cosine within a result set; `distinct` is how many documents it spans):

| lambda | hit@1 | hit@5 | MRR | redundancy | distinct |
| --- | --- | --- | --- | --- | --- |
| off | 0.727 | 0.924 | 0.802 | 0.123 | 3.79 |
| 1.0 | 0.727 | 0.924 | 0.802 | 0.123 | 3.79 |
| 0.9 | 0.727 | 0.924 | 0.802 | 0.117 | 3.94 |
| **0.7** | 0.727 | **0.932** | **0.803** | 0.098 | 4.35 |
| 0.5 | 0.727 | 0.917 | 0.797 | 0.090 | 4.58 |
| 0.3 | 0.727 | 0.871 | 0.787 | 0.082 | 4.74 |

`lambda = 1.0` reproduces the unmodified ranking exactly, which is the cheapest
available correctness check on the implementation. Hit@1 has now been flat
across the entire sweep at five consecutive corpus sizes, which is the durable
part: re-ordering for diversity does not disturb what sits at rank one, because
MMR's first pick is always the most relevant candidate by construction.

**The flat band is no longer flat, and it broke in the useful direction.** At
16, 18, 20 and 24 documents hit@5 and hit@1 did not move at all from 1.0 down to
0.5, and this section described the 0.9-to-0.5 band as "nearly free". At
twenty-six, `lambda = 0.7` is the best row in the table on both hit@5 (0.932
against 0.924) and MRR (0.803 against 0.802) while cutting redundancy 20% and
covering 15% more documents, and `lambda = 0.5` now costs a question at hit@5
rather than none. So the band has resolved into a peak at 0.7 and a cost below
it.

That reading should be held loosely. The hit@5 gain is 0.008 - **one question** -
and the MRR gain is 0.001, which is the smallest non-zero this table can print.
The defensible statement is that diversity re-ranking at 0.7 is free on this
gold set and possibly very slightly positive, not that it helps. What makes the
row interesting is not its size but its direction: every previous revision could
only argue that MMR cost nothing, and this is the first measurement where the
best accuracy row is also the diverse one.

That the optimum has moved at all across corpus sizes - 0.5 at fourteen
documents, then four revisions of a flat band, now a peak at 0.7 - is the
argument for shipping the sweep rather than a tuned constant, so re-ranking
stays **off by default** and the trade-off is a flag rather than a hidden number.

One caveat that the holdout section adds and this table cannot see: `off` wins
here on the full gold set, but it is *not* what a cross-validated pick chooses on
most splits, and that gap now costs 0.004 of nDCG@5. The sweep below prices it.

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

Twenty-two of the one hundred and thirteen (question, relevant document) pairs
are missed at k=5. The tail of that table, and the whole of it in
`python evals/failures.py`:

| id | slice | document | hybrid | bm25 | vector | coverage | category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h15 | hard | rank-fusion | 7 | 8 | 7 | 0.62 | near-miss |
| h18 | hard | rag | 8 | 8 | 7 | 0.54 | near-miss |
| h20 | hard | reranking | 12 | 13 | 11 | 0.37 | deep-miss |
| h22 | hard | reranking | 10 | 10 | 10 | 0.28 | near-miss |
| h26 | hard | multi-hop-retrieval | 7 | 7 | 6 | 0.33 | near-miss |
| h27 | hard | domain-shift | 7 | 8 | 7 | 0.50 | near-miss |
| h27 | hard | evaluation | 12 | 16 | 9 | 0.30 | deep-miss |
| h28 | hard | domain-shift | 6 | 6 | 6 | 0.60 | near-miss |
| h30 | hard | embeddings | 10 | 10 | 10 | 0.38 | near-miss |
| h32 | hard | negative-mining | 8 | 6 | 9 | 0.71 | near-miss |
| h33 | hard | inverted-index | 8 | **5** | 10 | 0.44 | **fusion-loss** |
| h35 | hard | vector-compression | 10 | 11 | 9 | 0.47 | near-miss |
| h36 | hard | dynamic-pruning | 7 | 7 | 7 | 0.54 | near-miss |
| h37 | hard | dynamic-pruning | 11 | 11 | 10 | 0.32 | deep-miss |

Two of the twenty-two are fusion losses, and the new one is the cleanest example
the project has produced. In h33 BM25 ranked `inverted-index` at 5 - inside k -
and the fusion pushed it out to 8, because TF-IDF had it at 10 and outvoted the
retriever that was right. RRF is doing what an unweighted vote does when one
voter is right and the other is confidently wrong. That is a real cost of
fusion, it is invisible in any averaged table, and it is exactly the kind of
thing growing the corpus is meant to expose - h33 was written for this
increment and landed on it immediately.

The other twenty still are not fusion's fault: in most of them the fused rank is
identical to both component ranks, and where it differs, neither component had
the document inside k either. The `coverage` column - the share of the
question's distinct terms that occur anywhere in the document - says why. It
averages 0.43 and is below 0.40 on nine of the twenty-two. These are mostly
hard-slice questions written to avoid the target document's own vocabulary, and
both word retrievers index whole words, so they are two scores over one
representation that cannot see what neither can match.

Which lever recovers them is then an empirical question rather than an argument,
so the script re-runs every failing pair under all four (`*` marks a rank inside
k=5):

| id | document | expansion | rerank | mmr | hybrid3 | ngram-only |
| --- | --- | --- | --- | --- | --- | --- |
| q15 | evaluation | 9 | 14 | 12 | 8 | 1* |
| q26 | citations | 6 | 5* | 6 | 3* | 1* |
| h01 | embeddings | 20 | 20 | 20 | 20 | 20 |
| h02 | bm25 | 13 | 17 | 16 | 16 | 6 |
| h03 | tfidf | 18 | 20 | 19 | 15 | 4* |
| h05 | chunking | 15 | 16 | 15 | 16 | 17 |
| h05 | evaluation | 10 | 15 | 16 | 15 | 13 |
| h06 | rag | 2* | 6 | 9 | 4* | 2* |
| h15 | rank-fusion | 4* | 5* | 9 | 5* | 4* |
| h18 | rag | 3* | 9 | 12 | 6 | 4* |
| h20 | reranking | 18 | 12 | 12 | 9 | 5* |
| h22 | reranking | 15 | 10 | 10 | 6 | 2* |
| h26 | multi-hop-retrieval | 11 | 8 | 7 | 8 | 5* |
| h27 | domain-shift | 7 | 8 | 6 | 7 | 2* |
| h27 | evaluation | 11 | 12 | 12 | 12 | 11 |
| h28 | domain-shift | 7 | 6 | 6 | 3* | 2* |
| h30 | embeddings | 3* | 11 | 11 | 8 | 8 |
| h32 | negative-mining | 4* | 6 | 8 | 6 | 6 |
| h33 | inverted-index | 9 | 7 | 8 | 7 | 6 |
| h35 | vector-compression | 14 | 9 | 10 | 8 | 3* |
| h36 | dynamic-pruning | 7 | 7 | 6 | 6 | 6 |
| h37 | dynamic-pruning | 13 | 11 | 11 | 10 | 12 |
| | **recovered** | 5/22 | 2/22 | 0/22 | 4/22 | **12/22** |

**The character n-gram list alone recovers twelve of the twenty-two failures the
word retrievers share.** It is the only lever in the repo that changes the
*representation* rather than re-scoring or re-ordering the same one, and this is
the sharpest statement of that difference the project has produced: MMR now
recovers *none* and re-ranking recovers two, because re-ordering a pool cannot
admit a document the pool never contained. Both were already known to be
pool-bound in principle; here it is in the only arithmetic that can show it, and
MMR's row going 1/16 to 0/22 is that principle tightening rather than a
regression.

Expansion is the row that keeps moving with the corpus. At 14 documents it
recovered one failure; at 16, three; at 18, four; at 20, five (h06, h15, h18,
h30, h32), all at the shipped `alpha=0.8`. It is a lever with wide variance
whose good runs and bad runs are both properties of the feedback pool, not of
the code - which is the same instability that broke a test this increment.

It also explains a result from the section above that had been left as a
curiosity. The three-list table showed `ngram` beating `hybrid3` on the hard
slice's ordering columns (nDCG@5 0.712 vs 0.656, recall@5 0.842 vs 0.750), which
reads as a fusion that dilutes its best member. This table is the same fact at
question granularity: on the pairs where the word lists both fail, the n-gram
list carries twelve rescues and fusion averages two thirds of them back out
against two word lists that ranked the document fifteenth.

What survives every lever is the backlog, which has grown from five pairs to
eight:

```
nothing in the repo recovers these:
  h01 -> embeddings (rank 20, coverage 0.28): If someone writes automobiles but
         the page only ever says cars, what kind of search still finds it?
  h02 -> bm25 (rank 17, coverage 0.38)
  h05 -> chunking (rank 16, coverage 0.39): How do I decide how much text to put
         in each piece, and how would I know the choice was right?
  h05 -> evaluation (rank 15, coverage 0.33)
  h27 -> evaluation (rank 12, coverage 0.30)
  h33 -> inverted-index (rank 8, coverage 0.44): When the entries are already in
         ascending order, what is cheaper to write down than the entries
         themselves, and why do the terms everybody searches for benefit most?
  h36 -> dynamic-pruning (rank 7, coverage 0.54)
  h37 -> dynamic-pruning (rank 11, coverage 0.32)
```

`h01` is the honest one, and it is dead last of twenty under all five
configurations. The question asks for the retrieval method that survives a
synonym, using a car/automobile analogy the corpus never uses; the answer is in
`embeddings.md`, which never says either word. Character n-grams fix
*morphology* - "controls" against "control" - not synonymy, so the one lever
that rescues the other twelve cannot touch this. Nothing short of a semantic
representation will, which makes `h01` a fair statement of where a
dependency-free design actually stops rather than a bug to be tuned away. `h05`
is a different shape: one question with two relevant documents, both stuck in
the rank 15-17 band, where every lever moves it a place or two and none moves it
eleven. It gets *worse* with each corpus growth, which is what a rank-based
backlog does when the collection grows and the question still matches nothing.

**The three new entries are the useful ones, and they say something the previous
backlog could not.** h33, h36 and h37 all target the two documents added this
round, and unlike most of the backlog they are not recovered by the n-gram list
either - it ranks them 6, 6 and 12. That is the first time this project has
produced failures the representation change does not fix, and the reason is the
one the n-gram section now argues: these questions fail because two documents
share topic vocabulary, not because the query and the document share none. h33
in particular is a fusion loss with coverage 0.44 whose gold document BM25 alone
had at rank 5 - so the pool contained it, one retriever found it, and the
architecture threw it away. That is a ranking problem rather than a
representation one, and it is the most actionable thing in this backlog.

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

Take `h18`, one of the seven questions hybrid misses outright at k=5. Its gold
document is `rag.md`, and `--doc` explains a chunk whether or not it was
retrieved, which is the only way to ask why the right answer lost:

```
$ python -m ragkb.cli explain \
    "Why can what a system knows be changed without any training at all?" \
    --doc rag --terms 5

#11  [rag]  rag::1  score=0.02857

fusion (rrf_k=60):
  bm25        rank 10  1/(60+10) = 0.014286
  vector      rank 10  1/(60+10) = 0.014286
  total                          0.028571

bm25 score 5.90266 (top 5 of 5 matched terms)
  term                    qw      tf      df       idf     contrib
  system               1.000       1       8     2.393     2.25038
  be                   1.000       2      34     0.992     1.35547
  can                  1.000       3      40     0.831     1.33852
  at                   1.000       1      35     0.963     0.90585
  a                    1.000       2      89     0.038     0.05244
  missing from this chunk: all, any, changed, training, what, why, without
  not in the corpus at all: knows
```

Every point `rag.md` scored came from `system`, `be`, `can`, `at` and `a`. Not
one content word of the question contributed anything. The document that beat it
to rank one, `citations.md`, won on the same kind of vocabulary: `system`,
`without`, `what`, `can`, `be`. So this is not a ranking that needs tuning; both
documents were ranked on function words because the question and the corpus
share no content words, and the retriever had nothing else to go on.

**One word in this trace moved between corpus sizes, and it is a good
advertisement for the tool.** At eighteen documents the last line read `not in
the corpus at all: changed, knows`, and the paragraph above said both words
"carry its meaning" and are absent. `dynamic-pruning.md` happens to contain the
phrase "could not have changed it", so at twenty documents `changed` has crossed
from the unfixable line to the fixable one - the corpus now holds it, this chunk
does not - without anyone intending that and without any metric moving. Only
`knows` remains genuinely absent. A diagnostic that re-derives which side of that
boundary each term is on, every run, is the reason this was caught rather than
left as a confidently wrong sentence.

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
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.837 |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.837 |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.837 |

`cited gold` is the only column that uses the labels: the share of answers
citing a document the gold set marks relevant. It fell from 0.900 to 0.792 when
the hard questions were added - the correct direction, since retrieval misses
more of them, so more answers are built from the wrong passages - recovered to
0.857 at fourteen documents, dropped to 0.812 at sixteen, recovered to 0.833 at
eighteen and sits at 0.837 at twenty. Five moves, all small, none of them caused
by a change to the answerer: this column tracks the corpus and the gold set, not
the generation code. The grounding and citation columns stayed at 1.000
throughout, which is the distinction the measure exists to draw - the answerer is
not less faithful when this number falls, it is being handed worse evidence. The
k-dependence noted at eighteen documents (0.846 at k=3 falling to 0.833 at k=4)
has vanished again: at twenty the column is flat across k, so the question whose
gold citation was displaced by an extra passage no longer is.

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
twenty questions the corpus genuinely does not cover, split deliberately:

- **off-domain** (6) - Kubernetes, Raft, the GIL. Different subject entirely.
- **near-domain** (14) - dense-passage-retrieval training objectives, sharding a
  search cluster, click models, which engine has the fastest Block-Max WAND,
  bytes per posting in a named codec. Information retrieval questions this
  corpus happens not to answer.

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

The two near-domain questions added alongside `index-compression.md` and
`dynamic-pruning.md` were written to survive the same check by construction:
`u19` asks which open source engine has the fastest Block-Max WAND and `u20` asks
the bytes per posting of a named codec. `dynamic-pruning.md` explains Block-Max
WAND and `index-compression.md` explains posting encodings, but the corpus names
no engine, no product and no measured byte count anywhere - so both questions sit
squarely inside the corpus's vocabulary and outside its content, which is the
hardest kind of near-domain probe to build.

```bash
python evals/abstention_eval.py
```

Mean score: **0.462** answerable, **0.286** near-domain, **0.100** off-domain.

| threshold | answer rate | abstain off | abstain near | accuracy | balanced | caught misses | lost hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 1.000 | 0.000 | 0.000 | 0.821 | 0.500 | 0.000 | 0.000 |
| 0.20 | 0.989 | 0.667 | 0.214 | 0.875 | 0.670 | 0.000 | 0.012 |
| 0.24 | 0.967 | 0.833 | 0.357 | 0.884 | 0.734 | 0.111 | 0.024 |
| 0.26 | 0.967 | 1.000 | 0.357 | **0.893** | **0.759** | 0.111 | 0.024 |
| **0.28** (shipped) | 0.913 | 1.000 | 0.429 | 0.857 | 0.757 | 0.111 | 0.084 |
| 0.30 | 0.859 | 1.000 | 0.500 | 0.821 | 0.754 | 0.333 | 0.120 |
| 0.36 | 0.663 | 1.000 | 0.714 | 0.688 | 0.732 | 0.889 | 0.277 |
| 0.50 | 0.402 | 1.000 | 1.000 | 0.509 | 0.701 | 1.000 | 0.554 |

**The shipped 0.28 has been the balanced-accuracy optimum, then not, and is now
within 0.002 of it.** At sixteen documents it was the optimum; at eighteen it was
beaten by 0.24 and 0.30 and sat in a dip between them; at twenty the optimum is
0.26 at 0.759 against 0.28's 0.757 - a two-thousandth of a point, well under one
question. The instability is real but it is oscillation around a plateau rather
than drift, and three revisions of watching it wander is a better argument for
leaving the constant alone than any single sweep would be.

The mean scores show what the corpus did: the answerable mean rose 0.449 ->
0.462 and the near-domain mean rose 0.262 -> 0.286, while off-domain barely moved
(0.096 -> 0.100). The two new near-domain questions are built from corpus
vocabulary, so they land high and pull that mean up - which is the file getting
harder on purpose, and it shows in the `abstain near` column falling from 0.583
to 0.429 at the shipped threshold. The gate did not get worse; the probes got
better.

The default is left at 0.28 rather than re-tuned, for the same reason as
everywhere else in this README: a 0.002 balanced-accuracy difference on 112
questions is well inside one question, and re-tuning a constant every time the
corpus grows is how a project ends up reporting its own selection noise. The
candidate is recorded; `--abstain` takes any threshold.

The last two columns are the ones worth arguing about. Nine of the 92
answerable questions are ones retrieval already misses at k=4 - it returns
chunks, none from a relevant document - and on those the ideal gate abstains
too, because answering from evidence that does not contain the answer is the
same failure whether or not the corpus *could* have answered it. At 0.28 the
gate declines 11% of those misses and 8% of the questions retrieval got right.
Push to 0.36 and it catches 89% of them, at the cost of throwing away 28% of
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
passage retrieval trained with"*, at 0.42. Every content word in it appears in
`embeddings.md` and `negative-mining.md`, both of which discuss dense retrieval
and its training - the corpus simply never names the loss. No lexical signal can
tell those apart, and saying so is more useful than tuning the threshold until
that one row flips. Its score has now barely moved across two corpus growths
(0.42 -> 0.41 -> 0.42), which is worth noting: the gate's near-domain failures
are not a coverage problem it could grow its way out of. The newest probe `u19`
joined it immediately at 0.38, wrongly answered at the shipped threshold, for
exactly the same reason - every word in it is in the corpus and the fact it asks
for is not.

### Is the "hard slice" a real slice? Auditing the partition before the deltas

Every table above this line that says *hard slice* is computed over the
questions carrying `"difficulty": "hard"` in `evals/gold.jsonl` - the expansion
sweep, the re-ranking sweep, the weighted-RRF table, `evals/holdout.py --slice
hard`. That label was **assigned by hand, by the same person who wrote the
corpus**, and nothing in the repo had ever checked whether it corresponds to a
property of the question rather than to a mood on the afternoon it was typed. A
partition carrying that much weight deserves the same treatment the next
section gives the deltas.

```bash
python evals/difficulty.py
python evals/difficulty.py --metric ndcg@5
```

`ragkb/difficulty.py` measures the one property the retrievers actually depend
on: **containment**, the share of a question's distinct content terms that
occur anywhere in its own relevant documents, weighted by BM25's own IDF so
matching `okapi` counts for more than matching `parameter`. It is a ceiling
diagnostic, not a score. A word index gives weight exactly zero to a term the
answer does not contain, so containment bounds what lexical matching *can* do
on a question - and it is precisely the quantity the character n-gram list was
added to attack, which makes it the right axis to stratify that list's gain
along.

**The label survives.** Hand-`hard` questions really do have lower containment,
and not marginally:

| label | n | containment | idf-weighted | mean terms |
| --- | --- | --- | --- | --- |
| easy | 66 | 0.690 | 0.614 | 6.8 |
| hard | 52 | 0.480 | 0.327 | 11.8 |

A label-shuffle permutation test - unpaired, because these are two disjoint
sets of different questions and there is nothing to pair - puts both columns
below 5e-05 at 20000 resamples. That is a real vindication rather than a
formality: the label could easily have been noise, and the honest thing was
that nobody knew.

**But it is not one axis.** Crosstabbed against the size of the answer key, the
label turns out to be a proxy for it as much as for anything lexical:

| label | multi-document | single-document |
| --- | --- | --- |
| hard | 31 | 21 |
| easy | 0 | 66 |

Fisher's exact test gives p = 7.1e-15, and the interesting cell is the zero:
*every* multi-document question in the gold set is labelled hard, and no easy
question has more than one relevant document. So a hard-slice number is not a
lexical-difficulty number. It is a blend of a lexical effect and a
"find-two-documents-instead-of-one" effect, and the two have to be separated
before either can be quoted. Holding the answer-key size fixed separates them,
and the spread it exposes is wide (hit@1, k=5):

| slice | n | bm25 | hybrid2 | hybrid3 |
| --- | --- | --- | --- | --- |
| hand hard (all) | 52 | 0.519 | 0.558 | 0.712 |
| hand hard, single-document | 21 | 0.333 | 0.333 | 0.571 |
| hand hard, multi-document | 31 | 0.645 | 0.710 | 0.806 |

The reported 0.519 is an average over two sub-populations that differ by 0.312
- the single-document hard questions are nearly twice as hard as the
multi-document ones the label groups them with. Nothing above this line is
wrong because of that, but "the hard slice" has been doing duty as a single
population and it is two.

**The payoff: the third list's gain lands exactly where its mechanism says it
must.** Stratifying along measured containment instead of the hand label, on
single-document questions so the multi-document effect is held out (hit@1):

| idf-containment | n | bm25 | hybrid2 | hybrid3 | hybrid3 - bm25 |
| --- | --- | --- | --- | --- | --- |
| [0.00, 0.35) | 24 | 0.250 | 0.250 | 0.500 | **+0.250** |
| [0.35, 0.55) | 19 | 0.789 | 0.789 | 0.947 | **+0.158** |
| [0.55, 0.75) | 24 | 1.000 | 1.000 | 1.000 | +0.000 |
| [0.75, 1.00] | 20 | 0.850 | 0.850 | 0.850 | +0.000 |

The gain declines across the range and reaches *exactly* zero in the top two
bands. It is monotone at this cut, though that part does not survive a finer
one and the third limit below is where it breaks - the concentration at low
containment is the claim, not the monotonicity. That is the prediction the
module was written to test and it could have failed: n-grams exist to match
terms the word index scores at zero, so their
value has to be concentrated where the question's words are missing from the
answer and has to vanish where they are all present. An aggregate delta of
+0.119 over the whole gold set (+0.103 over the 87 single-document questions)
says the third list helps. This says where it helps, why, and where it provably
does nothing - and a flat profile would have meant that +0.119 was coming from
somewhere other than the lexical gap the module claims.

The `hybrid2` column is the same result from the other side, and it is blunter
than the aggregate tables managed to be: fusing BM25 with TF-IDF buys **0.000
in every single band**. The design note about two scores over one
representation not being two retrievers stops being an argument at that point;
the two lists share a tokenizer, so they share their containment blindness
band for band.

**The negative control, because a sceptic is owed it.** Hard questions are also
*longer* questions here (11.8 terms against 6.8), and containment's denominator
is the term count, so "long questions are hard" is a live rival explanation
that would make containment a restatement of length. Stratifying the same
questions by term count instead:

| terms | n | bm25 | hybrid3 | hybrid3 - bm25 |
| --- | --- | --- | --- | --- |
| [0, 7) | 27 | 0.741 | 0.852 | +0.111 |
| [7, 10) | 41 | 0.780 | 0.878 | +0.098 |
| [10, 14) | 13 | 0.615 | 0.692 | +0.077 |
| [14, inf) | 6 | 0.333 | 0.500 | +0.167 |

Flat and non-monotone - it never reaches zero and it turns back up at the end.
Length does not reproduce the profile, and length correlates with containment
at only r = -0.386, so the two are not the same measurement. The rival
hypothesis is the one that fails here, which is the only reason the containment
table above is worth believing.

Three limits, stated where they apply rather than at the end:

- **Containment is not discriminativeness.** It measures whether the question's
  terms are in its answer, not whether they are *only* there. A question whose
  every term appears in its answer and in six other documents scores 1.000 and
  is still a coin flip. The top band is where that shows: bm25 hit@1 falls from
  1.000 to 0.850 going from the third band to the fourth, which is the proxy
  failing, not the retriever.
- **The bands hold 19-24 questions each.** At those counts one question is
  worth 0.04-0.05, so the +0.250 and +0.158 are two-to-five-question effects
  and the two zeros are the load-bearing entries - they are exact, not
  rounded. Read the shape of the profile, not the digits.
- **The band edges were fixed before any metric was computed**, at 0.35 / 0.55
  / 0.75, to give comparable counts - but the profile is not indifferent to
  them, and `--edges` is how that was found out. Re-cut at two bands (0.50),
  three (0.30/0.60, 0.40/0.70) and the shipped four, the decline holds and the
  top band stays at exactly +0.000 every time. Re-cut at **five**
  (0.25/0.45/0.65/0.85) it breaks: the deltas run +0.333, +0.133, +0.042,
  +0.045, **-0.125**, so the monotonicity fails twice over in the last two
  bands. That top band holds 8 questions, where one row is worth 0.125 and the
  -0.125 *is* one row. The defensible claim is therefore the weaker one - the
  gain is concentrated at low containment and reaches zero by the top - and
  **not** strict monotonicity, which is an artifact of a cut coarse enough to
  hide the noise. Four bands over 87 questions was already the sweep most
  likely to be fitting itself; this is it caught doing so.

### How much of any of that is real? Error bars on the ablation table

> **Stale (106 questions, preamble 92).** Every figure in this section was
> measured before the gold set reached 118. The `W/L/T` columns sum to 106 and
> the sentence below still prices a question at the n=92 rate of 0.011; it is
> 0.008 now. `python evals/significance.py` prints the current table. The
> machinery and the argument are unaffected - the numbers are not, and the
> `hybrid2 vs bm25` verdict quoted above is one of the readings most likely to
> move, which is why it is being re-run with the ablation rather than patched
> here.

Every table above this line reports a mean over the same 92 questions, and
several of the differences in them are one question wide. That is not a
rhetorical point: at n=92 one question is worth 0.011 of any per-query metric,
and the hard slice - 38 questions - cannot resolve anything finer than 0.026.
Nothing in the repo could distinguish "this lever helps" from "this lever moved
one row", which made the whole evaluation section easy to over-read.
`evals/significance.py` closes that hole:

```bash
python evals/significance.py
```

**This section vindicated itself between two revisions, which is the best thing
it has done.** At eighteen documents the evaluation section led with hybrid
beating BM25 on every hard-slice column, and this section answered with the
`W/L/T` column: the sweep rested on *one question* for three of five metrics,
with a `floor` of 1.000, meaning those rows could not have been significant no
matter how the data fell. Fourteen questions later the sign has flipped and
hybrid no longer leads at all. The test said the result was not there; the
larger gold set confirmed it. That is a cheaper way to be right than shipping
the fusion as a default and discovering it in production.

```
hard slice - 38 questions, one question is worth 0.026
metric    comparison                 A       B    delta              95% CI       p  p_holm   floor       W/L/T
----------------------------------------------------------------------------------------------------------------
hit@1     bm25 vs hybrid2        0.526   0.526    0.000    [+0.000, +0.000]   1.000   1.000   1.000      0/0/38
recall@5  bm25 vs hybrid2        0.697   0.697    0.000    [-0.039, +0.039]   1.000   1.000   0.500      1/1/36
ndcg@5    bm25 vs hybrid2        0.604   0.603    0.001    [-0.020, +0.021]   1.000   1.000   0.062      3/2/33
mrr       bm25 vs hybrid2        0.674   0.674    0.000    [-0.013, +0.013]   1.000   1.000   0.500      1/1/36
```

The hard-slice comparison the project was built around is now a row of exact
zeroes with `0/0/38` on hit@1 - fusion and BM25 return the same verdict on every
single hard question. A tie measured this precisely is a more informative result
than the one-question win it replaced.

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

The verdict on the word-level part of the evaluation section is unchanged:
**none of it is significant.** Among these three word-level methods, not one
comparison at any k, on any slice, has a 95% interval that excludes zero. That
is the honest headline, and it is more useful than the tables it qualifies.

**One comparison in the whole repo now survives the familywise correction, and
it arrived with the gold set rather than with any change to the code.** Across
150 tests, `hybrid3` beats `hybrid2` on nDCG@5 over the all slice by +0.042,
16 wins / 2 losses / 74 ties, raw p = 0.000, **p_holm = 0.045**. At 78 questions
the same comparison did not clear the correction; at 92 it does, and nothing
about the retriever changed in between. That is the first familywise-significant
result the project has produced, and it is worth being exact about what it
claims: *adding the character n-gram list to the fusion beats the two-list
fusion*. It does not claim the fusion beats the n-gram list alone - that
comparison is +0.039 the other way with p_holm 1.000, still NOT SHOWN - and it
does not rescue two-list fusion, which remains NOT SHOWN against BM25 alone
(-0.002, 3W/5L/84T).

There is a lesson in *which* comparison survived. `ngram` vs `hybrid2` on the
same metric is a much larger delta (+0.081) and does not survive, because Holm
orders by raw p and that comparison is noisier: 28 wins against 11 losses out of
39 discordant questions. `hybrid3` vs `hybrid2` moves less but almost never
backwards - 16 against 2 out of 18 - so its raw p is far smaller and it clears a
150-fold multiplier that the bigger delta does not. Consistency beat magnitude,
which is the opposite of what an accuracy table alone would suggest.

The `W/L/T` column explains the rest, and is the number worth reading first. The
three *word-level* retrievers agree on almost every question - the largest
disagreement among them is 13 questions out of 92, and one hard-slice row is
0/0/38 outright. (That ceiling turns out to be a property of comparing three
word-level methods rather than of the gold set; adding the character n-gram list
reaches 39.) The comparison's real sample size is not 92, it is the discordant
count, and every tie is a question that carries no evidence either way. Four
rounds of gold-set growth have moved the largest word-level discordant count
8 -> 10 -> 12 -> 13, lowering the best reachable p-value and changing no verdict
among the word methods at any step. That is the measured rate of return on
gold-set growth for this particular claim, and it is slow: the questions being
added are ones the three word methods mostly answer the same way, so they add
resolution to the means without adding evidence to those comparisons. The
n-gram comparisons are where the growth paid off, because that is where the
methods actually disagree. Writing questions the retrievers *disagree* on is a
different and harder task than writing more questions.

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

### The dead pointer: the one comparison that was never tested

The section above ends by saying the retrievers agree on almost every question
and that the largest disagreement anywhere is 13 out of 92. That was true, and
it was also an artefact of *which* retrievers were being compared. Until this
increment `evals/significance.py` scored exactly three methods - `hybrid`,
`bm25` and `vector` - all three of which tokenise into whole words. Meanwhile
`evaluate.py` prints an `ngram only` control and closes with:

> run `python evals/significance.py` before believing any gap above

That instruction pointed at a script which had never scored the n-gram list.
The comparison this project leans on hardest - the third list alone against the
three-way fusion, the claim that drives the whole "what it cost the thesis"
section - was the one comparison with no error bars anywhere in the repo.
`METHOD_ROUTING` now mirrors the table in `evals/robustness.py`, so both
harnesses score the same five methods through the same two engines and cannot
drift into meaning different things by `hybrid3`.

```bash
python evals/significance.py
python evals/significance.py --methods ngram,hybrid3 --metrics ndcg@5
```

**The gold set is more discriminative than the previous section concluded.**
Three word-level methods disagreed on at most 12 questions. Add the character
n-gram list and the largest discordant count is **52** - `vector` against
`ngram` on nDCG@5, 14 wins to 38 losses - while the largest among the three
word-level methods is 30. Quadrupling the evidence did not require a single new
gold question; it required comparing methods that fail differently. The earlier
reading, that gold-set growth was the only lever on discordance, was too
pessimistic: *method diversity* is the other lever, and it was free.

**And that is exactly when the multiple-comparison problem arrives.** Three
methods make 3 pairs per metric per slice; five make 10. A full run went from 45
tests to 150. At α=0.05, 150 tests are expected to produce **7.5** spurious wins
even if every method were identical, so a table that invites the reader to scan
for small p-values is no longer a table that can be read row by row.
`ragkb/significance.py` gained `holm_bonferroni`, and the harness now corrects
across every comparison in the run and prints `p_holm` beside the raw `p`. Holm
rather than plain Bonferroni because both control the familywise rate under
arbitrary dependence - and these rows are heavily dependent, sharing questions,
methods and nested slices - while Holm's step-down is uniformly more powerful,
so there is no reason to take the weaker one.

The correction earns its place immediately:

| | 132 questions | 118 questions |
|---|---|---|
| comparisons in a full run | 150 | 150 |
| expected spurious at raw α=0.05 | 7.5 | 7.5 |
| **observed** at raw p<0.05 | **42** | 44 |
| surviving Holm | **6** | 8 |

Forty-two rows cross the uncorrected threshold and six survive, which is the
whole argument for the column. Anyone reading the raw column alone would come
away with forty-two confident findings; the honest count is six, and 42 against
an expected 7.5 is what a family this size does to an uncorrected table.

**Both figures in the right-hand column are corrections, and the older text was
wrong in two incompatible ways at once.** Previous revisions of this section
carried a table reading 14 raw and 0 surviving, and a paragraph directly beneath
it reading 29 raw and 1 surviving. Those cannot both be right, and re-running
the 118-question configuration from the previous commit gives **44 and 8**, so
neither was. The provenance test added last increment catches drift in table
*captions*; it does not read the numbers inside a table or a fenced block, which
is precisely where these two sat. That is a real limit of the test and it is
worth stating next to the thing it failed to catch.

**Growing the gold set lowered the survivor count, which is the opposite of the
obvious expectation.** Fourteen more questions took Holm survivors from 8 to 6
and raw crossings from 44 to 42. More data usually buys power, so the reflex
reading is that something went wrong; what actually happened is that the
composition changed. At 118 questions seven of the eight survivors were
comparisons involving `ngram` *alone* - the standalone third list beating BM25,
TF-IDF and the two-list fusion. At 132 not one survivor involves `ngram` alone,
and all six involve `hybrid3`: the three-list fusion beating BM25, TF-IDF and
`hybrid2` on nDCG@5 and MRR. The new questions did not weaken the evidence, they
moved which comparison the evidence supports, and a fusion result replaced a
standalone-list result. A count alone would have hidden that entirely, which is
the argument for printing the survivors rather than tallying them.

The three claims the project's own prose makes are checked by name, so they
cannot quietly drift out of agreement with the measurement:

```
claims this project makes, tested on ndcg@5 / all slice
NOT SHOWN   ngram vs hybrid3     delta +0.026  p_holm 1.000  25W/15L/92T
SUPPORTED   hybrid3 vs hybrid2   delta +0.035  p_holm 0.015  25W/0L/107T
NOT SHOWN   hybrid2 vs bm25      delta +0.004  p_holm 1.000   9W/9L/114T
```

**The middle line has now changed verdict twice, in both directions, and this
section predicted both.** At eighteen documents all three read NOT SHOWN, and
the prose said of `hybrid3 vs hybrid2`: "it is defeated by the family size
rather than by the data; it is the claim most likely to survive a larger gold
set." A later revision recorded it as SUPPORTED. At twenty-four documents it
read NOT SHOWN at p_holm **0.057** - the wrong side of its own line by four
thousandths, re-measured from the previous commit rather than taken on trust -
and the power section wrote that this was "the strongest single argument in this
file for growing the gold set before trusting any of these verdicts." Fourteen
questions later it is SUPPORTED again at p_holm **0.015**, with no change to the
retriever, and its split has gone 21W/2L to **25W/0L**: every one of the
twenty-five questions the two systems now disagree on is a win for the three-list
fusion, and the two losses it carried at twenty-four documents are gone.

That is the cleanest demonstration in this file of what the whole evaluation
apparatus is for. The underlying system was not touched across those three
measurements. What moved was the amount of evidence, and the verdict moved with
it - out, back, and out again - which is exactly what a claim sitting near a
significance boundary is supposed to do and exactly why the boundary should not
be reported as though it were a property of the retriever.

The three still fail or pass for three different reasons worth telling apart.
`hybrid2 vs bm25` is a *dead* claim: 9W/9L is eighteen discordant questions
splitting exactly evenly, and a delta of +0.004 - fusing BM25 with TF-IDF does
essentially nothing here, which the earlier `rrf_k` section reached
independently and the main ablation table now shows directly. It has been
negative and is now marginally positive, and eighteen questions splitting 9-9 is
the most precise way this file can say "no effect". `hybrid3 vs hybrid2` is the
survivor. (Restricting the run to one slice with `--slices all` shrinks the
family from 150 tests to 50, which was how this comparison bought its evidence
cheaply before the gold set was large enough to pay full price.)
`ngram vs hybrid3` is the interesting failure and has stayed one: 25W/15L across
40 discordant questions is a genuine mixed result, not an absence of evidence.
On clean text the third list and the fusion trade wins, and the nDCG@5 gap - now
+0.026, down from +0.047 at twenty-four documents - is not distinguishable from
noise even as the gold set grows. The direction of that movement is worth
noting: across the last fourteen questions the fusion closed about half the gap
on the list it cannot beat, without the verdict changing.

Note what the survivor does *not* license. It says adding the n-gram list to the
fusion beats the two-list fusion. It does not say the fusion beats the n-gram
list alone - that is the line above it, still unproven and pointing the other
way - so "use three lists instead of two" is supported and "fuse rather than
just switching to n-grams" is not.

That last one is worth stating plainly because it **narrows an earlier claim in
this README.** The robustness section found the n-gram list beating the fusion
under noise by a wide margin - 0.790 against 0.596, far too large to be a
sampling artefact. The clean-text version of that same comparison does not hold
up. The defensible statement is therefore the narrower one: *under query noise*
the fusion is outvoted by its own third list, and on clean queries the two are
indistinguishable on this gold set. The two-axis reading in the robustness
section overstated the clean-text axis, and this is the measurement that caught
it.

### The other bias: every sweep above picked its own winner

`significance.py` asks whether the gap between two *named* systems is real. It
does not touch a second problem that runs through every sweep table in this
README, and that no amount of error-barring a single comparison can reach: the
top row of a sweep was **chosen by looking at the same questions it is scored
on**. The maximum of several noisy estimates is biased upward even when every
configuration is identical, so "the best fusion weighting scored 0.818 hit@1" is
not a prediction about the next 132 questions - it is the best of seven draws
from the same 132.

`evals/holdout.py` measures the size of that bias by nested cross-validation:
split the questions into folds, pick the winner on the training folds, score
that choice on the held-out fold, and repeat over twenty different splits.

```bash
python evals/holdout.py
python evals/holdout.py --slice hard --folds 4
```

```
132 questions (all slice), k=5, 5-fold x 20 splits, seed 0
one question is worth 0.008 of any per-query metric; a held-out fold is 26 questions

sweep           metric    picked          in-sample   held-out   optimism   vs default   stable
-----------------------------------------------------------------------------------------------
rrf_k           hit@1     60                  0.727      0.727     +0.000       +0.000     100%
rrf_k           ndcg@5    1                   0.777      0.776     +0.001       -0.000       0%
rrf_k           mrr       1                   0.803      0.802     +0.001       -0.000       0%
fusion weights  hit@1     ngram=4             0.818      0.802     +0.016       +0.029       0%
fusion weights  ndcg@5    ngram only          0.838      0.825     +0.013       +0.013       5%
fusion weights  mrr       ngram=4             0.866      0.853     +0.014       +0.013       0%
mmr lambda      hit@1     off                 0.727      0.727     +0.000       +0.000     100%
mmr lambda      ndcg@5    0.9                 0.777      0.775     +0.002       -0.002       0%
mmr lambda      mrr       0.7                 0.803      0.801     +0.002       -0.001       0%
```

Read `optimism` first, on the `fusion weights / hit@1` row. The weighted-RRF
table earlier in this README reports `ngram=4` at 0.818 against uniform's 0.773
and invites exactly the conclusion it looks like. Scored on questions the choice
never saw, that configuration earns **0.802** - the headline shed 0.016, two of
the six questions the sweep appeared to win.

**That penalty has now gone up, down, up, down and up across five corpus sizes**
(+0.043 at 16, +0.013 at 18, +0.032 at 20, +0.008 at 24, +0.016 at 26), and the
eighteen-document revision of this section read the middle value as progress:
"the first sweep row in the project that has survived cross-validation with its
margin substantially intact". It was not progress, it was a draw - and the
twenty-four-document value was the lowest yet, which is precisely why it should
not have been read as progress either. It has since doubled. The `stable` column
is the reason: this row sits at **0%**, meaning the winner changed on every one
of the twenty splits. An optimism estimate from a sweep whose winner is never the
same twice is itself a draw from a noisy distribution, and five measurements
reading 0.043, 0.013, 0.032, 0.008, 0.016 are the distribution, not a trend.

Two controls were supposed to make that number trustworthy rather than merely
encouraging, and they are why the script sweeps three different knobs instead of
the one it wanted to indict. **Both have now broken at least once, and both
broke in the same direction: away from the clean story.**

- **`rrf_k` no longer picks the default on every metric.** At twenty-four
  documents every row picked `60` at 100% stability with exactly zero optimism,
  and this section said the control "now agrees on every metric without the
  caveat". At twenty-six, nDCG@5 and MRR both pick `rrf_k = 1` at **0%**
  stability with +0.001 optimism, and only hit@1 still picks the default at
  100%. That is a return to the twenty-document pattern, and it restores the
  illustration the previous revision deleted: optimism and stability are
  independent, and a 0% stable row with +0.001 optimism means "nothing to
  choose", not "a choice that will not generalise". The underlying control is
  intact - `evals/tune.py` establishes `rrf_k` is inert over two lists, so there
  is nothing for selection to inflate - but the claim that the harness agrees
  without caveat lasted one increment.
- **MMR no longer picks the default either**, which breaks the sentence that
  once followed this table - "selection cannot inflate a sweep whose winner is
  the default". At twenty-four documents `picked` was `off` on all three metrics
  with `stable` already down to 15% and 30%. At twenty-six, nDCG@5 picks `0.9`
  and MRR picks `0.7`, both at 0% stability, costing 0.002 and 0.001 out of
  sample, and only hit@1 still picks `off`. The full-set winner being the
  default never made the sweep free, because cross-validation does not get to
  see the full set; now the full-set winner is not even the default on two of
  three metrics. Note that the MMR sweep table above independently found 0.7 to
  be its best row on hit@5 and MRR, so these two harnesses are agreeing, which
  is more interesting than either alone.

The `stable` column is the same finding read a different way, and it has not
stopped moving. All three `fusion weights` rows were at or near 0% at twenty
documents, where at eighteen hit@1 was 65% and mrr 95%; at twenty-four nDCG@5
recovered to 40%; at twenty-six it has fallen back to **5%** and all three rows
are at or near zero again. So the earlier reading - that stability "has moved
consistently in one direction while optimism oscillates" - does not survive, and
neither does the reading that replaced it. Both columns oscillate. That is what
a small gold set looks like, and five corpus sizes have now failed to produce a
trend in either column.

The reversal worth recording is `ngram only`, and it has now held twice running.
At sixteen documents it was the control the weighted fusion could not beat out
of sample on nDCG@5. At eighteen the holdout picked `ngram=4` on all three
metrics and the control lost even out of sample. At twenty the holdout picked
`ngram only` back for nDCG@5, and at twenty-four and twenty-six it still does -
though its stability has gone 0%, 40%, 5%, so the "only fusion row that is not a
coin flip" reading lasted exactly one increment and is withdrawn. Five
measurements, two reversals, one conclusion that has never actually changed:
**the fusion has still not been shown to beat its own best input.**
Four of the nine sweep/metric pairs have a negative `vs default` - on those, the
honest estimate of tuning is worse than shipping the constant and never
sweeping. Both the count and the membership are unchanged from twenty-four
documents: the two `rrf_k` rows and the two MMR rows, on nDCG@5 and MRR in each
case. That is the one thing in this section that five corpus sizes have not
managed to move.

The scope is limited on purpose and the module says so: folds are resampled from
one small gold set, so `optimism` is itself a noisy estimate, and this cannot
rescue a sweep from a gold set too small to distinguish anything. What it does
is price a cost that the tables had been reporting as zero. Combined with the
`floor` column above, the standard for the next retrieval claim is now explicit:
it has to survive being scored on questions that did not choose it.

### So how many questions would it take? Pricing the standing excuse

Every negative result above ends the same way, and `evals/significance.py`
prints it verbatim: *the fix is a bigger and more discriminative gold set rather
than a different test.* That sentence has been doing a lot of work for a long
time, and until now it was unfalsifiable, because nothing here could say **how
much** bigger. `python evals/power.py` prices it.

The instrument is the exact two-sided sign test. That is not an analogy for what
the repo already runs - for a binary per-query metric every discordant question
contributes a difference of exactly ±1, so the approximate-randomization test in
`ragkb/significance.py` *is* the sign test on hit@k. For graded metrics (MRR,
nDCG@5) the randomization test uses magnitude information the sign test throws
away, so these question counts are a conservative upper bound on those rows
rather than a prediction.

The parameterisation splits the two things a gold set contributes, because they
are not interchangeable and conflating them is how "just add more questions"
becomes bad advice. **Discordance rate** is how often the two methods disagree
at all - a property of how discriminative the questions are. **Win share** is
who wins when they do - the effect size, with 0.5 as the null. Only discordant
questions carry evidence; ties are invariant under the sign flips. A question
both methods already answer correctly adds to `n` and contributes exactly
nothing to the test.

What this gold set can resolve, before looking at any result: the three claim
comparisons disagree on 21.0% of questions on average, about 28 discordant pairs
out of 132. At that rate **the smallest effect 132 questions can detect at 80%
power is a win share of 0.775** - one method has to take 78% of the
disagreements. Against a genuinely small effect (win share 0.60) this gold set
has **13.1% power**. It would miss a real difference of that size nearly nine
times in ten.

Fourteen questions moved the detectable bar from 0.789 to 0.775 and the power
against a small effect from 11.9% to 13.1%. That is the going rate, and it is
worth quoting because it sets expectations honestly: growing the gold set by
twelve percent bought about one and a half points of power. The claim recovered
this increment was recovered by landing near the boundary, not by a step change
in what the set can see.

Required questions by assumed truth, which looks at no result at all and would
print the same numbers if every retriever here were replaced tomorrow:

```
 discordance   q=0.55   q=0.60   q=0.65   q=0.70   q=0.80   q=0.90
------------------------------------------------------------------
         5%     16064     4086     1840     1035      463      251
        10%      8032     2043      920      517      231      125
        15%      5355     1362      613      345      154       83
        20%      4016     1021      460      258      115       62
        30%      2677      681      306      172       77       41
```

Halving the effect roughly quadruples the questions needed - but so does
dividing the discordance rate by four, and **the discordance rate is the cheaper
lever**. This is the quantitative form of the corpus-growth strategy described
at the top of this page, where each new document is placed beside an existing
one it shares vocabulary with so that some question gets a lexically plausible
wrong neighbour. That strategy is not an aesthetic preference. Writing questions
that *separate* the methods moves the table as hard as writing four times as
many that do not.

**That claim was tested directly at twenty-two documents, then re-tested at
twenty-four with a proper instrument, and it did not survive either.**
`evals/corpus_lever.py` is that instrument. It builds the engine twice - once
on the corpus as it was, once as it is - holds the question set fixed at the
questions that predate the increment, and reports the discordance rate for each
arm. Because only the corpus differs between those two rows, the document
effect is isolated from the question effect by construction.

The first increment it was pointed at was the previous one, which added
`query-performance-prediction.md` and `passage-aggregation.md` plus fourteen
gold questions written specifically to separate `hybrid2` from `bm25`. Eight of
the fourteen were phrased entirely in a near-neighbour document's vocabulary,
avoiding the target's own terms, which is the construction that is supposed to
punish a purely lexical list. `--exclude` rolls the corpus back so that
increment can be re-measured today:

```
python evals/corpus_lever.py --exclude relevance-feedback,score-normalisation \
    --added query-performance-prediction,passage-aggregation --pair hybrid2,bm25
```

| `hybrid2` vs `bm25`, nDCG@5 | W/L/T | discordance | exact 95% CI |
| --- | --- | --- | --- |
| 20 documents, the original 92 questions | 3/5/84 | 8.7% | [3.8%, 16.4%] |
| 22 documents, **the same** 92 questions | 5/8/79 | 14.1% | [7.7%, 23.0%] |
| 22 documents, the 14 new questions only | 1/1/12 | 14.3% | [1.8%, 42.8%] |
| 22 documents, all 106 questions | 6/9/91 | 14.2% | [8.1%, 22.3%] |

The four counts reproduce the hand-computed table this section used to carry,
digit for digit, which is the reason to trust the two columns that are new. The
intervals are Clopper-Pearson, exact rather than normal-approximate, and the
third row's `[1.8%, 42.8%]` is the same range the old prose estimated in words.

**The conclusion drawn from that table was wrong, and the error was not in the
counts.** It read 8.7% rising to 14.1% as the document lever working, and
called the effect "clearly visible". But those two rows are measured on *the
same ninety-two questions*, so they are paired, and the right test asks how many
individual questions changed status rather than comparing two marginal rates.
Eight questions became discordant and three stopped; under the null that the
change was symmetric that is an exact two-sided p of **0.227**. The rate moved.
Nothing established that the corpus moved it.

The replication is worse for the claim than the re-analysis is. This increment
added `learning-to-rank.md` and `pooling-and-judgments.md`, both placed beside an
existing document they share heavy vocabulary with - the first beside
`reranking.md` and `score-normalisation.md`, the second beside `evaluation.md` -
exactly as the strategy prescribes. The document lever moved in both directions
and reached significance in neither:

| pair, nDCG@5 | 24 docs, carried 118 | 26 docs, **same** 118 | switched | exact p |
| --- | --- | --- | --- | --- |
| hybrid2 vs bm25 | 11.9% (14) | 12.7% (15) | 5 gained, 4 lost | 1.000 |
| hybrid3 vs hybrid2 | 19.5% (23) | 18.6% (22) | 2 gained, 3 lost | 1.000 |
| ngram vs hybrid3 | 32.2% (38) | 28.8% (34) | 3 gained, 7 lost | 0.344 |

Three increments, eight paired measurements, nothing significant, and the sign
has now flipped in both directions. **The honest standing of "documents are the
lever" is that it is unsupported**, and it was only ever supported by reading a
paired design as an unpaired one. Two documents moved the discordance rate by
+0.8%, -0.8% and -3.4% on the three pairs, and the largest of those is ten
questions changing hands in a set of 118 with an exact p of 0.344.

The fourteen new questions, meanwhile, landed at 21.4%, 21.4% and 42.9% -
at or above the prevailing rate on all three pairs, which is the second
increment running where question-writing out-discriminated and the corpus did
not. That is six paired measurements where the new questions beat the prevailing
rate and eight where the document lever returned nothing. Neither result is
significant on its own; taken together the
asymmetry is consistent enough to be worth acting on, and this increment's own
allocation - two documents, fourteen questions - reflects it.

What survives about documents is narrower and worth stating on its own: a new
document re-prices every question already in the set, which is a mechanism, not
a claimed direction. It can raise the discordance rate or lower it, and at this
corpus size the measurement still cannot tell which it did.

Then the three claims, priced against their own observed splits:

| claim | W/L/T | rate | obs. share | questions needed | verdict at n=132 |
| --- | --- | --- | --- | --- | --- |
| ngram vs hybrid3 | 25/15/92 | 30.3% | 0.62 | 434 | NOT SHOWN |
| hybrid3 vs hybrid2 | 25/0/107 | 18.9% | 1.00 | 41 | **SUPPORTED** |
| hybrid2 vs bm25 | 9/9/114 | 13.6% | 0.50 | - | NOT SHOWN |

**The last column is not an input to the fourth.** Required-sample-size is
derived from the binomial alone, with no knowledge of what the significance
harness concluded. At twenty-six documents the two agree three times out of
three, and the middle row is the interesting agreement: the grid says 41
questions would suffice, the set holds 132, and the harness returns SUPPORTED.
At twenty-four documents that same row said 59 questions against a set of 118
and the harness still said NOT SHOWN - the grid and the harness disagreeing,
which the previous revision read as the grid's resolution limit. They now agree,
and the disagreement is resolved in the grid's favour.

**The claim lost last increment has been recovered, and the recovery is the
whole point of the increment.** `hybrid3 vs hybrid2` read NOT SHOWN at
twenty-four documents at `p_holm` 0.057 - just the wrong side of the line its
own harness draws - and the text here said that was "the strongest single
argument in this file for growing the gold set before trusting any of these
verdicts." Fourteen questions were added, three of them discordant on this pair
and all three won by `hybrid3`, taking the split from 21/2 to **25/0**. It
returns SUPPORTED at `p_holm` 0.015. The argument was made here, acted on in the
next increment, and the prediction held.

**The top row moved the other way, and that is the honest cost of the same
fourteen questions.** `ngram vs hybrid3` needed 116 questions at twenty-four
documents and needs **434** now. Nothing got worse: the required sample
exploded because the *estimated effect shrank*, from an observed share of 0.74
to 0.62, and required sample scales roughly with the inverse square of the
distance from 0.50. The six new questions that were discordant on this pair all
went to `hybrid3`, which is what pulled the share down. So the same fourteen
questions that proved one claim made another one four times more expensive to
prove - and both movements are the gold set doing its job rather than two
different results.

Read the bottom row carefully, because it is the most useful line in the table
and the easiest to misread. `hybrid2 vs bm25` is the project's headline claim,
and its observed share is exactly 0.50 for the second increment running - the
nine discordant questions the fusion wins are matched by the nine it loses, so
the required sample diverges and the table prints a dash. That is the correct
output, not a gap: it reports "no effect worth sizing was seen here". After
twenty-six documents and 132 questions the fusion-versus-BM25 comparison has
accumulated eighteen disagreements in total and they split evenly.

Two caveats, stated plainly because the second is a genuine statistical trap.
First, an observed share near 0.50 sends the required sample to infinity, which
is correct rather than a bug: it reports "no effect worth sizing was seen here".
Second, **none of this is retrospective power.** Computing "the power I had to
detect the effect I just observed" is circular - observed power is a monotone
function of the p-value, so it restates the result and calls it a diagnosis, and
a non-significant finding always comes back with low observed power by
construction. Every figure here is conditional and prospective: *if the truth
were this, this is what it would cost to see it.* The observed splits are
hypotheses resting on fourteen to thirty-eight disagreements, which pin a
proportion to roughly a third either way at the low end, so the claims table
carries error bars far wider than the grid above it. `ragkb.power.clopper_pearson`
is the function that makes those bars explicit, and `evals/corpus_lever.py`
prints them beside every rate it reports. The grid is the finding; the claims
table is an illustration of how to use it.

### What each stage costs

> **Stale (20 documents, 92 questions).** Timings below were measured on the
> twenty-document index. `python evals/latency.py` re-runs them in about nine
> seconds. Ratios between stages are the durable part; absolute milliseconds
> were never portable across machines anyway.

Every table above answers "did it rank better?". None of them answered "what
did that cost?" - and four stages here are *optional*, which is only a real
choice once both numbers exist. `python evals/latency.py` measures the other
half: per-query wall time for each configuration, on the same 92 gold
questions, pooled over repeats so `p95` is a genuine tail over queries rather
than a mean in disguise.

```
query latency (per query, k=5)          cost against quality (k=5)
stage             median_ms   xbase       hit@1   recall@5    ndcg@5       mrr
------------------------------------------------------------------------------
hybrid (2 lists)       0.13    1.00       0.685      0.853     0.772     0.787
hybrid (3 lists)       0.24    1.80       0.739      0.886     0.814     0.827
  + mmr 0.7            1.13    8.60       0.685      0.848     0.765     0.783
  + expansion          1.36   10.42       0.576      0.880     0.745     0.722
  + rerank             1.52   11.62       0.761      0.864     0.804     0.830
  + all three          3.69   28.22       0.707      0.880     0.797     0.802
```

The ordering is the finding, and it is not the one the build order implies. The
cheapest addition in the pipeline is also the best value: the character n-gram
third list costs 1.8x a plain fused query and improves every accuracy column at
once. Re-ranking has retaken rank one (0.761 against 0.739) while costing six
times as much and still giving back recall - across three corpus sizes these two
rows have now traded the hit@1 lead twice, which is a fair summary of how close
they are. Expansion at these settings is sharply negative at rank 1, which the
sweep table already said. Stacked together the three optional stages cost 28x
and land below the 1.8x row on three of four columns. If this were a latency
budget rather than a portfolio project, the recommendation would be one line:
ship the third list, keep re-ranking behind a flag, leave the rest off.

The `xbase` multipliers have fallen again (MMR 11x -> 10x -> 8.6x, the full stack
41x -> 31x -> 28x) while the absolute milliseconds rose. That is the denominator
moving, not the numerator: a bigger corpus makes the base two-list query slower,
while the fixed per-query cost of MMR and re-ranking barely changes, so the ratio
narrows. Three corpus sizes have now confirmed the prediction this section made
at sixteen documents, which is the point of writing predictions down: a ratio
needs its base stated, and these multipliers shrink as the corpus grows.

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

The honest caveat is scale: 20 documents and 92 chunks is small enough that
these ratios describe *this* corpus, not a curve. Some of them move predictably,
and there are now three data points confirming it - MMR's work is O(k²) over a
fixed `k` and so barely grows, while first-stage retrieval grows with the corpus,
so MMR's multiplier was predicted to shrink and has, monotonically (10.85x at
sixteen documents, 9.86x at eighteen, 8.60x at twenty). Absolute milliseconds are
this machine's; the `xbase` column is the part that travels, and even it has a
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
bm25                  92     97.4%      119836       31284        0.26
tfidf                 92     97.4%      108210       31284        0.29
ngram(4)              92     99.9%      459621      115135        0.25
```

`reach` is the share of the collection a gold question touches through at least
one term: 97.4%. One common word in a question - *what*, *does*, *the* - matches
almost every chunk on its own, so the entire prize for filtering by membership
is the remaining 2.6%. For the n-gram index it is 0.1%, because a 4-gram query
reaches everything. Nor does that improve with scale - it keeps getting *worse*,
rising 96.0% at 57 chunks, 96.5% at 69, 97.0% at 80, 97.4% at 92, which is the
prediction confirmed four times: *the* appears in nearly every document at 92
chunks and at 92,000. The `taat/scan` ratio, meanwhile, has not moved off 0.26
across all four corpus sizes - a count-based number behaving like a count-based
number.

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
implemented here. The corpus now contains `dynamic-pruning.md`, which explains
exactly what is missing and why it would be rank-safe; the engine reading that
document cannot do what it describes, which is a gap worth naming rather than
letting the corpus imply otherwise.

### Robustness: the one condition the third list was added for

> **Stale (20 documents, 92 questions).** The degradation curves below predate
> the last two corpus increments. `python evals/robustness.py` re-runs them in
> about eight seconds.

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

nDCG@5, 92 questions, averaged over 7 noise draws per rate:

| method | clean | 10% | 20% | retained at 20% | sd(seeds) |
|---|---|---|---|---|---|
| `bm25` | 0.774 | 0.656 | 0.508 | 65.6% | 0.048 |
| `vector` | 0.779 | 0.660 | 0.512 | 65.7% | 0.046 |
| `ngram` | 0.853 | 0.835 | 0.790 | **92.6%** | 0.022 |
| `hybrid2` | 0.772 | 0.661 | 0.510 | 66.1% | 0.049 |
| `hybrid3` | 0.814 | 0.724 | 0.596 | 73.2% | 0.044 |

The design claim survives contact with a measurement, and by a wide margin. The
two word-level indexes lose better than a third of their ranking quality, and
they lose it *together* - 65.6% and 65.7%, which is what "miss the same
misspelling" looks like when you put a number on it. Fusing them changes
nothing (66.1%): reciprocal rank fusion can only reorder what its inputs
retrieved, and two lists that failed for the same structural reason have no
disagreement for it to arbitrate. The character index barely notices, holding
92.6%, because a one-character typo leaves most of a word's 4-grams intact.
hit@1 tells the same story more bluntly: at 20% noise BM25 answers 38% of
questions correctly at rank one and the n-gram index answers 69%.

This is the most stable table in the README across corpus growth. Every
retention figure has moved by less than two points between eighteen and twenty
documents, and the gap the section exists to measure - roughly 27 points between
the word lists and the character list - has not moved at all. Where the clean-text
comparisons flip sign with every fourteen questions added, this one does not,
because it is measuring a structural property rather than a handful of decided
queries.

**And then the fusion undoes most of it.** Adding the third list lifts retention
from 66.1% to 73.2%, which reads like a win until you compare it against the
third list on its own: 92.6%, at 0.790 nDCG@5 against the fusion's 0.596. Under
noise the two brittle lists do not merely fail to help - they outvote the list
that is still working. This is the `ngram only` control that `evaluate.py`
already flags on clean text, failing here far more loudly, and it is the second
independent axis on which the three-way fusion loses to its own third list. The
script prints that comparison as a computed verdict rather than as prose, so it
cannot drift out of agreement with the table when the corpus grows.

Two things this does not establish. The seed spread (0.02-0.05 nDCG, now printed
as its own column) bounds the variance from *drawing the noise*, which is the
only uncertainty this script measures; the sampling error over 92 questions is a
separate and larger one, and
`evals/significance.py` is not yet wired into this harness - a real gap rather
than a rounding detail. And "robust" here means robust to typing errors only.
Vocabulary mismatch and paraphrase are a different failure mode with a different
lever, which is what the query-expansion table above measures, and nothing here
transfers to it.

The uncomfortable reading is the useful one - but only on the axis this script
measures. A later increment put error bars on the clean-text half of that
sentence and it did not hold: `ngram` against `hybrid3` on clean queries is
22W/12L, a mixed result whose +0.039 nDCG@5 gap does not survive a familywise
correction (see *The dead pointer*, above). The noisy half stands - 0.790
against 0.596 is far too wide to be sampling error. So the defensible claim is
the narrower one: **under query noise** the fusion is outvoted by its own third
list, and on clean text the two are indistinguishable on this gold set. The
"wins on both axes" reading was one axis of evidence and one axis of
over-reading.

## Persistence

> **Stale (20 documents, 92 chunks, 92 questions).** The load-versus-rebuild
> timings and the incremental-growth figures below were measured at twenty
> documents; the corpus is now 24 documents and 116 chunks. The mechanism and
> the fingerprinting behaviour are unchanged - only the numbers are old.

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

Measured on this corpus (20 documents, 92 chunks), loading is faster than a cold
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
document arriving at a process that already holds the other nineteen. Until now
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
multipliers have now held their ranges across four corpus sizes - 14, 16, 18 and
20 documents, landing at 1.49-1.51x and 0.61-0.64x each time - which is more than
a stability check this cheap had any right to establish. The warm multiplier
drifting 0.64x -> 0.61x is in the direction the paragraph below predicts is
*wrong*, and at this size that is measurement noise rather than a refutation:
the prediction is about the limit, not about two adjacent corpus sizes.

Two different answers, and the difference is what you start from.

From cold the incremental path **loses, by about half again**. Reaching twenty
documents by way of nineteen pays the TF-IDF fit twice - once at nineteen, once
at twenty - and that refit is the largest single stage in the build (3.67 ms of
it, against 2.01 ms for BM25 and 2.77 ms for chunking). If you do not already
have an index, build one; there is no version of this where assembling a corpus
document by document is the fast way.

From warm - a running process holding an index and handed one new document,
which is the only case the method exists for - it costs roughly **0.61x a full
rebuild**, against a corpus rebuild that is itself only ~10 ms. The saving is
the tokenising and counting of the nineteen documents already indexed, and it is
real but bounded, because the TF-IDF refit stays in the bill.

That framing is also the honest scaling statement, and it does not flatter the
feature: the saving is a *fixed* cost removed (one pass over the documents
already indexed, which is what add avoids re-doing) set against a cost that
*grows* with the collection (rebuilding every vector in it). The 0.64x is not a
floor that improves with scale - it is a number that gets worse as the corpus
grows, until the refit dominates and the warm row converges on 1.00x. Four
corpus sizes is still not a curve, and the multiplier has not visibly moved
across them, which is what the argument predicts at this scale: at 20 documents
the fixed saving is still large relative to the growing refit. The claim is about
which way the curve points, not about where the bend is.

The correctness claim is the part that has to be exact, and it is checked
against real from-scratch builds rather than against a re-derivation of the same
arithmetic. All 92 gold questions, ranked by both a grown engine and a whole
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
  not earning its complexity. Across four corpus sizes the two-list hybrid has
  led its components once, on a one-question margin, and the next fourteen
  questions took it back - which is a result the ablation columns produced for
  free and a fused-only table would have hidden indefinitely.
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
  the price it puts on the largest sweep in the README has gone 0.043 hit@1 at
  sixteen documents, 0.013 at eighteen, 0.032 at twenty - it oscillates, and the
  eighteen-document reading of that dip as "the effect outlasting the selection"
  did not survive. The `stable` column, which fell 65% -> 0% over the same span,
  was the honest signal all along. The two knobs where nothing is being selected
  report ~0.000 at every corpus size, which is how the number earns belief.
- **Audit the partition, not just the delta.** Error bars ask whether a
  difference between two slices is real; they assume the slices are. The
  `easy`/`hard` label underneath a dozen tables here was hand-assigned and
  unexamined for the entire life of the project, and `ragkb/difficulty.py`
  found it to be two things at once - a genuine lexical axis *and* a proxy for
  how many documents answer the question, perfectly confounded at p = 7.1e-15.
  The measured version earns its keep by predicting where a component's gain
  must appear: the n-gram list's advantage concentrates at low query-document
  containment and is exactly 0.000 where every query term is already in the
  answer, which is a mechanism rather than an aggregate. And the honest test of
  such a stratification is a rival one - question length, stratified the same
  way, produces a flat non-monotone profile, which is the reason to believe
  containment is measuring what it claims.
- **A score you cannot decompose is a score you cannot debug.** Both retrievers
  and the fusion are additive, so `ragkb/attribution.py` reconstructs any
  single score from its summands exactly and the tests assert that identity.
  Aggregates say a change helped; only the decomposition says which word did it,
  and whether the miss was fixable at all.

## Tests

```bash
pytest -q      # 575 tests
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
own floor rather than below it), the familywise correction (a hand-computed
Holm example so the step-down multipliers are pinned rather than assumed, that
a larger raw p never receives a smaller adjusted one, that shuffling the family
changes no element's verdict, that Holm never rejects less than Bonferroni -
the entire reason to prefer it - and that the routing table actually builds the
n-gram list, since a claim naming a method the harness cannot score is
unfalsifiable), the timing harness (asserted on its arithmetic
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
noise rate the harness reported), the corpus-versus-questions lever (the exact
binomial interval checked against its own closed form at both boundaries, the
textbook 3-of-10 value, its symmetry under counting failures instead of
successes, and the defining property that its bounds invert the exact binomial
test; the paired McNemar p-value against the hand-computed `2 ** (1 - d)` floor
and a hand-summed tail; and the partition that decides which questions predate
an increment - in particular that a question naming one old and one new document
is *not* carried, since scoring it against a corpus missing half its answer key
would charge the retriever for the setup), measured question difficulty (that
containment is 1.0 when every query term is in the answer and 0.0 when none
is, that it counts distinct terms so a repeated word cannot skew the ratio,
that stopwords are excluded from both sides, that a multi-document answer is
scored against the union, that the IDF weighting collapses onto the unweighted
form when every term shares a document frequency, that an out-of-vocabulary
query term counts *against* containment and outweighs a matched one - and the
case the module exists to expose, that `saturate` against `saturates` is
counted as missing, because a containment that silently credited morphological
variants would be measuring a stemmer nobody wrote and would hide exactly the
questions the n-gram list rescues; plus that the IDF is BM25's own, asserted
term for term against a fitted `BM25Index` rather than trusted to a comment),
the band arithmetic (half-open upwards at every edge, monotone, a genuine
partition that keeps empty bands so a band emptying out is visible, and an
open-ended top label when the banded quantity is a term count rather than a
ratio), Fisher's exact test (the textbook tea-tasting table, a hand-computed
2x0/0x2, invariance under all four table symmetries, 1.0 for a perfectly
balanced table, agreement with a directly summed hypergeometric, and the
perfect-separation shape the real audit finds), and the unpaired label-shuffle
test (1.0 for identical and for constant groups, a hand-counted 2/20 for three
against three perfectly separated, exactness and seed-independence below the
enumeration limit, symmetry in its arguments, that it accepts the unequal group
sizes a paired test cannot even be handed, and that the Monte-Carlo branch is
reproducible and never returns 0.0), and an end-to-end pass over the real
corpus and gold set - including the three properties the difficulty audit
turns on, pinned so they cannot change silently: that hand-`hard` questions
really do have lower measured containment, that no easy question is
multi-document, and that the label's confound with answer-key size is still
significant.

One of those tests exists because a bug shipped into a draft: `--ngram` was
first declared with argparse's `nargs="?"`, which quietly consumes the
*subcommand* as its value, so `--ngram search ...` died with "invalid int
value: 'search'" and the flag only worked when it happened to be last on the
line. `tests/test_cli.py` now asserts the flag survives having a subcommand
after it.
