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
| `evals/holdout.py` | nested cross-validation over four sweeps - three query-time knobs and, since this increment, index-time chunking - pricing what each sweep's top row cost by picking itself |
| `evals/difficulty.py` | audits the hand-assigned `easy`/`hard` label against measured containment, then stratifies the ablation along the measured axis with question length as a negative control |
| `evals/diversity.py` | audits the MMR sweep: the selection-depth mismatch between its accuracy and diversity columns, and the recall it never reported, on the multi-document questions that are the only place recall differs from hit |
| `evals/arbitration.py` | audits the ablation table: how often the fused ranking differs from the lists it fuses, and what the difference bought - which separates a fusion that had nothing to do from one whose wins and losses cancelled |

`corpus/` holds the sample knowledge base (twenty-eight documents on information
retrieval). `evals/gold.jsonl` holds one hundred and forty-nine labelled questions with
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
python evals/arbitration.py  # did the fusion reorder anything, and did the reordering pay?
python evals/power.py        # how many questions the differences would need to be real
python evals/corpus_lever.py  # did the new documents or the new questions move the rate?
python evals/difficulty.py   # is the hard slice a real slice? and where the n-gram gain lives
python evals/holdout.py      # what each sweep's top row cost by picking itself
python evals/holdout.py --no-chunking   # the same, query-time knobs only
python evals/tune.py         # index-time sweeps (chunk size, overlap, rrf_k)
python evals/tune.py --ngram 4   # the same rrf_k sweep against three lists
python evals/faithfulness_eval.py  # is the answer inside the passages it cites?
python evals/abstention_eval.py    # does it know when there is no answer?
python evals/failures.py           # why each miss missed, and what would fix it
python evals/latency.py            # what each optional stage costs per query
python evals/robustness.py         # how each method holds up when the query is misspelled
pytest -q
```

Current numbers on the bundled corpus and gold set (28 documents, 138 chunks,
149 questions, k=5, document-level relevance).

> **Provenance.** Recomputed at twenty-eight documents and 149 questions: the
> ablation tables, the third-retriever tables, the chunking sweep, `rrf_k` over
> two lists and three, the weighted-RRF weight sweep, the MMR lambda sweep, the
> arbitration and rank-agreement sections, the whole significance section,
> difficulty, holdout, power and the corpus lever.
>
> **Still stale, and marked at the point of use:** latency, robustness and
> persistence remain at the twenty-document, ninety-two-question figures, and
> the expansion and re-ranking sweeps remain at ninety-two.
>
> **This increment grew the corpus, and the gold set paid for itself in the one
> place it was supposed to.** Seventeen questions over two new documents took
> Holm survivors from 6 to 20 - the number of comparisons that clear familywise
> correction more than tripled, on a family whose size did not change (150
> comparisons both times). That is the lever the power section has been naming
> for five increments, and it is the first time it has moved this far.
>
> **Reported against it, because the ablation's headline moved and it is not
> defensible.** At 132 questions hybrid led or tied BM25 and TF-IDF on the all
> slice. At 149 it leads nothing: TF-IDF takes hit@5, recall@5, nDCG@5 and MRR,
> and BM25 takes hit@3. **That reversal is not significant** - `bm25 vs vector`
> on nDCG@5 is delta -0.006 at p_holm 1.000 with 15W/14L, and `hybrid2 vs bm25`
> is -0.003 at p_holm 1.000. The bold cells moved; the evidence did not.
>
> Replaying nDCG@5 on the all slice at every corpus size this repo has shipped
> gives `bm25, hybrid, vector, bm25, hybrid, vector` at 16, 18, 20, 22, 26 and
> 28 documents - **the leader changed at all five transitions**, and the two
> largest gaps in that sequence are 0.008 and 0.008. No significance test has
> ever supported any of those changes. That is a fact about the table rather
> than about the methods: on this gold set the three-way ablation on the all
> slice is not measuring an ordering, and the bolding should be read as
> decoration until the `ngram` comparisons, which do clear correction, are
> reached further down.
>
> **A stale constant, found by running the thing it configures.**
> `evals/corpus_lever.py` hard-codes which documents count as "new" when no
> `--added` flag is passed. That constant was set at twenty-two documents and
> never updated, so for three increments the script silently measured a
> two-increment-old document lever while printing a header that truthfully named
> the documents it had used. Nothing caught it because the output was internally
> consistent. Fixed here, and the comment now records the failure rather than
> just the value.
>
> **A pinned test lost its pin, exactly as its own docstring predicted.**
> `test_expansion_recovers_a_question_the_plain_query_misses` asserted that h15
> is recovered by pseudo-relevance feedback at the shipped setting, on the stated
> grounds that h15 "has held at every corpus size so far". At twenty-eight
> documents it does not. The claim is retracted in place and the test re-pinned
> to h30, chosen by replaying five corpus sizes rather than by preference.

The gold set is split by difficulty. The `easy` slice is eighty-one
questions whose wording shares vocabulary with the target document. The `hard`
slice is sixty-eight questions written to avoid the target document's own terms -
"if someone writes automobiles but the page only ever says cars" instead of
"vocabulary mismatch" - and forty-three of them are answered by more than one
document rather than one. Those counts are no longer a description of intent:
`evals/difficulty.py` confirms the split does separate on measured
query-document overlap (0.622 against 0.334 IDF-weighted, p < 5e-05), and also
that all forty-three multi-document questions sit in the `hard` slice and none
in `easy`, so the two axes coincide exactly and a hard-slice figure carries
both. [The audit is
here](#is-the-hard-slice-a-real-slice-auditing-the-partition-before-the-deltas).

### The ablation: hybrid against the two lists it fuses

This subsection did not previously have a heading of its own, which is part of
why an earlier drift went unnoticed: the tables read as a continuation of the
gold-set description above them rather than as a result with its own provenance.
They were marked stale at 106 questions for two revisions and are current again
here.

**all (149 questions, 43 with more than one relevant document)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.711 | 0.846 | 0.879 | 0.799 | 0.742 | 0.777 |
| bm25 | 0.705 | **0.866** | 0.886 | 0.802 | 0.745 | 0.778 |
| vector | 0.711 | 0.839 | **0.893** | **0.815** | **0.750** | **0.780** |

**easy (81 questions)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.877 | 0.938 | 0.963 | 0.963 | 0.922 | 0.909 |
| bm25 | 0.877 | 0.938 | **0.975** | **0.975** | **0.930** | **0.915** |
| vector | 0.864 | 0.938 | 0.963 | 0.963 | 0.917 | 0.902 |

**hard (68 questions, 43 with more than one relevant document)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.515 | 0.735 | 0.779 | 0.603 | 0.527 | 0.619 |
| bm25 | 0.500 | **0.779** | 0.779 | 0.596 | 0.524 | 0.615 |
| vector | **0.529** | 0.721 | **0.809** | **0.640** | **0.551** | **0.635** |

**Hybrid is bold in no column of any of the three tables, for the first time
since this section was written.** It is beaten or tied everywhere, by BM25 on
the easy slice and by TF-IDF on the other two.
[The error bars](#how-much-of-any-of-that-is-real-error-bars-on-the-ablation-table)
say none of it clears correction, and
[the arbitration audit](#the-tie-that-is-not-inertness-what-the-fusion-actually-reordered)
says the two-list fusion reorders 69% of questions against BM25 and 72% against
TF-IDF to arrive there, so this is a fusion acting constantly and cancelling,
not one standing still. The useful reading of these three tables is that the two-list
fusion is the wrong thing to be looking at; the third list is where the
significant results live.

The easy slice is still the less informative half. Every question on it has
exactly one relevant document, which makes recall@k numerically identical to
hit@k - the column is decoration. hit@5 sits at 0.963 or better for all three
methods, so there is almost nothing left there for a retrieval change to move.

The hard slice restores headroom: hit@1 falls from 0.877 to 0.515 and MRR from
0.909 to 0.619, and recall@k finally diverges from hit@k because a question
answered by several documents can be half-satisfied. Twenty-four questions land
in exactly that state at k=5, which is a failure mode the easy slice cannot
express at all.

The seventeen questions added this increment are harder than the set they
joined, and the absolute numbers fall accordingly: hybrid's all-slice nDCG@5
goes from 0.777 to 0.742 with no code change. A drop like that is what growing
a gold set towards the failure modes looks like, and it is the reason the
absolute values in this file are only ever compared within a single revision.

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
decimals on five of the six columns - hit@3 is the only one that separates
them, and it separates them by 0.007. The lead did not shrink; it inverted, and
it has now stayed inverted across three corpus sizes.

That row of identical numbers is not the same fact as "the fusion did nothing",
and the table cannot tell the two apart. [It reorders 73% of the gold
set](#the-tie-that-is-not-inertness-what-the-fusion-actually-reordered) and the
reorderings cancel.

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
one without. This increment adds `click-models.md` beside `learning-to-rank.md`,
whose closing paragraph raised the clicks-are-biased problem and stopped there,
and `deduplication.md` beside `chunking.md` and `ann-search.md` - overlapping
chunking manufactures near-duplicates, and locality-sensitive hashing is the
one family of approximate-search structures `ann-search.md` never covered.

The pattern that survived the growth unchanged for several increments has now
broken: BM25 no longer simply takes the easy slice at rank one, it *ties*
hybrid there (0.877 each), while taking hit@5, recall@5, nDCG@5 and MRR. On the
all slice hybrid ties BM25 and TF-IDF at rank one and leads nothing. The
summary "fusion buys ordering quality on paraphrased questions and pays a little
at rank one on literal ones" no longer describes the tables it was written for.

One documented behaviour has now moved three times, which is worth tracking
rather than quietly re-pinning. `tests/test_expansion.py` originally asserted
that pseudo-relevance feedback rescues h18 (`rag`) at the shipped weight. At
sixteen documents the feedback pool changed and it no longer did - recovering
h18 needed `original_weight=0.7`, which lost three other questions - so the test
was re-pinned to h15. At eighteen documents h18 was recovered again at the
shipped setting, landing at rank 4. At twenty, expansion does not change h18's
top-5 at all. **At twenty-eight, h15 stops being recovered too**, so the test is
re-pinned again, this time to h30 - chosen by replaying five corpus sizes rather
than by preference, with h32 as the standing alternate. The lever never broke;
its operating point moves with the feedback pool, and a test that pins one
question to it is pinning the corpus, not the code.

### The tie that is not inertness: what the fusion actually reordered

This section was written when the ablation table printed the two-list fusion
and plain TF-IDF as the same numbers on five of six columns. **At 149 questions
that premise has weakened: only hit@1 is now identical, and TF-IDF is ahead by
0.008 to 0.016 on the other four.** The question the section asks survives the
change - a near-tie is as silent as an exact one, and the gaps here are one to
two questions wide - but the phrase "the tie" in the heading is now shorthand
for a near-tie, and the argument below is about what a small aggregate delta
does and does not license, whether or not it is exactly zero.

There are two readings of a small delta, they are opposites, and the table is
silent between them:

* **Inertness.** RRF handed back the TF-IDF list. The inputs agreed, or one
  dominated, and there was nothing to arbitrate.
* **Cancellation.** RRF handed back a *different* list on most questions, and
  the questions it improved balanced the ones it damaged.

This is not a distinction without a difference. Each reading implies a
different repair. If the fusion is inert, it needs inputs that disagree - which
is precisely the argument [the third
retriever](#a-third-retriever-character-n-grams-and-what-it-cost-the-thesis)
makes, and this repo acted on that reading. If the fusion is arbitrating
constantly and netting to zero, more disagreement buys nothing on its own,
because the problem is not a shortage of opinions but the absence of any reason
to prefer one on a given query - a weighting problem, not a coverage one.

`agreement()` in `evals/evaluate.py` is the nearest existing measurement, and it
answers the other question: how much the fused lists overlap **with each
other**. That is a property of the inputs. Whether the *output* departs from
them was never measured. `evals/arbitration.py` measures it, on the
deduplicated document ranking that the metrics actually score.

**Reordering rate against each input (149 questions, k=5)**

| fusion | input | same order | same set, reordered | different set | acted on |
| --- | --- | --- | --- | --- | --- |
| hybrid2 | bm25 | 46 (0.309) | 51 (0.342) | 52 (0.349) | **103 (0.691)** |
| hybrid2 | vector | 42 (0.282) | 47 (0.315) | 60 (0.403) | **107 (0.718)** |
| hybrid3 | bm25 | 7 (0.047) | 36 (0.242) | 106 (0.711) | **142 (0.953)** |
| hybrid3 | vector | 10 (0.067) | 38 (0.255) | 101 (0.678) | **139 (0.933)** |
| hybrid3 | ngram | 5 (0.034) | 18 (0.121) | 126 (0.846) | **144 (0.966)** |

**The inertness reading is false.** The two-list fusion returns TF-IDF's exact
ranking on 42 of 149 questions and departs from it on the other 107. It was
never idle. The near-tie in the ablation table is a cancellation, and the
decomposition says so arithmetically: a question where the fused ranking equals
the input's contributes exactly zero to the aggregate delta, so

    mean delta over all  =  (acted / n)  x  mean delta over acted

and the script checks that identity numerically on every row rather than
asserting it.

**What the reordering bought (hybrid2 vs vector)**

| metric | hybrid2 | vector | delta | acted | on acted | W/L/T | 95% CI (acted) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hit@1 | 0.711 | 0.711 | +0.0000 | 107 | +0.0000 | 3/3/101 | [-0.047, +0.047] |
| hit@5 | 0.879 | 0.893 | -0.0134 | 107 | -0.0187 | 0/2/105 | [-0.047, +0.000] |
| recall@5 | 0.799 | 0.815 | -0.0168 | 107 | -0.0234 | 1/5/101 | [-0.051, +0.000] |
| nDCG@5 | 0.742 | 0.750 | -0.0082 | 107 | -0.0114 | 10/12/85 | [-0.033, +0.008] |
| MRR | 0.777 | 0.780 | -0.0037 | 107 | -0.0051 | 8/8/91 | [-0.030, +0.021] |

**At 132 questions this table was the cleaner result and it has degraded into a
messier one.** Every delta then was +0.0000 to four decimal places on four of
five metrics - a hundred-odd reorderings netting to nothing. At 149 the deltas
are small and consistently negative, and the W/L splits (10/12 on nDCG@5, 1/5
on recall@5) tilt against the fusion instead of sitting at a coin flip. Nothing
here clears correction and none of these intervals excludes zero, so the honest
statement is that cancellation has become *slightly adverse* cancellation, not
that the fusion has been shown to hurt. Against BM25 the split is 5W/10L on
nDCG@5 for a net of -0.0027 with [-0.017, +0.008], the same shape.

The `T` column is worth reading separately from a tie in the ablation table.
`10/12/85` means the fusion reordered 107 questions, 22 of them changed the
nDCG@5 score, and **85 reorderings were invisible to the metric** - it moved
documents around below the position the metric cared about. Set-only metrics
show this most starkly: hit@5 and recall@5 cannot see a same-set reordering at
all, which is why their `T` counts sit at 105 and 101.

**The third list is the case where reordering does pay.** Against BM25,
`hybrid3` reorders 142 of 149 and returns +0.042 nDCG@5 with a 23W/5L split and
a confidence interval of [+0.023, +0.068] that excludes zero; against TF-IDF,
+0.037 at 28W/5L, [+0.022, +0.059]. Same instrument, same gold set, and the
contrast with the two-list rows above is the clearest statement of what the
third list bought that the second one did not.

**And it is also the case where reordering costs.** Against the standalone
n-gram list, `hybrid3` reorders 144 of 149 and *loses*: recall@5 -0.046 at
2W/13L with [-0.082, -0.017], hit@5 -0.028 at 0W/4L with [-0.056, -0.007], both
excluding zero. The `ngram only` control in `evaluate.py` already reported that
the third list alone beats the fusion containing it. This says how: fusion is
not diluting by failing to act, it is actively reordering the n-gram list on 97%
of questions and the reordering is worse than leaving it alone.

One number sets a ceiling on all of this. A fused top-5 holding a document that
appeared in **no** input's top-5 is the only outcome no single list could have
produced, and for the two-list fusion it happens on **7 of 149 questions
(4.7%)**. Nearly everything the two-list fusion does is a permutation of
documents TF-IDF and BM25 had already surfaced. The three-list fusion reaches
28 of 149 (18.8%), four times as often - which is the same finding as the
reordering table, arrived at from the membership side rather than the order
side.

A caution the script prints and this section should repeat: a high reordering
rate is not a quality score. A fusion that reorders every question and nets to
zero is *worse* than an inert one, because it pays for the work and gets
nothing. The rate is diagnostic. What makes it worth measuring is that the
ablation table prints the same near-tie whether the rate is 0% or 72%, and
those are different systems.

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

**chunking, all 149 questions, k=5.** One question is worth 0.007 of any
column here, so a 0.030 gap is four questions and a 0.007 gap is one.

| target:overlap | chunks | mean tok | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60:15 | 369 | 52.3 | 0.698 | 0.852 | 0.906 | 0.819 | 0.750 | 0.779 |
| 80:20 | 242 | 66.5 | **0.725** | 0.839 | 0.879 | 0.796 | 0.749 | 0.789 |
| 100:25 | 171 | 83.6 | 0.705 | 0.852 | 0.879 | 0.809 | 0.750 | 0.777 |
| 120:0 | 113 | 99.9 | 0.671 | 0.826 | 0.879 | 0.799 | 0.728 | 0.755 |
| 120:15 | 135 | 99.8 | **0.725** | 0.852 | 0.893 | 0.806 | 0.751 | 0.789 |
| **120:30** (default) | 138 | 99.7 | 0.711 | 0.846 | 0.879 | 0.799 | 0.742 | 0.777 |
| 120:60 | 166 | 104.8 | **0.725** | **0.872** | 0.899 | 0.822 | 0.765 | 0.796 |
| 150:40 | 107 | 126.0 | 0.718 | **0.872** | **0.919** | **0.839** | **0.771** | **0.799** |
| 180:45 | 91 | 146.7 | 0.698 | 0.839 | 0.899 | 0.831 | 0.755 | 0.776 |
| 200:50 | 77 | 170.8 | 0.698 | 0.846 | 0.906 | 0.822 | 0.754 | 0.779 |

**Overlap earns its keep, and the claim has now held at six corpus sizes.**
`120:0` is the same chunk size with the overlap switched off, and it is the
worst row in the table at rank one, at hit@3, at nDCG@5 and at MRR: hit@1
falls 0.711 -> 0.671 and MRR 0.777 -> 0.755. That hit@1 loss has gone 0.031 at
sixteen documents - two questions, fairly called noise at the time - then 0.064,
0.043, 0.051, 0.038 and now 0.040. This remains the only claim in this table
that six corpus sizes have left standing. At twenty-eight documents `120:0` is
the outright worst row on four of the six columns - hit@1, hit@3, nDCG@5 and
MRR - which is as clearly as it has ever read. An idea straddling a chunk
boundary really is retrievable from both sides, and dropping overlap really does
cost the questions that depend on it.

**The depth claim beside it has now read four different ways in four
increments.** Two revisions ago the table was read as `120:0` buying back at
depth what it loses at rank one; at twenty-four documents that gain shrank to
+0.004 and the sentence was retracted; at twenty-six it went negative at -0.023
recall@5 and the row was called worse everywhere. At twenty-eight it is exactly
level with the default on both depth columns - 0.879 against 0.879 at hit@5 and
0.799 against 0.799 at recall@5 - so the row that was "worse everywhere" one
increment ago is now worse only at the top. The quantity has read +0.011,
+0.004, -0.023 and now 0.000 across four corpus sizes. **This sweep cannot pin
down overlap's effect at depth and should stop being asked to.** Its effect at
rank one is the part that has survived every measurement.

The extremes still fail asymmetrically rather than in the expected U shape.
`60:15` now *gains* at depth (hit@5 +0.027, recall@5 +0.020) while losing at
rank one (-0.013), which inverts the short-chunk story this table told at
twenty-six documents, where `60:15` cost hit@5 and recall@5. The long end still
shows no penalty: `150:40` takes four of six columns outright. The plateau claim
survives in the same weakened form - everything from 100 tokens up is within
about four questions of everything else.

**And the claim this section made most confidently two increments ago is gone
for good.** At twenty-four documents `120:60` took or tied five of six columns
and the prose said flatly that the older "winner scattered across four rows"
description "is no longer accurate and should not be recycled." At twenty-six it
took one column outright. At twenty-eight it takes none outright, ties hit@1
three ways and hit@3 with `150:40`, and has been overtaken by `150:40`, which
takes four of six columns and is the row the holdout section also picks on two
of three metrics.

That is the most useful thing this table has produced, so it should not be
buried in a row of decimals: **the row this sweep points at has changed at each
of the last three corpus sizes, and in both directions.** It is the fourth such reversal in this
file and the first one this README had explicitly predicted for itself, in the
paragraph immediately below, which is reproduced unchanged because it now
reads as a forecast rather than a caveat.

`120:60` was never adopted, for two reasons worth separating. The weak one is
bookkeeping: every other table in this README was measured at 120:30, so moving
the default would invalidate them all in one edit. The real one is that this is
exactly the pick the holdout section prices, and until this increment it could
not price it at all - the chunking sweep was not among the three knobs
`holdout.py` cross-validated, because each row is a full index rebuild, so
`120:60` had an in-sample margin and *no* out-of-sample estimate. A
four-question lead chosen by looking at the same questions it is scored on is
the pattern that had already reversed twice in this file. It has now reversed a
third time, on the row that prompted the warning.

**That estimate now exists, and it says this table is the most misleading one in
the README - though less emphatically than it did last increment.**
`evals/holdout.py` cross-validates the chunking sweep, and at twenty-six
documents the answer was unambiguous: `vs default` was negative on all three
metrics, so sweeping and taking the winner beat shipping the constant on none of
them. At twenty-eight it is negative on **one** of three - hit@1 at -0.017,
against nDCG@5 at **+0.022** and MRR at **+0.006**, with `150:40` picked in
sample and out on both of those. Optimism is still the largest of any sweep here
(+0.007 to +0.031), and the hit@1 and MRR winners still change on every one of
the twenty splits. [The table is
below](#the-other-bias-every-sweep-above-picked-its-own-winner). The honest
summary has moved from "this sweep loses to its own default everywhere" to "this
sweep loses to its own default at rank one, and beats it on the two graded
metrics, where its pick is also the more stable one" - which is materially
weaker than what was claimed here last increment and is corrected rather than
carried.

**`rrf_k`, all 149 questions, k=5.** The `changed` column counts questions whose
top-5 ordering differs from the default - the measurement that separates
"equally good" from "does nothing at all".

| rrf_k | hit@1 | hit@5 | nDCG@5 | MRR | changed |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.711 | 0.879 | 0.742 | 0.778 | 15 |
| 5 | 0.711 | 0.879 | 0.741 | 0.777 | 3 |
| 10 | 0.711 | 0.879 | 0.742 | 0.777 | 2 |
| 20 | 0.711 | 0.879 | 0.742 | 0.777 | 0 |
| **60** (default) | 0.711 | 0.879 | 0.742 | 0.777 | 0 |
| 200 | 0.711 | 0.879 | 0.742 | 0.777 | 0 |
| 1000 | 0.711 | 0.879 | 0.742 | 0.777 | 0 |

`rrf_k` is inert on this corpus, and the `changed` column is what proves it:
every value at or above 20 returns byte-identical top-5 orderings on all 149
questions, and the values below it reorder a handful. This is not "tuning did
not help", it is "there was nothing to tune", and those are different claims
that an accuracy table alone cannot tell apart. The threshold at which the knob
goes fully inert has now sat at 20, then 60, and is back at 20, which is worth
noting only because an earlier revision treated that boundary as a property of
RRF rather than of this corpus.

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

**easy (81 questions)**

| terms | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | **0.877** | 0.938 | 0.963 | 0.963 | **0.909** |
| 5, a=1.0 | **0.877** | 0.938 | 0.963 | 0.963 | **0.909** |
| 3, a=0.9 | 0.802 | 0.926 | 0.963 | 0.963 | 0.871 |
| 5, a=0.8 | 0.716 | 0.901 | 0.963 | 0.963 | 0.824 |
| 10, a=0.8 | 0.728 | 0.938 | 0.963 | 0.963 | 0.835 |
| 5, a=0.7 | 0.630 | 0.901 | 0.963 | 0.963 | 0.769 |

**hard (68 questions, 43 with more than one relevant document)**

| terms | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | **0.515** | 0.735 | 0.779 | 0.603 | **0.619** |
| 5, a=1.0 | **0.515** | 0.735 | 0.779 | 0.603 | **0.619** |
| 3, a=0.9 | 0.426 | **0.750** | 0.779 | 0.625 | 0.586 |
| 5, a=0.8 | 0.397 | 0.721 | 0.779 | 0.618 | 0.564 |
| 10, a=0.8 | 0.382 | **0.750** | **0.794** | **0.632** | 0.557 |
| 5, a=0.7 | 0.353 | 0.662 | **0.794** | 0.630 | 0.528 |

`alpha = 1.0` puts all the weight on the original query, so nothing is borrowed
and the ranking has to be identical to the unexpanded one. It is, on every one
of the 149 questions - that row and the test behind it are the correctness check
on this whole path, the same job `lambda = 1.0` does for MMR.

**These tables were stale at 92 questions for four increments and are current
here, and the refresh weakened the section's headline.** The claim was a split:
expansion buys reach on the hard slice while costing rank one. At 92 questions
the reach was real - hard-slice hit@5 rose 0.868 to 0.895 and recall@5 0.697 to
0.750 at `3 terms, alpha=0.9`. At 149 that same setting buys **nothing at
hit@5** (0.779, identical to off) and 0.022 of recall@5, a third of a question
per multi-document question. Only the deepest setting, `10 terms at alpha=0.8`,
still buys reach: hit@5 0.779 to 0.794 and recall@5 0.603 to 0.632, two
questions and change.

The cost side did not weaken. Hard-slice hit@1 falls 0.515 to 0.426 at `3, a=0.9`
and to 0.382 at `10, a=0.8` - six and nine questions - and MRR falls with it. On
the easy slice the borrowing costs six questions at rank one and buys nothing at
all at any depth, because those questions already shared vocabulary with their
target, so almost every borrowed term is noise.

So the honest summary has moved from "a real trade" to "a bad trade at every
setting except the deepest, and a marginal one there". Expansion stays off by
default, which it already was.

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
the equivalence cannot pass vacuously.

**The other pinned test has now lost its pin to the same instability, which is
the third time this paragraph has had to be rewritten and the first time it was
predicted.** `test_expansion_recovers_a_question_the_plain_query_misses` was
pinned to h15 on the stated grounds that h15 "has held at every corpus size so
far". At twenty-eight documents it does not: the expanded query returns
`inverted-index`, `score-normalisation`, `index-compression`,
`learning-to-rank`, `click-models`, and `rank-fusion` is absent. Replaying the
shipped setting across five corpus sizes gives h30 and h32 as the only cases
that survive all five, so the test is re-pinned to h30 and h32 is named as the
alternate. The docstring records the table rather than the conclusion, because
the conclusion is what expired last time.

That is the textbook pseudo-relevance feedback trade-off, and this corpus
reproduces it cleanly, including its instability: expansion buys reach on
paraphrased questions, pays for it in early precision on literal ones, and which
specific questions land on which side is a property of the corpus rather than of
the method. It is **off by default** and exposed as `--expand`. Two cautions on
reading the tables: the hard slice is 68 questions, so a 0.015 step is one
question; and the feedback documents come from the same retriever being
evaluated, so when the first pass is wrong, expansion confidently makes it
wronger.

### Re-ranking: a second stage that reads the pair

Fusion recalls far better than it precisely ranks. On the full gold set hit@5 is
0.879 but hit@1 is only 0.711: for a sixth of the questions the answer is
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
| off | 0.711 | 0.846 | 0.879 | 0.742 | 0.777 |
| 1.0 | 0.711 | 0.846 | 0.879 | 0.742 | 0.777 |
| 0.8 | 0.732 | 0.866 | 0.899 | 0.760 | 0.795 |
| 0.6 | 0.718 | **0.879** | 0.899 | 0.757 | 0.791 |
| **0.5** | 0.738 | 0.872 | 0.906 | 0.769 | **0.806** |
| 0.4 | **0.738** | 0.872 | 0.906 | 0.769 | **0.806** |
| 0.2 | 0.705 | 0.872 | 0.913 | 0.768 | 0.791 |
| 0.0 | 0.725 | 0.866 | **0.919** | **0.777** | 0.801 |

And on the hard slice, where the first stage has the most room:

| alpha | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| off | 0.515 | 0.735 | 0.779 | 0.527 | 0.619 |
| 0.8 | 0.529 | 0.779 | 0.809 | 0.547 | 0.638 |
| 0.6 | 0.515 | **0.794** | 0.809 | 0.547 | 0.638 |
| **0.5** | 0.529 | 0.779 | 0.824 | 0.559 | **0.652** |
| 0.2 | 0.485 | 0.779 | 0.853 | 0.574 | 0.636 |
| 0.0 | **0.544** | 0.750 | **0.868** | **0.596** | 0.664 |

**These tables were stale at 92 questions for four increments and are current
here. The refresh cost this section its headline.** It used to read "the largest
single-increment gain the project has measured", on an all-slice hit@1 of
0.685 -> 0.750 at the shipped alpha=0.5. At 149 questions that gain is
0.711 -> 0.738: **0.027 rather than 0.065, four questions rather than ten.** On
the hard slice the shipped setting now buys 0.014 at hit@1, one question, where
it used to buy 0.079. The claim that this lever's effect was "growing with the
corpus rather than drifting" is retracted; across four measured sizes the
all-slice hit@1 gain has read 0.026, 0.065, and now 0.027, which is drift.

What survives is the mechanism and the direction. Every alpha below 1.0 still
beats `off` on nDCG@5 and hit@5 on both slices, and the deepest setting
(alpha=0.0, pure cross-encoder) is now the best row on hit@5 and nDCG@5 on both
slices - it converts recall the first stage already had into better ordering.
What it no longer reliably converts is rank one.

The shipped alpha=0.5 ties alpha=0.4 for the best all-slice hit@1 and takes MRR
on both slices, so for the first time in three revisions the default is also the
sweep's pick at rank one. That is recorded as a coincidence rather than a
vindication, given how far the rest of this section moved.

Read the trade-off honestly, and note that it has *unsharpened* further. At
eighteen documents leaning harder on the second stage (alpha <= 0.2) cost hit@1
and hit@5 both, and this section said so. At twenty, alpha=0.2 still cost rank
one but took hit@3 and nDCG@5 on both slices. At twenty-eight, alpha=0.0 takes
hard-slice hit@1 *as well* - 0.544 against the default's 0.529 and `off`'s 0.515
- so the "pure cross-encoder pays at rank one" half of the trade-off has
inverted on the slice where it was most confidently asserted. It is one
question, and it is the direction that matters rather than the size. What
remains true across every
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
reason the `rrf_k` sweep found nothing to tune: the two lists agree on 0.830 of
their top-5 by chunk, and a fusion constant has no room to arbitrate lists that
already concur. (That sentence is measured in set overlap, which cannot see the
one thing RRF reads. It is re-measured in rank agreement below, where the
direction survives and the magnitude roughly halves - see
[The unit that claim was measured in cannot see order](#the-unit-that-claim-was-measured-in-cannot-see-order).)

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
0.830 to 0.651, which is the mechanism working: the third list is contributing
a genuinely different opinion rather than a paraphrase of an existing one.
`hybrid2` is the shipped two-list fusion, `ngram` is the n-gram list alone and
`hybrid3` is all three fused.

**all 149 questions, k=5**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid2 | 0.711 | 0.846 | 0.879 | 0.799 | 0.742 | 0.777 |
| **ngram** | **0.772** | **0.913** | **0.946** | **0.891** | **0.822** | **0.841** |
| hybrid3 | 0.752 | 0.879 | 0.919 | 0.846 | 0.787 | 0.818 |

**hard slice (68 questions, 43 with more than one relevant document)**

| method | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid2 | 0.515 | 0.735 | 0.779 | 0.603 | 0.527 | 0.619 |
| **ngram** | **0.618** | **0.824** | **0.882** | **0.762** | **0.662** | **0.722** |
| hybrid3 | 0.574 | 0.779 | 0.838 | 0.676 | 0.596 | 0.680 |

**The result is the largest this project has measured, and it is not the result
the experiment was designed to find.** The n-gram list was added as a third
input to the fusion. It beats the fusion - alone, on every column of both
slices. Against `hybrid2` on the hard questions it takes hit@1 from 0.515 to
0.618 and recall@5 from 0.603 to 0.762, and its hit@5 of 0.882 against 0.779 is
seven of sixty-eight questions where the answer was simply not in the top five
before.

**This is also the only part of the ablation that clears familywise
correction.** Of the 20 comparisons surviving Holm at twenty-eight documents,
**every one involves either the standalone `ngram` list or `hybrid3`, and not
one is between `bm25`, `vector` and `hybrid2`** - checked against the shipped
significance output rather than eyeballed. The same held at twenty-six
documents, where all 6 survivors involved `hybrid3`. The three-way table at the
top of this file contributes nothing to either set.

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

**The one cell that contradicted that has gone back to a tie, and it lasted a
single increment.** The easy slice is not printed above. At twenty-four
documents its hit@1 read 0.864 / 0.879 / 0.833 and the fusion sat between its
inputs as everywhere else; at twenty-six it was the one place where `hybrid3`
beat *both*, 0.890 against 0.877 for `ngram` alone and 0.863 for `hybrid2`, and
this paragraph recorded that "fusion never beats its best input" had become
false in one cell. At twenty-eight the same cell reads 0.901 / 0.901 / 0.877:
`hybrid3` and `ngram` are level and the exception is gone. It was one question
wide when it was claimed, which is why it was flagged as too small to carry
weight, and one question is exactly what it cost to disappear. Every cell in all
three slices now has `hybrid3` at or below its best input.

Two cautions before anyone reads this as "character n-grams beat words". The
first is arithmetic: 149 questions, so the 0.060 gap in hit@1 is nine questions,
and the hard slice is 68, where one question is 0.015. The second is structural,
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
line is broken. At twenty-eight documents it stands at **+0.103**, the widest it
has been since fourteen. Replayed at every corpus size this repo has shipped,
the margin reads:

| documents | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard-slice hit@1 margin | +0.136 | +0.077 | +0.031 | +0.079 | +0.065 | +0.077 | +0.102 | +0.103 |

It fell for three sizes, then rose for four, and has not been below +0.031 at
any point. The monotone narrowing the eighteen-document revision extrapolated
from was three points of a sequence that has since gone the other way, which is
the second time in this file a three-point trend has been read as a direction
and then reversed. Every value from sixteen documents onwards was recomputed
from the archived commit rather than copied forward.

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

The cost is real and small in absolute terms: 5,857 gram features against 2,396
word features (2.4x), and fitting the third index takes about 22 ms against
about 4 ms for TF-IDF on this corpus. Denser vectors are also why the list is
noisier per-match - most English 4-grams are common, so everything matches
everything a little - which is what makes the standalone result surprising
rather than obvious.

### Weighted RRF: acting on "the third list is the best list"

The n-gram section ends on an uncomfortable number. The character n-gram list
*alone* scores nDCG@5 0.822, while the three-list fusion that contains it
scores 0.787. Textbook RRF weights every list identically, so the fusion is
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

Sweeping the weight on the n-gram list, over all 149 gold questions at k=5:

| weights | hit@1 | ndcg@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- |
| uniform (plain RRF) | 0.752 | 0.787 | 0.846 | 0.818 |
| `ngram=2` | 0.772 | 0.797 | 0.851 | 0.830 |
| `ngram=3` | 0.785 | 0.802 | 0.851 | 0.837 |
| **`ngram=4`** | **0.799** | 0.806 | 0.851 | 0.843 |
| `ngram=6` | 0.792 | 0.812 | 0.865 | **0.844** |
| ngram only (control) | 0.772 | **0.822** | **0.891** | 0.841 |

The gain concentrates in the hard slice, which is where fusion was supposed to
earn its keep: nDCG@5 0.596 → 0.629 and hit@1 0.574 → 0.662 at `ngram=4`.

Two honest readings of that table, both of which matter more than the win:

**The control holds the ordering and reach columns, and has now done so at six
corpus sizes out of seven.** At 16 documents `ngram only` beat every weighted
fusion on both nDCG@5 and recall@5. At 18, `ngram=4` took nDCG@5 by 0.003 - a
lead this section correctly refused to call a win, noting the interval crossed
zero. At 20, 24 and 26 the control led again on nDCG@5, and at 28 it still does
(0.822 against 0.806) while holding recall@5 outright and by a wide margin
(0.891 against 0.851). Weighting buys rank one and MRR; it has never been shown
to buy either ordering or reach against the list it is weighting up. The split
is consistent across six corpus sizes - fusion reorders what it has better than
its best input at rank one, and retrieves less - and the 0.003 nDCG@5 lead at
eighteen documents is now visibly the noise it was flagged as.

**Both margins widened this increment, which is worth stating because the
previous revision had explicitly checked for the opposite.** That revision
quoted the control's nDCG@5 lead over `ngram=4` going 0.006 to 0.005 and its
recall@5 lead going 0.021 to 0.027, and read the pair as "neither catching nor
falling further behind". At twenty-eight documents those leads are 0.016 and
0.040 - both roughly doubled, and both now several questions wide rather than
under one. The reading changes accordingly: the control is not merely holding,
it is pulling away, and the seventeen new questions are where it did so.

Note also that `ngram=4` is no longer the best weighted row on MRR - `ngram=6`
takes it, and takes nDCG@5 and recall@5 among the weighted rows too. The bolded
`ngram=4` is kept because it is the setting every other table here quotes, not
because the sweep still points at it.

**Twenty comparisons in this repo now survive multiple-comparison correction,
and none of them is this one.** See the significance section below. Every
survivor involves the standalone n-gram list or `hybrid3`, and the pair this
section is about - weighted fusion against the control - is not among them.
"Adding the third list to the fusion helps" is SUPPORTED; "weighting it up
helps" and "the fusion beats the third list alone" are both still NOT SHOWN. So
the shipped default stays uniform. A non-significant delta measured against a
control that wins two of four columns is not grounds for changing what the
engine does by default; it is grounds for a knob and a table.

### `rrf_k` was not inert. It had nothing to arbitrate.

The tuning section above closed with a claim it could not test: that `rrf_k`
"would start to matter with more retrievers in the fusion, or with retrievers
that disagree more than BM25 and TF-IDF do here - and the honest version of
that claim is that this corpus cannot demonstrate it. **It can now**: adding a
third, structurally different list moves `rrf_k = 1` from 15 changed orderings
to 88. The third list is the
experiment that demonstrates it. Same corpus, same gold set, same sweep, one
more ranked list:

| rrf_k | hit@1 | hit@3 | hit@5 | nDCG@5 | MRR | changed (2 lists) | changed (3 lists) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.738 | **0.886** | **0.933** | 0.786 | 0.813 | 15 | **88** |
| 5 | **0.752** | 0.879 | 0.919 | 0.787 | 0.816 | 3 | 62 |
| 10 | **0.752** | 0.879 | 0.919 | 0.787 | 0.816 | 2 | 44 |
| 20 | **0.752** | 0.879 | 0.919 | **0.790** | **0.818** | 0 | 20 |
| **60** (default) | **0.752** | 0.879 | 0.919 | 0.787 | **0.818** | 0 | 0 |
| 200 | **0.752** | 0.879 | 0.913 | 0.787 | 0.817 | 0 | 7 |
| 1000 | **0.752** | 0.879 | 0.913 | 0.787 | 0.817 | 0 | 7 |

With two lists, every value at or above 20 returns byte-identical orderings on
all 149 questions and the accuracy columns move by at most 0.001. With three,
`rrf_k = 1` reorders **88 of 149** - again about three fifths of the gold set -
and the sweep develops a shape. The contrast is the whole point of running both,
and it has widened again with the corpus: 15 against 88 is the sharpest
separation this comparison has shown.

The shape itself keeps flipping between corpus sizes, which is the second
lesson. At sixteen documents small `rrf_k` beat the default on reach; at
eighteen the default and everything above it took hit@5, nDCG@5 and MRR; at
twenty `rrf_k = 1` took hit@3 and hit@5 while giving up rank one; at twenty-four
it kept hit@5 alone and gave up both hit@1 and hit@3; at twenty-six it swung
back to taking hit@3 and hit@5 while giving up rank one and MRR. At twenty-eight
it holds hit@3 and hit@5 again, still sells rank one (-0.013) and MRR (-0.005) -
and the nDCG@5 and MRR columns are now topped by `rrf_k = 20`, a value that has
never led anything here before and does so by 0.003, which is half a question.
Six corpus sizes, six different shapes, every reading one to four questions
wide. The stable part across all six is the direction of the trade at the
extreme end - small `rrf_k` buys depth and sells rank one - and nothing finer
than that has held. A sweep whose winner is different every time it is run is
reporting the gold set, not the knob.

The point is not the gain, it is that the earlier "there was nothing to tune"
was a statement about the *inputs* rather than about the knob, and it took a
genuinely different list to tell those two explanations apart. Note that the
largest values still change orderings rather than none - 7 here, against 9 at
twenty-six documents - because on a few questions the ranks are far enough
apart for the Borda limit to differ from the default.

### The unit that claim was measured in cannot see order

Both arguments above rest on one number, and it is the wrong kind of number.

"The two lists agree on 0.830 of their top-5 by chunk, and a fusion constant has
no room to arbitrate lists that already concur" is a statement about **sets**.
RRF is a function of **rank**: `1 / (rrf_k + rank)` reads position and nothing
else. Two lists holding the same five chunks in opposite order have a set
overlap of 1.000 and produce different fused scores, different hit@1, different
MRR and different nDCG@5. Every accuracy column in this README except recall@5
is order-sensitive. The one statistic used to explain them is not.

`ragkb/rankagree.py` adds the order-aware measurements and keeps the set one
beside them, so the claim can be restated rather than assumed:

```bash
python evals/rank_agreement.py
```

**all 149 questions, k=5, RBO p=0.8**

| pair | set overlap | RBO | tau_b | tau n | conjoint | same top-1 | identical |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 vs vector | 0.830 | 0.578 | 0.749 | 149 | 4.15 | 0.893 | **0.094** |
| bm25 vs ngram | 0.542 | 0.412 | 0.557 | 129 | 2.71 | 0.698 | 0.000 |
| vector vs ngram | 0.580 | 0.434 | 0.571 | 135 | 2.90 | 0.745 | 0.007 |

**BM25 and TF-IDF return the same five chunks in the same order on 14 of 149
questions.** The number the concurrence argument was built on is 0.830. The
number a rank-based fusion actually experiences is 0.094. Both are correct
measurements of the same pair of lists, and a reader given only the first would
form a badly wrong picture of what RRF is being handed.

Three measures because they fail differently. Kendall tau-b is the textbook
answer and the worst choice here: it is defined only over items *both* lists
retrieved, and that conjoint set averages 4.15 items for BM25-against-TF-IDF but
2.71 for the cross-representation pairs - where two shared items can only score
+1 or -1. Tau is least trustworthy exactly where the interesting comparison is,
which is why `conjoint` and `tau n` sit beside it in the table rather than in a
footnote. Rank-biased overlap (Webber, Moffat & Zobel, TOIS 2010) is built for
truncated non-conjoint lists: it sums overlap at every depth under a geometric
weight, so an item at rank 1 in one list and rank 5 in the other is partially
credited instead of discarded, and disagreement near the top costs more than
disagreement at the bottom. `p = 0.8` is not a free parameter - RBO's expected
viewing depth is `1/(1-p)`, so 0.8 is the value that matches this project's
`k = 5`. It leaves a residual of `0.8**5 = 0.328`, reported rather than
extrapolated away, because comparing five-item prefixes genuinely does leave
that much unknown.

**The comparative claim survives; its magnitude does not.** BM25 and TF-IDF do
concur more than either concurs with the n-gram list, in all three units - so
the reason given for adding a third representation was sound. But the size of
the gap depends heavily on which unit you read it in. In set overlap the
within-representation pair leads the cross-representation pairs by 0.269. In RBO
that lead is 0.155, and in tau_b 0.185. Set overlap roughly doubles the apparent
separation, and it is the one every earlier revision quoted.

**And the mechanism holds - tested where it could have failed.** The claim is
not only that the lists concur, it is that concurrence is *why* `rrf_k` has
nothing to arbitrate. That is falsifiable: the questions where `rrf_k` does move
the fused ordering should be the questions whose input lists agreed least. If
the two groups had the same rank agreement, the concurrence story would be a
coincidence and the real cause something else.

| lists | changed by rrf_k | RBO on changed | RBO on unchanged | delta | p |
| --- | --- | --- | --- | --- | --- |
| two | 15/149 | 0.476 | 0.589 | **+0.113** | 0.0000 |
| three | 88/149 | 0.437 | 0.529 | **+0.091** | 0.0000 |

The prediction holds in both configurations and in the right direction, by an
unpaired two-sided permutation test over disjoint groups of very unequal size
(15 against 134). `rrf_k` moves precisely the questions where the lists disagree
about order, which is the mechanism stated properly - and the two-list delta
remaining larger than the three-list one is consistent with it, since a
genuinely different list creates order disagreement on questions that used to
have none, which dilutes the contrast between the moved and unmoved groups.
Both deltas are within 0.02 of their values at twenty-six documents, which makes
this the most stable measurement in this README across the increment.

The honest summary: the conclusion in the two sections above was right, the
evidence offered for it was in the wrong unit, and restating it in the right one
both confirms the direction and cuts the effect size roughly in half. That is a
better outcome than the alternative, and it is only visible because the claim
was specific enough to re-measure.

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

Sweeping lambda over all 149 questions at k=5 (`redundancy` is the mean
pairwise cosine within a result set; `distinct` is how many documents it spans):

| lambda | hit@1 | hit@5 | recall@5 | MRR | redundancy | distinct |
| --- | --- | --- | --- | --- | --- | --- |
| off | 0.711 | 0.879 | 0.799 | 0.777 | 0.122 | 3.83 |
| 1.0 | 0.711 | 0.879 | 0.799 | 0.777 | 0.122 | 3.83 |
| 0.9 | 0.711 | 0.886 | **0.805** | 0.778 | 0.116 | 3.93 |
| **0.7** | 0.711 | **0.893** | 0.802 | **0.779** | 0.099 | 4.36 |
| 0.5 | 0.711 | 0.886 | 0.802 | 0.775 | 0.091 | 4.57 |
| 0.3 | 0.711 | 0.866 | 0.768 | 0.771 | 0.082 | 4.74 |

`lambda = 1.0` reproduces the unmodified ranking exactly, which is the cheapest
available correctness check on the implementation. Hit@1 has now been flat
across the entire sweep at six consecutive corpus sizes, which is the durable
part: re-ordering for diversity does not disturb what sits at rank one, because
MMR's first pick is always the most relevant candidate by construction.

**The peak at 0.7 held, and grew from one question to two.** At 16, 18, 20 and
24 documents hit@5 and hit@1 did not move at all from 1.0 down to 0.5, and this
section described the 0.9-to-0.5 band as "nearly free". At twenty-six a peak
resolved at `lambda = 0.7`, worth one question at hit@5. At twenty-eight the
same row is still the best on hit@5 (0.893 against 0.879, two questions) and
MRR, while cutting redundancy 19% and covering 14% more documents. Two
consecutive corpus sizes agreeing on the same optimum is the first time this
sweep has repeated itself.

That reading should still be held loosely. Two questions on 149 is 0.014, the
MRR gain is 0.002, and nothing in this table has been through a significance
test. The defensible statement is that diversity re-ranking at 0.7 is free on
this gold set and possibly slightly positive, not that it helps.

**The `recall@5` column withdrew the "possibly positive" half of that sentence
last increment, and this increment partly gives it back.** The column is a
*recall* argument - five paraphrases of one passage cannot hold the second
document a two-document question needs - and recall@5 is the only metric here
that can see it. At twenty-six documents it was flat at 0.850 from `off` through
`lambda = 0.7`, so the hit@5 gain bought no recall at all. At twenty-eight it is
no longer flat: `0.9` takes the column at 0.805 and `0.7` and `0.5` both sit at
0.802, all three above `off` at 0.799. That is under half a question of movement
and the best recall row is not the best hit@5 row, so the honest reading is that
recall has stopped contradicting the peak rather than that it now supports it.
The section
[below](#the-column-the-diversity-argument-needed-and-the-two-settings-the-table-was-mixing)
measures the multi-document slice where the column can actually move.

That the optimum has moved at all across corpus sizes - 0.5 at fourteen
documents, then four revisions of a flat band, now a peak at 0.7 - is the
argument for shipping the sweep rather than a tuned constant, so re-ranking
stays **off by default** and the trade-off is a flag rather than a hidden number.

One caveat that the holdout section adds and this table cannot see: `off` wins
here on the full gold set, but it is *not* what a cross-validated pick chooses on
most splits, and that gap now costs 0.004 of nDCG@5. The sweep below prices it.

### The column the diversity argument needed, and the two settings the table was mixing

The table above scores a technique whose entire justification is about recall,
using metrics that are structurally unable to see recall, at a selection depth
where the technique barely runs. `evals/diversity.py` takes those three problems
in order.

```bash
python evals/diversity.py
python evals/diversity.py --lambdas 0.9,0.7,0.5 --slice all
```

**Averaging recall over the whole gold set mostly re-reports hit@5.** On a
question with exactly one relevant document, recall@5 is 1.0 when that document
is in the top five and 0.0 otherwise, which is the definition of hit@5 - the two
columns are the *same number* on those questions. 106 of the 149 gold questions
have one relevant document; the other 43 have two or three, and every one of
them is in the hard slice. Those 43 are the entire independent content of the
recall column, and the script checks the identity on all 106 rather than
asserting it (`0 mismatches`) before restricting to them.

**The accuracy columns and the diversity columns were never the same
configuration.** MMR is greedy against a running redundancy penalty, so what it
returns depends on how many items it is asked to select. `diversity_report`
takes `hit@1`, `hit@5` and `MRR` from `ranked_doc_ids`, which searches at
`k * 4 = 20` because it needs enough chunks left after deduplicating to
documents; it takes `redundancy` and `distinct` from `search` at `k = 5`.
Selecting 5 of 25 candidates is an aggressive diversity selection. Selecting 20
of 25 can only drop five and is close to inert. Every row of the table above
pairs one with the other:

```
selection depth audit (43 questions): where the diversification goes
lambda      shallow_docs  shallow_redund       deep_docs       kept_docs           churn
----------------------------------------------------------------------------------------
off                3.977           0.117          13.163           5.000           0.000
1.0                3.977           0.117          13.163           5.000           0.000
0.9                4.093           0.109          13.349           5.000           0.047
0.7                4.512           0.093          13.721           5.000           0.302
0.5                4.698           0.088          14.000           5.000           0.465
0.3                4.721           0.082          14.349           5.000           0.814
```

`kept_docs` is pinned at exactly 5.000 for every lambda, because truncating to
the first five *distinct* documents always fills when the pool holds that many.
No document-level metric in this README can see a change in how many documents a
result set spans - that quantity is a constant by construction. `churn` is the
guard against over-reading that: it counts how many of the five differ from the
`off` list, and at `lambda = 0.9` it is **0.047**, which is two questions out of
forty-three. At that setting the printed `distinct` column moves from 3.83 to
3.93 while the list every accuracy column is computed from is unchanged on
forty-one of the forty-three questions. (At twenty-six documents this figure was
exactly 0.000 and the point could be made absolutely; it is now very nearly
zero, which makes the same point slightly less cleanly and is reported as it
measured.)

**On the 43 questions where recall can move, it still does not move up.**

```
recall@5 on the multi slice (43 questions), paired against lambda off
lambda      recall    docs found      95% interval     W/L/T         p
----------------------------------------------------------------------
off          0.605         53/88       0.492-0.705    0/0/43     1.000
1.0          0.605         53/88       0.492-0.705    0/0/43     1.000
0.9          0.605         53/88       0.492-0.705    0/0/43     1.000
0.7          0.593         52/88       0.481-0.695    1/2/40     1.000
0.5          0.593         52/88       0.481-0.695    2/3/38     1.000
0.3          0.547         48/88       0.436-0.652    1/6/36     0.125
```

`lambda = 1.0` reproducing `off` exactly is the correctness check, the same one
the sweep above uses. Below it every row is now a net loss rather than the wash
the previous revision could report at `0.7`. `lambda = 0.3` moves seven
questions and loses six; p = 0.125 does not clear 0.05, and the direction is the
same one it has pointed at every corpus size.

**The `lambda = 0.7` row is the whole argument for printing recall, and it is
worth naming the three questions.** Its W/L/T is 1/2, and all three movers are
two-document hard questions that move by exactly 0.5:

| question | relevant | `off` | `0.7` | what changed |
| --- | --- | --- | --- | --- |
| h51 | relevance-feedback, query-performance-prediction | 0.000 | 0.500 | `relevance-feedback` displaced `sparse-neural-retrieval` at rank 5 |
| h47 | score-normalisation, rank-fusion | 1.000 | 0.500 | `rank-fusion` dropped out of the top five, `bm25` entered at rank 5 |
| h58 | learning-to-rank, reranking | 1.000 | 0.500 | `reranking` dropped out, `query-performance-prediction` entered at rank 5 |

h51 is exactly the win the diversity argument predicts: the ranking was five
documents deep into one neighbourhood, and spreading it out reached a second
relevant document. h47 and h58 are exactly the loss that argument does not
mention: spreading out pushed a relevant document off the end. **At twenty-six
documents this was 1W/1L and the section called it a wash. At twenty-eight it is
1W/2L.** One new question changed a wash into a small net loss, which is a fair
description of how much this table can bear.

**hit@5 counted the win and was incapable of counting the losses.** h51 went
from zero relevant documents to one, so its hit@5 went 0 to 1. h47 and h58 each
went from two relevant documents to one - and their hit@5 stayed at 1, because
hit@k stops looking after the first. Those questions are the entire `0.893`
against `0.879` that makes `lambda = 0.7` the best row in the table above. The
row is not better; it is measured with an instrument that sees diversity's
benefit and is blind to its cost by construction. At twenty-six documents this
was one win against one invisible loss; at twenty-eight it is one win against
two.

So the narrower and defensible claim, replacing "free and possibly very slightly
positive": **on this gold set MMR does not improve document-level retrieval at
any lambda, it is at best free at 0.7 and costly below, and the positive reading
the sweep table invites is an artefact of the metric.** It keeps the case it
always had - redundancy down 19%, the top-5 chunks spanning 14% more documents
at 0.7, both of which
matter to an answerer reading the passages - and that case is now separated from
a retrieval claim it never supported. Re-ranking stays **off by default**, and
the reason is now measured rather than inherited.

Two limits worth stating. `churn` counts document identity, so a re-ranking that
reorders the same five documents registers as 0.000 and MRR is the column that
would catch it; and 37 questions with 75 relevant documents between them is a
small slice - the Clopper-Pearson interval on `off` alone spans 0.535 to 0.760,
which is most of the range the rate could take. This section establishes that
the earlier positive reading does not hold, not that a better-targeted diversity
method could not.

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

Forty-eight of the one hundred and ninety-four (question, relevant document)
pairs are missed at k=5 - 24.7%, against 19.5% at twenty-six documents. A slice
of that table, and the whole of it in `python evals/failures.py`:

| id | slice | document | hybrid | bm25 | vector | coverage | category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q24 | easy | query-processing | 6 | **5** | 6 | 0.60 | **fusion-loss** |
| h08 | hard | reranking | 6 | 9 | **4** | 0.39 | **fusion-loss** |
| h33 | hard | inverted-index | 9 | 6 | 12 | 0.44 | near-miss |
| h45 | hard | bm25 | 6 | **5** | **5** | 0.56 | **fusion-loss** |
| h48 | hard | score-normalisation | 6 | 8 | **4** | 0.52 | **fusion-loss** |
| h51 | hard | relevance-feedback | 6 | 7 | **5** | 0.67 | **fusion-loss** |
| h60 | hard | learning-to-rank | 6 | 8 | **5** | 0.50 | **fusion-loss** |
| h61 | hard | click-models | 7 | 7 | 7 | 0.48 | near-miss |
| h62 | hard | click-models | 7 | 7 | 7 | 0.52 | near-miss |
| h64 | hard | click-models | 10 | 9 | 10 | 0.50 | near-miss |
| h66 | hard | deduplication | 11 | 11 | 11 | 0.39 | deep-miss |
| h66 | hard | ann-search | 21 | 20 | 24 | 0.33 | deep-miss |
| h67 | hard | evaluation | 7 | 7 | 7 | 0.65 | near-miss |

**Fusion losses went from two to six, and five of the six are TF-IDF being
outvoted rather than BM25.** In h08, h48, h51 and h60 the vector list had the
document at rank 4 or 5 - inside k - and the fusion pushed it to 6, because BM25
had it at 7 to 9 and the unweighted vote split the difference.

h45 is the odd one and worth spelling out, because it looks impossible: *both*
components rank `bm25` fifth and the fusion returns it sixth, displaced by
`query-performance-prediction`, which both components rank *sixth*. Fusing two
lists that agree should not invert them. The explanation is that the `bm25`,
`vector` and `hybrid` columns in this table are **document** ranks, and they are
produced by deduplicating each list down to its best chunk per document - while
RRF is applied to the *chunk* rankings underneath, before any of that
deduplication happens. A document with several mid-ranked chunks can accumulate
a better fused score than one with a single slightly-higher chunk. So the
columns either side of the fused rank are not the input RRF was given, and a
row like h45 is the case where that distinction becomes visible rather than
academic. It is a limitation of this table's presentation, not of the fusion.

That is a real cost of fusion, it is invisible in any averaged table, and it now
has six instances rather than two. It also cuts against the ablation table at
the top of this file in a consistent direction: TF-IDF leads the all slice on
four of six columns there, and here it is the list the fusion most often
overrules when it is right.

**h33, which the previous revision called "the cleanest example the project has
produced", is no longer a fusion loss.** BM25 ranked `inverted-index` at 5 and
the fusion pushed it to 8; at twenty-eight documents BM25 ranks it 6 and the
fusion 9, so the pair is a near-miss and the illustration is gone. The mechanism
described was right and the example was one question wide, which is how a
worked example chosen from a table of this size tends to age.

The other forty-two are not fusion's fault: in most of them the fused rank is
identical to both component ranks, and where it differs, neither component had
the document inside k either. The `coverage` column - the share of the
question's distinct terms that occur anywhere in the document - says why. These
are mostly hard-slice questions written to avoid the target document's own
vocabulary, and both word retrievers index whole words, so they are two scores
over one representation that cannot see what neither can match.

Which lever recovers them is then an empirical question rather than an argument,
so the script re-runs every failing pair under all four (`*` marks a rank inside
k=5):

| id | document | expansion | rerank | mmr | hybrid3 | ngram-only |
| --- | --- | --- | --- | --- | --- | --- |
| q15 | evaluation | 9 | 18 | 17 | 13 | 1* |
| q24 | query-processing | 7 | 5* | 6 | 4* | 2* |
| q26 | citations | 5* | 8 | 9 | 3* | 1* |
| h01 | embeddings | 12 | 27 | 27 | 27 | 27 |
| h03 | bm25 | 12 | 5* | 5* | 5* | 3* |
| h06 | rag | 2* | 7 | 13 | 6 | 3* |
| h08 | reranking | 10 | 6 | 3* | 5* | 4* |
| h12 | evaluation | 5* | 6 | 10 | 6 | 13 |
| h15 | rank-fusion | 5* | 10 | 15 | 9 | 6 |
| h18 | rag | 5* | 13 | 15 | 11 | 6 |
| h20 | reranking | 22 | 14 | 14 | 9 | 4* |
| h30 | embeddings | 3* | 17 | 19 | 12 | 7 |
| h32 | negative-mining | 3* | 7 | 10 | 8 | 5* |
| h45 | bm25 | 7 | 6 | 6 | 4* | 4* |
| h48 | score-normalisation | 14 | 2* | 7 | 2* | 1* |
| h51 | relevance-feedback | 20 | 5* | 7 | 4* | 3* |
| h60 | learning-to-rank | 5* | 7 | 9 | 9 | 3* |
| h62 | click-models | 7 | 8 | 6 | 4* | 4* |
| h65 | chunking | 8 | 7 | 6 | 4* | 3* |
| h67 | evaluation | 15 | 7 | 8 | 6 | 4* |
| | **recovered (all 48)** | 8/48 | 5/48 | 2/48 | 10/48 | **23/48** |

**The character n-gram list alone recovers twenty-three of the forty-eight
failures the word retrievers share - just under half.** It is the only lever in
the repo that changes the *representation* rather than re-scoring or re-ordering
the same one, and the gap to everything else has widened: hybrid3 recovers ten,
expansion eight, re-ranking five, MMR two. Re-ordering a pool cannot admit a
document the pool never contained, which is why the two pool-bound levers sit at
the bottom.

MMR's row is worth a note because it moved the wrong way for a clean story: it
went 1/16, then 0/22, and is now **2/48**. The previous revision read the 0/22
as "the pool-bound principle tightening". Two recoveries at twenty-eight
documents show that reading was over-drawn - MMR can recover a pair when the
relevant document is already in the candidate pool at k=20 and redundancy is
what is keeping it out of the top five, which is exactly what it was built for.
The rate is still 4%, so the conclusion is unchanged and only the absolute claim
is withdrawn.

Expansion is the row that keeps moving with the corpus. At 14 documents it
recovered one failure; at 16, three; at 18, four; at 20, five; and now eight of
forty-eight. It is a lever with wide variance whose good runs and bad runs are
both properties of the feedback pool, not of the code - which is the same
instability that cost a pinned test its pin this increment.

It also explains a result from the section above. The three-list table shows
`ngram` beating `hybrid3` on the hard slice's ordering columns (nDCG@5 0.662 vs
0.596, recall@5 0.762 vs 0.676), which reads as a fusion that dilutes its best
member. This table is the same fact at question granularity: on the pairs where
the word lists both fail, the n-gram list carries twenty-three rescues and
fusion keeps ten of them.

**Twenty-eight of the forty-eight are recovered by at least one lever. The
backlog that survives all five has grown from eight pairs to twenty**, and five
of those twenty involve the two documents added this increment:

```
nothing in the repo recovers these (20 pairs; 5 shown):
  h01 -> embeddings (rank 27, coverage 0.28): If someone writes automobiles but
         the page only ever says cars, what kind of search still finds it?
  h05 -> chunking (rank 24, coverage 0.39)
  h61 -> click-models (rank 7, coverage 0.48): Something got picked constantly
         because it sat at the top, not because it was any good. How do you tell
         those two apart?
  h66 -> deduplication (rank 11, coverage 0.39): Comparing every item against
         every other one is too slow. What buys a cheap shortlist of likely
         matches instead?
  h66 -> ann-search (rank 21, coverage 0.33)
```

`h01` is the honest one, and it is dead last of twenty-eight under all five
configurations. The question asks for the retrieval method that survives a
synonym, using a car/automobile analogy the corpus never uses; the answer is in
`embeddings.md`, which never says either word. Character n-grams fix
*morphology* - "controls" against "control" - not synonymy, so the one lever
that rescues nearly half the rest cannot touch this. Nothing short of a semantic
representation will, which makes `h01` a fair statement of where a
dependency-free design actually stops rather than a bug to be tuned away.

`h66` is the new one worth naming, because it is a *two-document* question and
both halves fail. It asks how to avoid comparing everything against everything,
which `deduplication.md` answers with locality-sensitive hashing and
`ann-search.md` answers with IVF and graph indexes - and the question uses
neither document's vocabulary, so nothing retrieves either. It was written to
span the two and it spans them by being equally invisible to both, which is a
harder question than intended rather than a better one.

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
| easy | 81 | 0.686 | 0.622 | 6.7 |
| hard | 68 | 0.493 | 0.334 | 12.1 |

A label-shuffle permutation test - unpaired, because these are two disjoint
sets of different questions and there is nothing to pair - puts both columns
below 5e-05 at 20000 resamples. That is a real vindication rather than a
formality: the label could easily have been noise, and the honest thing was
that nobody knew.

**But it is not one axis.** Crosstabbed against the size of the answer key, the
label turns out to be a proxy for it as much as for anything lexical:

| label | multi-document | single-document |
| --- | --- | --- |
| hard | 43 | 25 |
| easy | 0 | 81 |

Fisher's exact test gives p = 4.8e-20, and the interesting cell is the zero:
*every* multi-document question in the gold set is labelled hard, and no easy
question has more than one relevant document. So a hard-slice number is not a
lexical-difficulty number. It is a blend of a lexical effect and a
"find-two-documents-instead-of-one" effect, and the two have to be separated
before either can be quoted. Holding the answer-key size fixed separates them,
and the spread it exposes is wide (hit@1, k=5):

| slice | n | bm25 | hybrid2 | hybrid3 |
| --- | --- | --- | --- | --- |
| hand hard (all) | 68 | 0.500 | 0.515 | 0.662 |
| hand hard, single-document | 25 | 0.280 | 0.280 | 0.440 |
| hand hard, multi-document | 43 | 0.628 | 0.651 | 0.791 |

The reported 0.500 is an average over two sub-populations that differ by 0.348
- the single-document hard questions are well over twice as hard as the
multi-document ones the label groups them with, and that gap has widened from
0.312 at twenty-six documents. Nothing above this line is wrong because of that,
but "the hard slice" has been doing duty as a single population and it is two.

**The payoff: the third list's gain lands exactly where its mechanism says it
must.** Stratifying along measured containment instead of the hand label, on
single-document questions so the multi-document effect is held out (hit@1):

| idf-containment | n | bm25 | hybrid2 | hybrid3 | hybrid3 - bm25 |
| --- | --- | --- | --- | --- | --- |
| [0.00, 0.35) | 29 | 0.276 | 0.276 | 0.448 | **+0.172** |
| [0.35, 0.55) | 23 | 0.826 | 0.826 | 0.913 | **+0.087** |
| [0.55, 0.75) | 30 | 1.000 | 1.000 | 1.000 | +0.000 |
| [0.75, 1.00] | 24 | 0.875 | 0.875 | 0.875 | +0.000 |

The gain declines across the range and reaches *exactly* zero in the top two
bands, for the third corpus size running. It is monotone at this cut, though
that part does not survive a finer one and the third limit below is where it
breaks - the concentration at low containment is the claim, not the
monotonicity. That is the prediction the module was written to test and it could
have failed: n-grams exist to match terms the word index scores at zero, so
their value has to be concentrated where the question's words are missing from
the answer and has to vanish where they are all present. An aggregate delta of
+0.047 over the whole gold set says the third list helps. This says where it
helps, why, and where it provably does nothing - and a flat profile would have
meant that delta was coming from somewhere other than the lexical gap the module
claims.

The two non-zero bands both shrank this increment - +0.250 to +0.172 and +0.158
to +0.087 - while the two zero bands stayed exactly zero. The shape held and the
amplitude fell, which is consistent with seventeen new questions that are harder
in the lexical sense the top bands cannot express.

The `hybrid2` column is the same result from the other side, and it is blunter
than the aggregate tables managed to be: fusing BM25 with TF-IDF buys **0.000
in every single band**. The design note about two scores over one
representation not being two retrievers stops being an argument at that point;
the two lists share a tokenizer, so they share their containment blindness
band for band.

**The negative control, because a sceptic is owed it.** Hard questions are also
*longer* questions here (12.1 terms against 6.7), and containment's denominator
is the term count, so "long questions are hard" is a live rival explanation
that would make containment a restatement of length. Stratifying the same
questions by term count instead:

| terms | n | bm25 | hybrid3 | hybrid3 - bm25 |
| --- | --- | --- | --- | --- |
| [0, 7) | 37 | 0.838 | 0.892 | +0.054 |
| [7, 10) | 47 | 0.787 | 0.851 | +0.064 |
| [10, 14) | 14 | 0.571 | 0.643 | +0.072 |
| [14, inf) | 8 | 0.250 | 0.375 | +0.125 |

Flat and non-monotone in the direction that matters - it never reaches zero, and
where the containment profile *falls* to exactly zero in its top two bands, this
one *rises* across the range. Length does not reproduce the profile, and length
correlates with containment at only r = -0.434, so the two are not the same
measurement. The rival hypothesis is the one that fails here, which is the only
reason the containment table above is worth believing.

Three limits, stated where they apply rather than at the end:

- **Containment is not discriminativeness.** It measures whether the question's
  terms are in its answer, not whether they are *only* there. A question whose
  every term appears in its answer and in six other documents scores 1.000 and
  is still a coin flip. The top band is where that shows: bm25 hit@1 falls from
  1.000 to 0.875 going from the third band to the fourth, which is the proxy
  failing, not the retriever.
- **The bands hold 23-30 questions each.** At those counts one question is
  worth 0.03-0.04, so the +0.172 and +0.087 are two-to-five-question effects
  and the two zeros are the load-bearing entries - they are exact, not
  rounded. Read the shape of the profile, not the digits.
- **The band edges were fixed before any metric was computed**, at 0.35 / 0.55
  / 0.75, to give comparable counts - but the profile is not indifferent to
  them, and `--edges` is how that was found out. Re-cut at two bands (0.50),
  three (0.30/0.60, 0.40/0.70) and the shipped four, the decline holds and the
  top band stays at exactly +0.000 every time. Re-cut at **five**
  (0.25/0.45/0.65/0.85) it breaks: the deltas run +0.190, +0.158, +0.000,
  +0.042, **-0.083**, so the monotonicity fails twice over in the last three
  bands. That top band holds 12 questions, where one row is worth 0.083 and the
  -0.083 *is* one row. The defensible claim is therefore the weaker one - the
  gain is concentrated at low containment and reaches zero by the top - and
  **not** strict monotonicity, which is an artifact of a cut coarse enough to
  hide the noise. Four bands over 106 questions was already the sweep most
  likely to be fitting itself; this is it caught doing so.

### How much of any of that is real? Error bars on the ablation table

Every table above this line reports a mean over the same 149 questions, and
several of the differences in them are one question wide. That is not a
rhetorical point: at n=149 one question is worth 0.007 of any per-query metric,
and the hard slice - 68 questions - cannot resolve anything finer than 0.015.
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

The block below is a **historical record from the twenty-document run**, kept
because it is the evidence for the paragraph above, not a current reading. The
live figures are further down this section.

```
hard slice - 38 questions, one question is worth 0.026          [20 documents, historical]
metric    comparison                 A       B    delta              95% CI       p  p_holm   floor       W/L/T
----------------------------------------------------------------------------------------------------------------
hit@1     bm25 vs hybrid2        0.526   0.526    0.000    [+0.000, +0.000]   1.000   1.000   1.000      0/0/38
recall@5  bm25 vs hybrid2        0.697   0.697    0.000    [-0.039, +0.039]   1.000   1.000   0.500      1/1/36
ndcg@5    bm25 vs hybrid2        0.604   0.603    0.001    [-0.020, +0.021]   1.000   1.000   0.062      3/2/33
mrr       bm25 vs hybrid2        0.674   0.674    0.000    [-0.013, +0.013]   1.000   1.000   0.500      1/1/36
```

The hard-slice comparison the project was built around was, at that size, a row
of exact zeroes with `0/0/38` on hit@1 - fusion and BM25 returning the same
verdict on every single hard question. A tie measured that precisely is a more
informative result than the one-question win it replaced. At 149 questions the
same pair reads `0/1/67` on hard-slice hit@1, so the tie has loosened by exactly
one question in eight corpus sizes.

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

**Twenty comparisons now survive the familywise correction, up from six, and the
change arrived with the gold set rather than with any change to the code.**
The family did not grow - it is 150 tests at both sizes - so this is a power
gain, not a multiplicity artefact. Seventeen questions took the survivor count
from 6 to 20.

Every one of the twenty involves either the standalone `ngram` list or
`hybrid3`. **Not one is among `bm25`, `vector` and `hybrid2`**, which is the
same verdict this section has returned at every corpus size: among the three
word-level methods, nothing is significant anywhere, on any metric, on any
slice.

The headline claim is unchanged in direction and stronger in evidence.
`hybrid3` beats `hybrid2` on nDCG@5 over the all slice by **+0.045**, 27 wins /
3 losses / 119 ties, raw p = 0.000, **p_holm = 0.015**. It is worth being exact
about what that claims: *adding the character n-gram list to the fusion beats
the two-list fusion*. It does not claim the fusion beats the n-gram list alone -
that comparison is +0.035 the other way at p_holm 1.000, still NOT SHOWN - and
it does not rescue two-list fusion, which remains NOT SHOWN against BM25 alone
(-0.003, 5W/10L/134T, and now trailing on the point estimate where it used to
lead).

**The lesson this section drew last increment has been overturned by this one.**
It read: `ngram` vs `hybrid2` is a much larger delta and yet *fails* Holm while
the smaller `hybrid3` vs `hybrid2` passes, because Holm orders by raw p and the
bigger delta was noisier - "consistency beat magnitude". At 149 questions
`ngram` vs `hybrid2` on nDCG@5 is +0.080 at 42W/12L and **p_holm = 0.015**: it
now survives too. The mechanism described was real, but it was a statement about
a sample size rather than about the comparison. The larger, noisier effect
needed more questions to clear the same multiplier and has now got them.
Consistency bought significance earlier; magnitude got there as well.

The `W/L/T` column explains the rest, and is the number worth reading first.
Among the three word-level retrievers the discordant counts on nDCG@5 / all are
29 for `bm25` vs `vector`, 22 for `vector` vs `hybrid2` and 15 for `bm25` vs
`hybrid2` - so the comparison's real sample size is never 149, it is the
discordant count, and every tie carries no evidence either way. The largest of
those three went 30 at twenty-six documents to 29 at twenty-eight: **seventeen
new questions added one net disagreement between BM25 and TF-IDF, and removed
one.** That is the measured rate of return on gold-set growth for the word-level
claims and it is approximately zero, which is exactly what the corpus-lever
section keeps finding. The n-gram comparisons are where the growth paid off,
because that is where the methods actually disagree: `bm25` vs `ngram` sits at
56 discordant questions. Writing questions the retrievers *disagree* on is a
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
and that the largest disagreement anywhere is 13 out of 92. That was true when
it was written, and it was also an artefact of *which* retrievers were being
compared. Until this
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
Add the character n-gram list and the largest discordant count on nDCG@5 is
**56** - `bm25` against `ngram`, 15 wins to 41 losses - against 29 for the
largest word-level pair. Nearly doubling the evidence did not require a single
new gold question; it required comparing methods that fail differently. The
earlier reading, that gold-set growth was the only lever on discordance, was too
pessimistic: *method diversity* is the other lever, and it was free.

The two levers are worth comparing directly, because this increment ran the
other one. Seventeen new questions moved the largest word-level discordant count
from 30 to 29. Adding a structurally different retriever moved it from 29 to 56,
at no labelling cost at all.

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

| | 149 questions | 132 questions |
|---|---|---|
| comparisons in a full run | 150 | 150 |
| expected spurious at raw α=0.05 | 7.5 | 7.5 |
| **observed** at raw p<0.05 | **52** | 42 |
| surviving Holm | **20** | 6 |

Fifty-two rows cross the uncorrected threshold and twenty survive, which is the
whole argument for the column. Anyone reading the raw column alone would come
away with fifty-two confident findings; the honest count is twenty, and 52
against an expected 7.5 is what a family this size does to an uncorrected table.
The left-hand column is 149 questions and the right-hand one 132.

**An older revision of this table was wrong in two incompatible ways at once.**
It carried a table reading 14 raw and 0 surviving, and a paragraph directly
beneath it reading 29 raw and 1 surviving. Those cannot both be right, and
re-running the 118-question configuration from that commit gives **44 and 8**,
so neither was. The provenance test catches drift in table *captions*; it does
not read the numbers inside a table or a fenced block, which is precisely where
those two sat. That is a real limit of the test and it is worth stating next to
the thing it failed to catch.

**Growing the gold set has now moved the survivor count in both directions, and
the composition is the part worth reading.** Fourteen questions took survivors
from 8 to 6 between 118 and 132; seventeen took them from 6 to 20 between 132
and 149. More data usually buys power, and the middle step looked like the
opposite until the survivors were listed rather than counted: at 118 seven of
the eight involved `ngram` *alone*; at 132 not one did and all six involved
`hybrid3`; at 149 both families are represented, with the `ngram`-alone
comparisons returning alongside the `hybrid3` ones.

The 6-to-20 jump is the largest power gain this project has measured from a
single increment, and the metric breakdown says where it came from. At 132 the
survivors were 4 nDCG@5 rows and 2 MRR rows, and **recall@5 contributed none**.
At 149 they are 11 nDCG@5, 5 recall@5 and 4 MRR. Recall@5 went from
contributing nothing to contributing a quarter of the total, which is what six
new multi-document questions buy: recall@5 is numerically identical to hit@5 on
a single-document question, so only the 43 multi-document ones give that metric
anything to say, and this increment took that count from 37 to 43.

The three claims the project's own prose makes are checked by name, so they
cannot quietly drift out of agreement with the measurement:

```
claims this project makes, tested on ndcg@5 / all slice
NOT SHOWN   ngram vs hybrid3     delta +0.035  p_holm 1.000  30W/14L/105T
SUPPORTED   hybrid3 vs hybrid2   delta +0.045  p_holm 0.015  27W/3L/119T
NOT SHOWN   hybrid2 vs bm25      delta -0.003  p_holm 1.000  5W/10L/134T
```

**The middle line has now changed verdict twice, in both directions, and this
section predicted both.** At eighteen documents all three read NOT SHOWN, and
the prose said of `hybrid3 vs hybrid2`: "it is defeated by the family size
rather than by the data; it is the claim most likely to survive a larger gold
set." A later revision recorded it as SUPPORTED. At twenty-four documents it
read NOT SHOWN at p_holm **0.057** - the wrong side of its own line by four
thousandths - and the power section wrote that this was "the strongest single
argument in this file for growing the gold set before trusting any of these
verdicts." At twenty-six it was SUPPORTED again at p_holm **0.015** with a
21W/2L split gone to 25W/0L, and at twenty-eight it stays SUPPORTED at the same
p_holm with **27W/3L**.

**Those three losses are worth flagging rather than rounding away.** The
previous revision made something of the 25W/0L split - "every one of the
twenty-five questions the two systems now disagree on is a win for the
three-list fusion". Three of the seventeen new questions go the other way, so
that sentence is retracted: the claim survives, the perfect record does not, and
a 0L column was always more likely to be a small-sample artefact than a property
of the fusion.

That sequence is the cleanest demonstration in this file of what the whole
evaluation apparatus is for. The underlying system was not touched across those
four measurements. What moved was the amount of evidence, and the verdict moved
with it - out, back, and out again - which is exactly what a claim sitting near
a significance boundary is supposed to do and exactly why the boundary should
not be reported as though it were a property of the retriever.

The three still fail or pass for three different reasons worth telling apart.
`hybrid2 vs bm25` is a *dead* claim: fifteen discordant questions splitting
5W/10L for a delta of -0.003 - fusing BM25 with TF-IDF does essentially nothing
here, which the earlier `rrf_k` section reached independently and the main
ablation table now shows directly. **Its point estimate has been negative,
then marginally positive, and is now negative again across three corpus sizes,
never once approaching significance**, which is the most precise way this file
can say "no effect". `ngram vs hybrid3` is the claim the fusion would least like
to be true and it remains NOT SHOWN, though its win share of 0.68 is the
strongest it has looked. `hybrid3 vs hybrid2` is the survivor. (Restricting the
run to one slice with `--slices all` shrinks the family from 150 tests to 50,
which was how this comparison bought its evidence cheaply before the gold set
was large enough to pay full price.)
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
not a prediction about the next 149 questions - it is the best of seven draws
from the same 149.

`evals/holdout.py` measures the size of that bias by nested cross-validation:
split the questions into folds, pick the winner on the training folds, score
that choice on the held-out fold, and repeat over twenty different splits.

```bash
python evals/holdout.py
python evals/holdout.py --slice hard --folds 4
python evals/holdout.py --no-chunking      # query-time knobs only
```

```
149 questions (all slice), k=5, 5-fold x 20 splits, seed 0
one question is worth 0.007 of any per-query metric; a held-out fold is 29 questions

sweep           metric    picked          in-sample   held-out   optimism   vs default   stable
-----------------------------------------------------------------------------------------------
rrf_k           hit@1     60                  0.711      0.711     +0.000       +0.000     100%
rrf_k           ndcg@5    1                   0.742      0.741     +0.002       -0.002       0%
rrf_k           mrr       1                   0.777      0.776     +0.001       -0.000       0%
fusion weights  hit@1     ngram=4             0.799      0.797     +0.001       +0.046      90%
fusion weights  ndcg@5    ngram only          0.822      0.818     +0.003       +0.031      70%
fusion weights  mrr       ngram=6             0.844      0.835     +0.009       +0.017       0%
mmr lambda      hit@1     off                 0.711      0.711     +0.000       +0.000     100%
mmr lambda      ndcg@5    0.9                 0.746      0.744     +0.001       +0.002      70%
mmr lambda      mrr       0.7                 0.779      0.776     +0.003       -0.001      10%
chunking        hit@1     80:20               0.725      0.694     +0.031       -0.017       0%
chunking        ndcg@5    150:40              0.771      0.765     +0.007       +0.022      45%
chunking        mrr       150:40              0.799      0.782     +0.017       +0.006       0%
```

**The `chunking` rows are new, and they are the reason the other three were not
enough.** Every sweep above them is *query-time*: one fitted engine serves the
whole sweep, because the knob only re-scores or re-orders lists that already
exist. Chunking is index-time - each configuration is a different index - and
that is why it sat outside this harness for six corpus sizes while the chunking
table upstream went on printing an in-sample winner with nothing to check it
against.

Fixing it required no change to `ragkb/holdout.py` at all, which is worth
stating because it is the part that was mis-diagnosed. That module consumes
`config -> per-question scores` and is indifferent to whether producing those
scores cost a re-fusion or a rebuild. The limitation was entirely in the eval
script, which held one engine and handed it to every sweep. And the stated
reason for the exclusion - that a rebuild per row is too expensive - turns out
to be worth about half a second for all ten rows on this corpus. It was a guess
about cost rather than a measurement of one, so the sweep now runs by default
and `--no-chunking` opts out.

Read `optimism` first, on the `fusion weights / hit@1` row. The weighted-RRF
table earlier in this README reports `ngram=4` at 0.799 against uniform's 0.752
and invites exactly the conclusion it looks like. Scored on questions the choice
never saw, that configuration earns **0.797** - the headline shed only 0.001,
well under one question.

**That penalty has now gone up, down, up, down, up and sharply down across six
corpus sizes** (+0.043 at 16, +0.013 at 18, +0.032 at 20, +0.008 at 24, +0.016
at 26, +0.001 at 28), and the eighteen-document revision of this section read
the middle value as progress: "the first sweep row in the project that has
survived cross-validation with its margin substantially intact". That reading
was wrong then and it would be wrong now for the same reason - six measurements
reading 0.043, 0.013, 0.032, 0.008, 0.016, 0.001 are a distribution, not a
trend, and the newest one being the smallest is not evidence that the problem is
solving itself.

**What genuinely did change this increment is the `stable` column, and it is the
one result here that looks like real power rather than noise.** At twenty-six
documents every `fusion weights` row sat at or near 0% - the winner changed on
essentially every one of the twenty splits. At twenty-eight, hit@1 is at **90%**
and nDCG@5 at **70%**. For the first time in this project a sweep is mostly
picking the same configuration regardless of how the questions are split, which
is what a sweep has to do before its top row means anything. MMR's nDCG@5 row
moved the same way, 0% to 70%. MRR is the holdout: 0% on fusion weights and 10%
on MMR, so the improvement is real but not uniform.

Two controls were supposed to make the optimism number trustworthy rather than
merely encouraging, and they are why the script sweeps three different knobs
instead of the one it wanted to indict. **Both broke at twenty-six documents,
and both are still broken at twenty-eight.**

- **`rrf_k` still does not pick the default on every metric.** At twenty-four
  documents every row picked `60` at 100% stability with exactly zero optimism,
  and this section said the control "now agrees on every metric without the
  caveat". At twenty-six and again at twenty-eight, nDCG@5 and MRR both pick
  `rrf_k = 1` at **0%** stability with +0.002 and +0.001 optimism, and only
  hit@1 picks the default at 100%. That restores the illustration an earlier
  revision deleted: optimism and stability are independent, and a 0% stable row
  with +0.002 optimism means "nothing to choose", not "a choice that will not
  generalise". The underlying control is intact - `evals/tune.py` establishes
  `rrf_k` is inert over two lists, so there is nothing for selection to inflate.
- **MMR still does not pick the default either**, which breaks the sentence that
  once followed this table - "selection cannot inflate a sweep whose winner is
  the default". nDCG@5 picks `0.9` and MRR picks `0.7`; only hit@1 picks `off`.
  The `vs default` column on the nDCG@5 row has turned positive (+0.002) for the
  first time, at 70% stability, which is the first out-of-sample evidence in this
  file that MMR buys anything at all. It is a quarter of a question, so it is
  recorded rather than acted on. Note that the MMR sweep table above
  independently found 0.7 to be its best row on hit@5 and MRR, so these two
  harnesses continue to agree.

**What the index-time rows say, in the order they should be read.** First,
`optimism`: +0.038, +0.021 and +0.026 are the three largest figures in the
table, roughly twice the worst query-time row and an order of magnitude above
`rrf_k` and MMR. That direction is not a surprise and it is the control working
- `rrf_k` is inert over two lists, so there is nothing for selection to inflate,
whereas ten chunkings really are ten different indexes and selection has
genuine variance to exploit. What is worth the ink is the *size*: +0.031 at
hit@1 is nearly five of the 149 questions, evaporating purely because the winner
was chosen on the questions it was scored on.

Second, and this is where the previous revision's finding has to be softened.
At twenty-six documents `vs default` was negative on **all three** metrics, and
this section said the chunking table "is not a table whose lead is overstated -
it is a table whose entire procedure loses to ignoring it." At twenty-eight it
is negative on **one of three**: hit@1 -0.017, but nDCG@5 **+0.022** and MRR
**+0.006**. On nDCG@5 the sweep now picks `150:40` both in sample and out, at
45% stability, and beats the shipped default out of sample by three questions.
The strong claim is withdrawn. What survives is the weaker and still useful one:
chunking carries by far the largest optimism in the table, and on two of three
metrics the winner is never the same twice.

Third, `stable` is **0%** on the hit@1 and MRR rows: across a hundred folds the
winner is never the same twice on either. nDCG@5 at 45% is the exception and is
the reason the `vs default` sign flipped on that row - a sweep that picks the
same configuration on nearly half the splits is a different object from one that
picks a new configuration every time, and the out-of-sample column notices.

One caveat on the `picked` column at hit@1, because it is a tie rather than a
win: `80:20`, `120:15` and `120:60` all score exactly 0.725 in-sample, and the
offer order resolves it. `ragkb.holdout` breaks ties on the order configurations
are offered, and `chunking_sweep` puts the shipped default first precisely so a
tie resolves towards not moving - but between non-default settings the tie-break
is arbitrary. It does not affect the held-out figure, which averages over folds
that mostly disagree anyway.

The reversal worth recording is `ngram only`, and it has now held three times
running. At sixteen documents it was the control the weighted fusion could not
beat out of sample on nDCG@5. At eighteen the holdout picked `ngram=4` on all
three metrics and the control lost even out of sample. At twenty the holdout
picked `ngram only` back for nDCG@5, and at twenty-four, twenty-six and
twenty-eight it still does - and its stability, having gone 0%, 40%, 5%, is now
**70%**. Six measurements, two reversals, one conclusion that has never actually
changed: **the fusion has still not been shown to beat its own best input**, and
that conclusion is now being reached by a sweep that mostly agrees with itself.

**Four of the twelve** sweep/metric pairs have a negative `vs default` - on
those, the honest estimate of tuning is worse than shipping the constant and
never sweeping. They are the two `rrf_k` rows (nDCG@5, MRR), the MMR MRR row,
and chunking's hit@1. Three of those four were negative at twenty-four and
twenty-six documents too; the count is down from seven because the chunking
sweep went from losing on all three metrics to losing on one.

That shifts the balance of the table rather than adding a footnote to it. Of the
four knobs now cross-validated, the fusion weights are positive against their
default on every metric - and the significance section has already recorded that
the comparison this sweep is built around does not survive correction. `rrf_k`
loses on two of three, MMR on one of three, chunking on one of three.

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
comparisons disagree on 19.9% of questions on average, about 30 discordant pairs
out of 149. At that rate **the smallest effect 149 questions can detect at 80%
power is a win share of 0.766** - one method has to take 77% of the
disagreements. Against a genuinely small effect (win share 0.60) this gold set
has **14.1% power**. It would miss a real difference of that size nearly nine
times in ten.

The detectable bar has gone 0.789 at 118 questions, 0.775 at 132 and 0.766 at
149, and power against a small effect 11.9%, 13.1%, 14.1%. **That is the going
rate and it is worth quoting because it is unimpressive**: three increments and
thirty-one questions have bought about two points of power against a small
effect, and the bar is still asking one method to win more than three
disagreements in four.

It is worth putting that next to the twenty survivors the significance section
gained this increment, because the two readings look contradictory and are not.
Holm survivors more than tripled while the detectable-effect bar moved by 0.009.
Nothing got easier to detect; what happened is that several comparisons already
close to the line crossed it. The power table is about the effects this gold set
*could* find. The survivor count is about the effects that happen to be present
and large. Growth helps the second far more visibly than the first, which is
exactly the trap this section exists to flag.

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

| pair, nDCG@5 | 26 docs, carried 132 | 28 docs, **same** 132 | switched | exact p |
| --- | --- | --- | --- | --- |
| hybrid2 vs bm25 | 13.6% (18) | 10.6% (14) | 3 gained, 7 lost | 0.344 |
| hybrid3 vs hybrid2 | 18.9% (25) | 19.7% (26) | 4 gained, 3 lost | 1.000 |
| ngram vs hybrid3 | 30.3% (40) | 30.3% (40) | 2 gained, 2 lost | 1.000 |

Four increments, eleven paired measurements, nothing significant, and the sign
has now flipped in both directions more than once. **The honest standing of
"documents are the lever" is that it is unsupported**, and it was only ever
supported by reading a paired design as an unpaired one. Two documents moved the
discordance rate by -3.0%, +0.8% and +0.0% on the three pairs, and the largest
of those is ten questions changing hands in a set of 132 with an exact p of
0.344. The bottom row is the cleanest null this measurement has produced: the
rate is identical to three decimal places, and it got there with two questions
becoming discordant and two stopping.

The seventeen new questions, meanwhile, landed at 5.9%, 23.5% and 23.5% against
prevailing rates of 10.6%, 19.7% and 30.3% - **above the prevailing rate on one
pair of three, which is the worst showing question-writing has had.** The two
previous increments beat the prevailing rate on all three pairs each. So the
running tally is seven paired measurements where new questions beat the
prevailing rate against eleven where the document lever returned nothing, and
this increment weakened the question side of that asymmetry rather than
strengthening it.

That is worth setting against what the same seventeen questions did to the Holm
survivor count, which tripled. Discordance on the three *claim* pairs and power
across the whole 150-test family are not the same quantity, and this increment
is the clearest case yet of them moving in opposite directions.

What survives about documents is narrower and worth stating on its own: a new
document re-prices every question already in the set, which is a mechanism, not
a claimed direction. It can raise the discordance rate or lower it, and at this
corpus size the measurement still cannot tell which it did.

Then the three claims, priced against their own observed splits:

| claim | W/L/T | rate | obs. share | questions needed | verdict at n=149 |
| --- | --- | --- | --- | --- | --- |
| ngram vs hybrid3 | 30/14/105 | 29.5% | 0.68 | 212 | NOT SHOWN |
| hybrid3 vs hybrid2 | 27/3/119 | 20.1% | 0.90 | 62 | **SUPPORTED** |
| hybrid2 vs bm25 | 5/10/134 | 10.1% | 0.33 | 742 | NOT SHOWN |

**The last column is not an input to the fourth.** Required-sample-size is
derived from the binomial alone, with no knowledge of what the significance
harness concluded. At twenty-eight documents the two agree three times out of
three, as they did at twenty-six. The middle row is the interesting agreement:
the grid says 62 questions would suffice, the set holds 149, and the harness
returns SUPPORTED.

**The middle row got more expensive while staying supported, which is the
subtler of the two movements here.** At twenty-six documents its observed share
was a perfect 1.00 on a 25/0 split and the grid asked for 41 questions. The
three new questions that went against `hybrid3` drop the share to 0.90 and raise
the asking price to 62 - still well under the 149 in hand, so the verdict is
unchanged, but the margin between "what this claim needs" and "what this gold
set has" narrowed from 3.2x to 2.4x. A share of 1.00 was never going to survive
contact with more questions, and the grid pricing it at 41 was the grid
extrapolating from an impossible estimate.

**The top row moved the other way, in the project's favour for once.** `ngram vs
hybrid3` needed 434 questions at twenty-six documents and needs **212** now,
because the observed share went from 0.62 back up to 0.68. Required sample
scales roughly with the inverse square of the distance from 0.50, so a 0.06
movement in share halves the price. It is still NOT SHOWN and still needs more
questions than the set holds, but the direction reversed after an increment that
had made it four times more expensive - which is the clearest illustration on
this page that an "observed share" from thirty-odd discordant questions is an
estimate with a wide interval, not a measured property.

**And the bottom row now prints a number where it used to print a dash.** At
twenty-six documents `hybrid2 vs bm25` had an observed share of exactly 0.50 -
nine wins against nine losses - so the required sample diverged. At twenty-eight
it splits 5/10, a share of 0.33, and the grid dutifully asks for 742 questions
to establish that the fusion is *worse* than BM25 alone. That is not a finding;
it is the grid answering the question it was asked about an effect nothing here
has shown to exist.

Read the bottom row carefully, because it is the most useful line in the table
and the easiest to misread. `hybrid2 vs bm25` is the project's headline claim,
and its observed share is exactly 0.50 for the second increment running - the
nine discordant questions the fusion wins are matched by the nine it loses, so
the required sample diverges and the table prints a dash. That is the correct
output, not a gap: it reports "no effect worth sizing was seen here". After
twenty-eight documents and 149 questions the fusion-versus-BM25 comparison has
accumulated fifteen disagreements in total, splitting 5/10 - three fewer
disagreements than at twenty-six documents, on seventeen more questions.

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
carries error bars far wider than the grid above it. The `ngram vs hybrid3` row
halving its asking price this increment, after quadrupling it last increment, is
that width made visible. `ragkb.power.clopper_pearson`
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
pytest -q      # 599 tests
```

> That count was **575** for several increments and the tree collected 592
> before this one, so it had drifted by seventeen without anyone noticing -
> the same failure mode `tests/test_readme_provenance.py` was written to catch
> for table captions, in a number that test does not read. It is corrected
> here rather than quietly re-pinned, because a stale count in the sentence
> introducing the test suite is exactly the kind of small false assurance this
> README keeps arguing against.

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
in-sample maximum with unstable winners), the index-time half of that harness
(that the shipped default leads the offer order even when the caller never asked
for it, since `vs default` is undefined without it and offer order is what
breaks ties; that listing it explicitly does not offer it twice; that its column
reproduces a plain `RAGEngine` ranking for ranking, because every optimism
figure in those rows is measured against it; and - the test that would catch the
failure mode worth catching - that distinct chunk settings really do build
distinct indexes with differing rankings, since a sweep that quietly reused one
engine would report exactly zero optimism, which looks like a clean result and
means nothing), incremental indexing (that extending
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
