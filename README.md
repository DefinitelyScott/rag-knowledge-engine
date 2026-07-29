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
| `ragkb/answerer.py` | extractive answerer (offline) and OpenAI answerer, both cite |
| `ragkb/engine.py` | corpus loading, chunking, indexing, question answering |
| `ragkb/metrics.py` | hit@k, recall@k, precision@k, MRR |
| `ragkb/cli.py` | `stats`, `search`, `ask` |

`corpus/` holds the sample knowledge base (twelve documents on information
retrieval). `evals/gold.jsonl` holds thirty labelled questions with the
documents that answer them.

## Evaluation

```bash
python evals/evaluate.py
pytest -q
```

Current numbers on the bundled corpus and gold set (12 documents, 47 chunks,
30 questions, k=5, document-level relevance):

| method | hit@1 | hit@3 | hit@5 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| hybrid | 0.800 | 0.933 | 0.967 | 0.967 | 0.869 |
| bm25 | 0.800 | 0.933 | 0.967 | 0.967 | 0.868 |
| vector | 0.833 | 0.933 | 0.967 | 0.967 | 0.886 |

Reported honestly: on this corpus the hybrid retriever does **not** beat
TF-IDF alone. The corpus is small and each document uses a distinctive
vocabulary, so BM25 and TF-IDF largely agree, and RRF's rank-only view discards
the score margin that TF-IDF was using to break ties correctly. Hybrid
retrieval earns its keep when the two retrievers disagree; making them disagree
usefully here means a larger, more paraphrase-heavy corpus and harder
questions. That is the next thing to fix, not a number to tune away.

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

## Tests

```bash
pytest -q      # 56 tests
```

The suite covers tokenisation and chunk boundaries, BM25 saturation and length
normalisation, TF-IDF normalisation, RRF scoring shape, every metric, the CLI,
and an end-to-end pass over the real corpus and gold set.
