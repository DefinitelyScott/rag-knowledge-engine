"""What does adding one document to a live index actually save?

`evals/latency.py` measures the one-off index build and treats it as a single
number, because until now it was one: the engine had exactly two ways to exist -
built from a corpus, or restored from a snapshot - and both produce the whole
index at once. Adding a document meant rebuilding.

`RAGEngine.add_document` adds a third way, and the interesting thing about it is
that it is only half incremental. BM25's state is additive: counters, lengths
and postings append, document frequency is a running count, and the one
collection-wide quantity - IDF - is a scoring coefficient that can be refreshed
in a single pass over the vocabulary. TF-IDF's is not. Its IDF is baked into
every stored vector before L2 normalisation, so one new document changes the
weights *and* the length of every vector already stored, including vectors for
documents sharing no term with the new one. There is nothing to update; there
is only a refit.

This script measures that asymmetry rather than asserting it, in three parts:

1. **Stage costs.** Fitting each index from scratch over the whole corpus,
   against extending it with one document. The BM25 rows are the same operation
   at two sizes and should differ by a lot; the TF-IDF row is the same operation
   twice and should not differ at all, which is the finding.

2. **End to end.** Building an engine over N documents, against building it over
   N-1 and adding the last one. The gap between the two is the whole prize, and
   it is smaller than the BM25 row alone suggests, because the refit the vector
   list forces is still in there.

3. **Identity.** Every gold question, ranked by both engines, compared position
   by position. A cheaper build that returns a different ranking is not a
   cheaper build.

Absolute milliseconds describe this machine and this 14-document corpus. The
ratio between the two build paths is the portable part, and even that is a
statement about a corpus this size: the saving is a fixed cost removed
(tokenising and counting N-1 documents) against a cost that grows with the
collection (rebuilding every vector), so it gets *worse*, not better, as the
corpus grows. That is stated here rather than extrapolated from one data point.

Usage:
    python evals/incremental.py [--corpus corpus] [--k 5] [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.bench import Timing, format_timings, measure_one  # noqa: E402
from ragkb.bm25 import BM25Index  # noqa: E402
from ragkb.engine import EngineConfig, RAGEngine, read_corpus  # noqa: E402
from ragkb.text import chunk_document  # noqa: E402
from ragkb.vector import TfidfIndex  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_questions(path: str) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])
    if not questions:
        raise ValueError("gold set is empty: {}".format(path))
    return questions


def split_corpus(documents: Dict[str, str]):
    """Hold out the document that sorts last, so both build paths agree in order.

    Any document could be held out and the timings would be much the same, but
    only the last one lets the grown engine and the from-scratch engine hold
    their chunks in the same order - and without that, the identity check in
    part 3 would be comparing two different chunk numberings and failing for a
    reason that has nothing to do with the index being wrong.
    """
    held_out = sorted(documents)[-1]
    rest = {key: value for key, value in documents.items() if key != held_out}
    return rest, held_out


def stage_timings(
    documents: Dict[str, str], held_out: str, config: EngineConfig, repeats: int
) -> List[Timing]:
    rest = {key: value for key, value in documents.items() if key != held_out}

    def chunks_of(mapping: Dict[str, str]) -> List:
        out = []
        for doc_id in sorted(mapping):
            out.extend(
                chunk_document(
                    doc_id,
                    mapping[doc_id],
                    target_tokens=config.target_tokens,
                    overlap_tokens=config.overlap_tokens,
                )
            )
        return out

    old_tokens = [chunk.tokens for chunk in chunks_of(rest)]
    new_tokens = [chunk.tokens for chunk in chunks_of({held_out: documents[held_out]})]
    all_tokens = old_tokens + new_tokens

    def bm25_add() -> None:
        # The index is rebuilt to N-1 outside the timed region on every call:
        # add_documents mutates, so timing it repeatedly against one index
        # would measure adding the document a second, third and fourth time
        # to a collection that keeps growing.
        index = BM25Index().fit(old_tokens)
        index.add_documents(new_tokens)

    return [
        measure_one("bm25 fit (all)", lambda: BM25Index().fit(all_tokens), repeats=repeats),
        measure_one("bm25 fit(N-1)+add", bm25_add, repeats=repeats),
        measure_one(
            "tfidf fit (all)", lambda: TfidfIndex().fit(all_tokens), repeats=repeats
        ),
        measure_one(
            "tfidf fit (N-1)", lambda: TfidfIndex().fit(old_tokens), repeats=repeats
        ),
        measure_one(
            "chunk (all)", lambda: chunks_of(documents), repeats=repeats
        ),
        measure_one(
            "chunk (1 doc)",
            lambda: chunks_of({held_out: documents[held_out]}),
            repeats=repeats,
        ),
    ]


def grow_timing(
    documents: Dict[str, str], held_out: str, config: EngineConfig, repeats: int
) -> Timing:
    """Time the whole incremental path: build over N-1 documents, then add one.

    The base engine is rebuilt inside the timed region on purpose.
    ``add_document`` mutates the engine and refuses a duplicate id, so timing
    the call alone against one engine would either raise on the second repeat
    or - if the engine were rebuilt outside the region - report a number that
    no caller can ever obtain, since nobody has an N-1 index lying around for
    free. The comparable question is "starting from these documents, how long
    until I can query all N of them", and both rows here answer exactly that.
    """
    rest = {key: value for key, value in documents.items() if key != held_out}

    def grow() -> RAGEngine:
        engine = RAGEngine(rest, config=config)
        engine.add_document(held_out, documents[held_out])
        return engine

    return measure_one(
        "cold: build {} then add 1".format(len(documents) - 1), grow, repeats=repeats
    )


def warm_timing(
    documents: Dict[str, str],
    held_out: str,
    config: EngineConfig,
    repeats: int,
    warmup: int = 1,
) -> Timing:
    """Time ``add_document`` against an index that is already in memory.

    This is the case the method exists for - a long-lived process that has
    already paid to index N-1 documents and is handed the Nth - so the base
    engine must not be inside the timed region. It also must not be *reused*:
    ``add_document`` mutates and refuses a duplicate id, so a second call
    against the same engine would raise.

    Both constraints are met by pre-building exactly as many engines as
    :func:`ragkb.bench.measure` will make calls (``warmup + repeats``, one
    callable) and popping one per call. Nothing is constructed while the clock
    runs, and every sample adds to a genuinely N-1 sized index.
    """
    rest = {key: value for key, value in documents.items() if key != held_out}
    pool = [RAGEngine(rest, config=config) for _ in range(warmup + repeats)]

    def call() -> None:
        pool.pop().add_document(held_out, documents[held_out])

    return measure_one(
        "warm: add 1 to a live index", call, repeats=repeats, warmup=warmup
    )


def identical_rankings(
    documents: Dict[str, str], held_out: str, config: EngineConfig, questions: Sequence[str], k: int
):
    rest = {key: value for key, value in documents.items() if key != held_out}
    grown = RAGEngine(rest, config=config)
    grown.add_document(held_out, documents[held_out])
    whole = RAGEngine(documents, config=config)
    mismatches = []
    for question in questions:
        for method in grown.retriever.methods:
            left = grown.ranked_doc_ids(question, k=k, method=method)
            right = whole.ranked_doc_ids(question, k=k, method=method)
            if left != right:
                mismatches.append((question, method, left, right))
    return grown, whole, mismatches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    args = parser.parse_args(argv)

    documents = read_corpus(args.corpus)
    questions = load_questions(args.gold)
    config = EngineConfig(
        target_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens
    )
    _, held_out = split_corpus(documents)

    print(
        "corpus: {} documents; held out `{}` and added it back".format(
            len(documents), held_out
        )
    )

    print("\n1. stage costs")
    print(format_timings(stage_timings(documents, held_out, config, args.repeats)))
    print(
        "\n   `bm25 fit(N-1)+add` still fits N-1 documents, so it is not the cost of\n"
        "   the add - it is the cost of the whole path, and it lands near `bm25 fit\n"
        "   (all)` because fitting N-1 is most of fitting N. The row that matters is\n"
        "   the pair below it: `tfidf fit (all)` and `tfidf fit (N-1)` are the same\n"
        "   work, and adding a document pays the first one again from nothing."
    )

    full = measure_one(
        "cold: build all {} docs".format(len(documents)),
        lambda: RAGEngine(documents, config=config),
        repeats=args.repeats,
    )
    cold = grow_timing(documents, held_out, config, args.repeats)
    warm = warm_timing(documents, held_out, config, args.repeats)
    print("\n2. end to end")
    print(format_timings([full, cold, warm], baseline=full))
    print(
        "\n   All three rows end with an index over all {n} documents; they differ in\n"
        "   what they start from. The two cold rows start from nothing, and the\n"
        "   incremental one loses: reaching {n} through {m} pays the TF-IDF fit twice,\n"
        "   once at {m} and again at {n}, and that refit is the largest single stage in\n"
        "   the build. If you do not already have an index, build one.\n"
        "\n   The warm row starts from an index already in memory, which is the case\n"
        "   add_document exists for - a running process handed one new document. Its\n"
        "   multiplier is the real question the method has to answer, and it is\n"
        "   answering it against a rebuild of the entire corpus.".format(
            n=len(documents), m=len(documents) - 1
        )
    )

    grown, whole, mismatches = identical_rankings(
        documents, held_out, config, questions, args.k
    )
    print("\n3. identity")
    print(
        "   {} chunks either way; {} questions x {} methods compared at k={}".format(
            len(grown.chunks), len(questions), len(grown.retriever.methods), args.k
        )
    )
    if mismatches:
        print("   MISMATCHES: {}".format(len(mismatches)))
        for question, method, left, right in mismatches[:5]:
            print("   - [{}] {}".format(method, question))
            print("     grown: {}".format(left))
            print("     whole: {}".format(right))
        return 1
    print("   every ranking identical to the from-scratch build")
    print(
        "\n   Read the saving as a fixed cost removed against a growing cost kept:\n"
        "   add_document skips tokenising and counting the {} documents already\n"
        "   indexed, but still rebuilds every TF-IDF vector in the collection. At a\n"
        "   larger corpus the first term stays one document's worth of savings while\n"
        "   the second grows, so this ratio gets worse, not better.".format(
            len(documents) - 1
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
