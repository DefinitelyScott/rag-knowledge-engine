"""Which lever moved the discordance rate: the new documents or the new questions?

``evals/power.py`` establishes that only *discordant* questions carry evidence,
and that raising the discordance rate is a cheaper route to a testable claim
than writing more questions at the prevailing rate. That leaves an obvious
follow-up which the README answered once, by hand, at twenty-two documents:
when an increment adds documents *and* questions together, which half did the
work?

The two are separable because they act on different things. A new **question**
is one more draw at whatever rate the corpus currently supports. A new
**document** re-prices every question already in the set, because it changes
what those questions compete against. So holding the question set fixed at the
ones that predate the increment, and changing only the corpus, isolates the
document effect exactly:

* **before / carried** - the old corpus, scored on the questions that do not
  mention the new documents.
* **after / carried** - the new corpus, scored on *the same* questions. The
  difference between this row and the one above is the document lever, measured
  on a question set that was not edited.
* **after / new** - the new corpus, scored on the added questions alone. Above
  the prevailing rate means the questions out-discriminated; at it means they
  added draws rather than discrimination.
* **after / all** - what the next run of ``evals/significance.py`` will see.

Every rate is printed with an exact Clopper-Pearson interval, because these are
proportions over counts that are frequently single-digit and the point estimate
alone is not readable. A twelve-question increment yielding two disagreements
estimates a rate to roughly 2% - 48%; without the interval it reports "16.7%"
and invites a comparison it cannot support.

This script measures; it does not judge. A run that shows the new questions
landing below the prevailing rate is a finding to report, not a failure - it is
the finding the README already records for the twenty-two-document increment.

Usage:
    python evals/corpus_lever.py
    python evals/corpus_lever.py --added late-interaction,ann-search
    python evals/corpus_lever.py --pair hybrid3,hybrid2 --metric hit@1
    python evals/corpus_lever.py --added "" --new-questions h77,h78,h79
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkb.engine import EngineConfig, RAGEngine, read_corpus  # noqa: E402
from ragkb.power import clopper_pearson, mcnemar_exact  # noqa: E402
from ragkb.significance import Comparison, compare  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Reuse the sibling harness's routing table, gold loader and metric dispatch for
# the same reason ``evals/power.py`` does: two scripts that resolve the label
# "hybrid2" independently are two scripts that will eventually disagree about
# what it means.
_spec = importlib.util.spec_from_file_location(
    "significance_eval", os.path.join(HERE, "significance.py")
)
significance_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(significance_eval)

#: Documents treated as "new" when ``--added`` is not supplied. These are the
#: two the current increment introduced; an older increment can be re-measured
#: at any time by naming its documents instead.
#:
#: This constant silently went stale for three increments. It was set to
#: ``relevance-feedback,score-normalisation`` when this file was written at
#: twenty-two documents, and the corpus then grew to twenty-four, twenty-six and
#: twenty-eight without it being touched. Running the script with no flags
#: therefore measured a two-increment-old document lever while printing a header
#: that said "added: relevance-feedback, score-normalisation" - true as a
#: description of what it measured, and misleading as a description of what the
#: reader almost certainly wanted. Nothing caught it because the output was
#: internally consistent; the header and the partition agreed with each other
#: and both disagreed with the increment.
#:
#: It then went stale a second time, one increment after that paragraph was
#: written: the thirty-document increment added ``query-likelihood`` and
#: ``term-dependence`` and left this naming ``click-models,deduplication``. The
#: README figures for that increment were produced with an explicit ``--added``
#: and are correct; only the no-flag default was wrong. The lesson is that a
#: comment is not a guard, so the question-only mode below does not depend on
#: this constant at all.
DEFAULT_ADDED = "query-likelihood,term-dependence"


def partition_gold(
    gold: Sequence[Dict],
    added: Sequence[str],
    new_questions: Sequence[str] = (),
) -> Tuple[List[Dict], List[Dict]]:
    """Split the gold set into questions that predate the increment and the rest.

    A question is "carried" only when *none* of its relevant documents is new
    and its id is not in ``new_questions``. A multi-document question naming
    one old and one new document has to count as new: on the old corpus part of
    its answer key does not exist, so its score there would be depressed by the
    missing document rather than by anything about retrieval.

    ``new_questions`` exists for increments that add questions and no
    documents. There the document partition is empty by definition, so without
    naming the questions directly every one of them would be scored as
    "carried" and the new-versus-carried comparison - the whole point of the
    script - would be impossible to make.
    """
    new_docs = set(added)
    new_ids = set(new_questions)
    carried, fresh = [], []
    for record in gold:
        is_new = record["id"] in new_ids or bool(new_docs & set(record["relevant"]))
        (fresh if is_new else carried).append(record)
    return carried, fresh


def build_engines(
    documents: Dict[str, str],
    routing: Sequence[Tuple[str, str, str]],
    ngram_n: int = 4,
    **config_kwargs
) -> Dict[str, RAGEngine]:
    """One engine per distinct index configuration the routing table needs.

    The n-gram index is index-time state, so a routing table that mentions it
    needs a second fitted engine rather than a query-time flag on the first.
    Building it unconditionally would roughly double the cost of every run that
    never asks for it.
    """
    engines = {"base": RAGEngine(documents, config=EngineConfig(**config_kwargs))}
    if any(entry[1] == "ngram" for entry in routing):
        engines["ngram"] = RAGEngine(
            documents, config=EngineConfig(ngram=ngram_n, **config_kwargs)
        )
    return engines


def discordance_flags(
    left_scores: Sequence[float], right_scores: Sequence[float]
) -> List[bool]:
    """Per-question "these two methods disagreed here", in question order.

    Kept separate from the aggregate counts because the paired test below needs
    to know *which* questions switched, and a W/L/T summary has already thrown
    that away.
    """
    return [a != b for a, b in zip(left_scores, right_scores)]


def cell(
    engines: Dict[str, RAGEngine],
    routing: Sequence[Tuple[str, str, str]],
    gold: Sequence[Dict],
    left: str,
    right: str,
    metric: str,
    k: int,
    bootstrap_resamples: int,
    permutations: int,
    seed: int,
) -> Tuple[Comparison, List[bool]]:
    """Score one (corpus, question-set) cell for the method pair under test."""
    ranked = significance_eval.rankings_by_method(engines, routing, gold, k)
    relevancies = [record["relevant"] for record in gold]
    scores = {
        method: [
            significance_eval.metric_value(metric, ranking, relevant)
            for ranking, relevant in zip(ranked[method], relevancies)
        ]
        for method in (left, right)
    }
    result = compare(
        scores[left],
        scores[right],
        bootstrap_resamples=bootstrap_resamples,
        permutations=permutations,
        seed=seed,
    )
    return result, discordance_flags(scores[left], scores[right])


def format_row(label: str, result: Comparison, confidence: float) -> str:
    rate = result.discordant / float(result.n) if result.n else 0.0
    low, high = clopper_pearson(result.discordant, result.n, confidence)
    return "{:<28} {:>3}/{:>3}/{:>3}  {:>7.1%}  [{:>5.1%}, {:>5.1%}]  {:>+7.3f}".format(
        label,
        result.wins,
        result.losses,
        result.ties,
        rate,
        low,
        high,
        result.delta,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    parser.add_argument(
        "--added",
        default=DEFAULT_ADDED,
        help=(
            "comma-separated document ids to treat as newly added; pass an "
            "empty string with --new-questions for a questions-only increment"
        ),
    )
    parser.add_argument(
        "--new-questions",
        default="",
        help=(
            "comma-separated gold question ids to treat as new regardless of "
            "which documents they name, e.g. h77,h78"
        ),
    )
    parser.add_argument(
        "--exclude",
        default="",
        help=(
            "document ids to drop from BOTH corpora, which rolls the repository "
            "back to an earlier state so a past increment can be re-measured "
            "with this instrument"
        ),
    )
    parser.add_argument(
        "--pair",
        default="hybrid2,bm25",
        help="the two methods to compare, e.g. 'hybrid3,hybrid2'",
    )
    parser.add_argument("--metric", default="ndcg@5")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--ngram-n", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    left, _, right = args.pair.partition(",")
    left, right = left.strip(), right.strip()
    known = {entry[0] for entry in significance_eval.METHOD_ROUTING}
    unknown = [name for name in (left, right) if name not in known]
    if unknown:
        raise SystemExit(
            "unknown methods: {} (known: {})".format(
                ",".join(unknown), ",".join(sorted(known))
            )
        )
    routing = [
        entry for entry in significance_eval.METHOD_ROUTING if entry[0] in (left, right)
    ]

    documents = read_corpus(args.corpus)
    added = [name.strip() for name in args.added.split(",") if name.strip()]
    excluded = [name.strip() for name in args.exclude.split(",") if name.strip()]
    missing = [name for name in added + excluded if name not in documents]
    if missing:
        raise SystemExit("not in the corpus: {}".format(",".join(sorted(set(missing)))))
    overlap = set(added) & set(excluded)
    if overlap:
        raise SystemExit(
            "a document cannot be both added and excluded: {}".format(
                ",".join(sorted(overlap))
            )
        )
    new_questions = [
        name.strip() for name in args.new_questions.split(",") if name.strip()
    ]
    if not added and not new_questions:
        raise SystemExit("--added needs at least one document id")

    gold = significance_eval.load_gold(args.gold)
    unknown_questions = sorted(set(new_questions) - {r["id"] for r in gold})
    if unknown_questions:
        raise SystemExit(
            "not in the gold set: {}".format(",".join(unknown_questions))
        )
    if excluded:
        # Rolling the corpus back means rolling the gold set back with it: a
        # question whose answer key names a removed document is unanswerable in
        # both arms and would score as a miss for reasons having nothing to do
        # with the comparison under test.
        dropped = set(excluded)
        gold = [
            record for record in gold if not dropped & set(record["relevant"])
        ]
        documents = {
            doc_id: text
            for doc_id, text in documents.items()
            if doc_id not in dropped
        }
        if not gold:
            raise SystemExit("--exclude removed every question")

    if len(added) >= len(documents):
        raise SystemExit("--added cannot name the whole corpus")

    before_documents = {
        doc_id: text for doc_id, text in documents.items() if doc_id not in added
    }
    carried, fresh = partition_gold(gold, added, new_questions)
    if not carried:
        raise SystemExit("no questions predate the added documents")

    config_kwargs = dict(
        target_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        rrf_k=args.rrf_k,
    )
    before = build_engines(
        before_documents, routing, ngram_n=args.ngram_n, **config_kwargs
    )
    # A questions-only increment leaves the corpus untouched, so the two arms are
    # the same engine. Reusing it makes the document-lever line an identity
    # check (it must read +0.0%, 0 became discordant, 0 stopped) rather than a
    # second fit that could only agree or reveal nondeterminism.
    after = (
        build_engines(documents, routing, ngram_n=args.ngram_n, **config_kwargs)
        if added
        else before
    )

    def measure(engines, subset):
        return cell(
            engines,
            routing,
            subset,
            left,
            right,
            args.metric,
            args.k,
            args.bootstrap_resamples,
            args.permutations,
            args.seed,
        )

    print(
        "{} vs {} on {}, k={}   added: {}".format(
            left, right, args.metric, args.k, ", ".join(sorted(added)) or "(none)"
        )
    )
    if new_questions:
        print("new questions named directly: {}".format(", ".join(new_questions)))
    print(
        "corpus {} -> {} documents   gold {} carried + {} new = {}".format(
            len(before_documents), len(documents), len(carried), len(fresh), len(gold)
        )
    )
    print()
    header = "{:<28} {:^13}  {:>7}  {:^16}  {:>7}".format(
        "corpus / questions", "W/L/T", "disc.", "exact 95% CI", "delta"
    )
    print(header)
    print("-" * len(header))

    before_carried, before_flags = measure(before, carried)
    after_carried, after_flags = measure(after, carried)
    print(
        format_row(
            "{} docs / carried {}".format(len(before_documents), len(carried)),
            before_carried,
            args.confidence,
        )
    )
    print(
        format_row(
            "{} docs / same {}".format(len(documents), len(carried)),
            after_carried,
            args.confidence,
        )
    )
    if fresh:
        print(
            format_row(
                "{} docs / new {} only".format(len(documents), len(fresh)),
                measure(after, fresh)[0],
                args.confidence,
            )
        )
    print(
        format_row(
            "{} docs / all {}".format(len(documents), len(gold)),
            measure(after, gold)[0],
            args.confidence,
        )
    )

    print()
    before_rate = before_carried.discordant / float(before_carried.n)
    after_rate = after_carried.discordant / float(after_carried.n)
    gained = sum(1 for was, now in zip(before_flags, after_flags) if now and not was)
    lost = sum(1 for was, now in zip(before_flags, after_flags) if was and not now)
    p_paired = mcnemar_exact(gained, lost)
    print(
        "document lever, question set held fixed: {:.1%} -> {:.1%} ({:+.1%}, "
        "{} -> {} disagreements)".format(
            before_rate,
            after_rate,
            after_rate - before_rate,
            before_carried.discordant,
            after_carried.discordant,
        )
    )
    print(
        "paired over the same questions: {} became discordant, {} stopped, "
        "{} unchanged   exact p={:.3f}".format(
            gained,
            lost,
            len(carried) - gained - lost,
            p_paired,
        )
    )
    # The two intervals in the table are marginal and overlap by construction;
    # the pairing is the whole design, so the paired p-value is the line that
    # actually adjudicates whether the corpus moved anything.
    print(
        "The two rows above share their questions, so nothing but the corpus "
        "differs between them - read the paired line, not the two intervals."
    )
    if fresh:
        print(
            "The new-questions row is only interpretable against the row above "
            "it, and at this size its interval spans most of the unit range."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
