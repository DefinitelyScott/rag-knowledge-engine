"""Cross-validated selection: what a sweep table costs you by picking its own winner.

Every sweep in this project - `rrf_k` in `evals/tune.py`, the fusion weights and
the MMR lambda in `evals/evaluate.py`, expansion alpha, re-ranking alpha - scores
a handful of configurations on the same 78 gold questions and invites the reader
to take the top row. `ragkb/significance.py` already asks whether the *gap*
between two named systems is real. This module asks the question one level up,
which nothing else in the repo covers: **the top row was chosen by looking at the
answers**, so its reported score is not an estimate of how that configuration
will do on the next 78 questions. It is the maximum of several noisy estimates,
and the maximum of noise is biased upward whether or not any configuration is
genuinely better.

The procedure is standard nested selection, done here with k-fold splits:

1. Partition the questions into ``folds`` groups (seeded, so a run reproduces).
2. For each fold, pick the best configuration on the *other* folds, then score
   that choice on the held-out fold - questions the choice never saw.
3. Average those held-out scores over the whole set.

The three numbers that come out are the point:

* ``in_sample_best`` - the best row of the sweep table, the number that gets
  quoted.
* ``cv_selected`` - what the *procedure* "sweep, then take the winner" actually
  earns on unseen questions.
* ``optimism`` - the difference. This is the part of the headline that came from
  the selection rather than from the configuration.

Two supporting read-outs matter as much as the optimism figure:

* ``tuning_gain`` compares ``cv_selected`` against simply shipping the default
  and never sweeping at all. A sweep with positive optimism and *negative*
  tuning gain is one that would have been better left unrun - and that outcome
  is common on a set this size, which is why it is measured rather than assumed.
* ``winners`` lists which configuration won each fold. When the folds disagree,
  the sweep is not identifying a better setting, it is tracking which questions
  happened to be in the training half. That instability is visible directly,
  without reference to any p-value.

The scope is honest about itself: this measures selection bias over the *given*
configurations and questions, with the folds resampled from one small gold set,
so ``optimism`` is itself a noisy estimate. It cannot rescue a sweep from a gold
set too small to distinguish anything - see
:func:`ragkb.significance.smallest_achievable_p` for that limit - and it makes
no attempt to. What it does is put a number on a cost the tables previously did
not report at all.

Standard library only, and every random choice runs through an explicitly seeded
:class:`random.Random`, so a printed table can be reproduced exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def k_fold(n: int, folds: int = 5, seed: int = 0) -> List[List[int]]:
    """Partition ``range(n)`` into ``folds`` index groups after a seeded shuffle.

    A partition, not a sample: every index appears in exactly one group, so the
    held-out scores can be averaged over the whole gold set without weighting.
    Group sizes differ by at most one.

    The shuffle is not decoration. ``evals/gold.jsonl`` is written in blocks -
    the easy questions first, then the hard ones - so contiguous folds would put
    an entire difficulty slice on one side of the split and measure the
    difference between slices instead of the effect of selection.
    """
    if n <= 0:
        raise ValueError("need at least one item to split")
    if folds < 2:
        raise ValueError("need at least 2 folds to hold anything out")
    if folds > n:
        raise ValueError(
            "cannot make {} folds from {} items".format(folds, n)
        )
    order = list(range(n))
    random.Random(seed).shuffle(order)
    groups: List[List[int]] = [[] for _ in range(folds)]
    for position, index in enumerate(order):
        groups[position % folds].append(index)
    for group in groups:
        group.sort()
    return groups


def _check_scores(scores: Mapping[str, Sequence[float]]) -> Tuple[List[str], int]:
    """Validate the config -> per-query score table and return its shape."""
    names = list(scores)
    if not names:
        raise ValueError("no configurations to select between")
    lengths = {len(scores[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError(
            "every configuration must be scored on the same questions: "
            "got lengths {}".format(sorted(lengths))
        )
    n = lengths.pop()
    if n == 0:
        raise ValueError("cannot select over empty score vectors")
    return names, n


def _argbest(
    scores: Mapping[str, Sequence[float]], names: Sequence[str], indices: Sequence[int]
) -> str:
    """Name of the configuration with the highest mean over ``indices``.

    Ties break on the order ``names`` came in, which makes the whole procedure a
    deterministic function of its inputs and the seed. That tie-break is not
    neutral - it favours whichever configuration the caller listed first, and on
    a small fold exact ties are common - so callers that care should list the
    shipped default first and read a tie as "no reason to move".
    """
    best_name = names[0]
    best_score = mean([scores[best_name][i] for i in indices])
    for name in names[1:]:
        value = mean([scores[name][i] for i in indices])
        if value > best_score:
            best_name, best_score = name, value
    return best_name


@dataclass(frozen=True)
class SelectionResult:
    """One sweep's worth of selection accounting."""

    n: int
    folds: int
    seed: int
    #: Configuration names in the order they were offered.
    configs: Tuple[str, ...]
    #: Mean over the whole set, per configuration - the sweep table itself.
    in_sample: Dict[str, float]
    #: The configuration each fold's training half chose, in fold order.
    winners: Tuple[str, ...]
    #: Mean of the held-out scores of those per-fold choices.
    cv_selected: float
    #: Name of the configuration the full-set sweep would report.
    in_sample_argbest: str
    #: Whole-set mean of the baseline configuration, when one was named.
    default_name: Optional[str]
    default_score: Optional[float]

    @property
    def in_sample_best(self) -> float:
        """The number a sweep table prints for its top row."""
        return self.in_sample[self.in_sample_argbest]

    @property
    def optimism(self) -> float:
        """``in_sample_best - cv_selected``: the part that came from selecting.

        Never negative in expectation, but a single run on a small set can land
        below zero - which is itself informative, since it means the folds were
        choosing something other than the full-set winner often enough to beat
        it by luck.
        """
        return self.in_sample_best - self.cv_selected

    @property
    def tuning_gain(self) -> Optional[float]:
        """``cv_selected - default_score``: was sweeping worth it at all?

        ``None`` when no default was named. Negative means the honest estimate of
        the tuning procedure is *worse* than shipping the constant, which is the
        result that most deserves to be printed and is the easiest to omit.
        """
        if self.default_score is None:
            return None
        return self.cv_selected - self.default_score

    @property
    def distinct_winners(self) -> int:
        """How many different configurations won at least one fold.

        1 means the sweep found the same answer however the questions were
        split. Anything higher means the winner depends on the sample, and the
        sweep table's ordering should not be read as a ranking.
        """
        return len(set(self.winners))

    @property
    def stable(self) -> bool:
        """True when every fold picked the same configuration."""
        return self.distinct_winners == 1

    def describe(self) -> str:
        parts = [
            "best={} in-sample={:.3f} held-out={:.3f} optimism={:+.3f}".format(
                self.in_sample_argbest,
                self.in_sample_best,
                self.cv_selected,
                self.optimism,
            )
        ]
        if self.tuning_gain is not None:
            parts.append(
                "vs {}={:+.3f}".format(self.default_name, self.tuning_gain)
            )
        parts.append(
            "winners={}/{}".format(self.distinct_winners, self.folds)
        )
        return " ".join(parts)


def cross_validated_selection(
    scores: Mapping[str, Sequence[float]],
    folds: int = 5,
    seed: int = 0,
    default: Optional[str] = None,
) -> SelectionResult:
    """Score the procedure "sweep these configurations, then take the winner".

    ``scores`` maps a configuration name to its per-query metric values, one per
    gold question, in a consistent question order across configurations - the
    same paired layout :mod:`ragkb.significance` consumes. Reuse of one
    retrieval pass across every metric is the caller's job; this function only
    does the accounting.

    ``default`` names the configuration that ships. When given, the result also
    reports what never sweeping at all would have earned, which is the only
    baseline that makes the sweep's held-out score interpretable.
    """
    names, n = _check_scores(scores)
    if default is not None and default not in scores:
        raise ValueError("default {!r} is not one of the configurations".format(default))

    groups = k_fold(n, folds=folds, seed=seed)
    winners: List[str] = []
    held_out: List[float] = []
    for group in groups:
        train = [i for i in range(n) if i not in set(group)]
        # With folds == n the training half is n - 1 questions and never empty;
        # k_fold already refuses folds > n, so this cannot select on nothing.
        winner = _argbest(scores, names, train)
        winners.append(winner)
        # Extend rather than average-of-averages: folds can differ in size by
        # one, and averaging the fold means would silently over-weight the
        # smaller folds.
        held_out.extend(scores[winner][i] for i in group)

    in_sample = {name: mean(scores[name]) for name in names}
    return SelectionResult(
        n=n,
        folds=folds,
        seed=seed,
        configs=tuple(names),
        in_sample=in_sample,
        winners=tuple(winners),
        cv_selected=mean(held_out),
        in_sample_argbest=_argbest(scores, names, list(range(n))),
        default_name=default,
        default_score=in_sample[default] if default is not None else None,
    )


def repeated_selection(
    scores: Mapping[str, Sequence[float]],
    folds: int = 5,
    repeats: int = 20,
    seed: int = 0,
    default: Optional[str] = None,
) -> List[SelectionResult]:
    """Run :func:`cross_validated_selection` over ``repeats`` different splits.

    One k-fold split is itself a sample: on 78 questions the held-out estimate
    moves visibly with the seed, and reporting a single split's optimism would
    repeat in miniature the error this module exists to expose. Seeds run
    ``seed, seed + 1, ...`` so a reported spread is reproducible.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return [
        cross_validated_selection(
            scores, folds=folds, seed=seed + offset, default=default
        )
        for offset in range(repeats)
    ]


def summarize(results: Sequence[SelectionResult]) -> Dict[str, float]:
    """Mean and spread of the quantities that vary with the split.

    ``in_sample_best`` is deliberately included even though it is constant
    across repeats: it is the anchor the other numbers are read against.
    """
    if not results:
        raise ValueError("nothing to summarize")
    selected = sorted(result.cv_selected for result in results)
    optimism = [result.optimism for result in results]
    gains = [
        result.tuning_gain for result in results if result.tuning_gain is not None
    ]
    summary = {
        "repeats": float(len(results)),
        "in_sample_best": results[0].in_sample_best,
        "cv_selected": mean(selected),
        "cv_selected_min": selected[0],
        "cv_selected_max": selected[-1],
        "optimism": mean(optimism),
        "stable_share": mean([1.0 if r.stable else 0.0 for r in results]),
    }
    if gains:
        summary["tuning_gain"] = mean(gains)
        summary["tuning_gain_negative_share"] = mean(
            [1.0 if value < 0.0 else 0.0 for value in gains]
        )
    return summary
