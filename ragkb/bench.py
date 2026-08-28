"""Timing harness for the retrieval pipeline.

Every measurement in this project so far answers "did it rank better?". None of
them answer "what did that cost?". That gap matters here more than it would in
a repo with one retrieval path, because four separate stages have been bolted on
as *optional* - pseudo-relevance feedback, cross-encoder re-ranking, MMR, and
the character n-gram third list - and "optional" is only a real decision if both
sides of it are measured. A stage that buys +0.02 nDCG for 4x the query time is
a different recommendation than one that buys it for 1.1x, and the accuracy
tables cannot tell those two apart.

The harness is deliberately small and standard-library only, in keeping with the
rest of the package: :func:`measure` runs callables and reports the distribution
of their wall-clock times. Two choices are worth stating because they change how
the numbers should be read:

* **Samples are per call, not per pass.** Timing a whole query set and dividing
  by its size would report a mean wearing a percentile's clothes: it cannot
  show that one slow query dominates. Pooling every individual call means p95
  is a real tail-latency number over queries.
* **Median leads, not mean.** These are wall-clock timings on a shared machine,
  so the distribution has a long right tail made of scheduler noise rather than
  of anything about retrieval. The median is the stable summary; p95 is
  reported next to it precisely so the tail stays visible instead of being
  averaged into the headline.

Absolute milliseconds here are not portable - they describe this machine, this
corpus size, this Python. The ratios between rows are the durable finding, which
is why :func:`format_timings` leads with a multiplier column against a baseline
row rather than with the raw numbers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Timing:
    """The wall-clock distribution of one labelled operation, in milliseconds."""

    label: str
    samples: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float

    def ratio_to(self, other: "Timing") -> float:
        """This timing's median as a multiple of ``other``'s.

        The portable half of the measurement: absolute milliseconds move with
        the machine, but "the re-ranker costs 3x a plain query" survives being
        run somewhere else.
        """
        if other.median_ms <= 0.0:
            return float("inf")
        return self.median_ms / other.median_ms


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of ``values`` for ``q`` in [0, 1].

    Nearest-rank rather than interpolated on purpose: with a few hundred
    samples, interpolating between two neighbours invents a latency no call
    actually took. Every value this returns was really observed.
    """
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def measure(
    label: str,
    calls: Iterable[Callable[[], object]],
    repeats: int = 3,
    warmup: int = 1,
) -> Timing:
    """Time each callable in ``calls``, ``repeats`` times, and pool the samples.

    ``warmup`` passes run first and are discarded. They are not a formality:
    the first call to a given code path pays for import-time work, dict
    resizing and the first fill of the interpreter's caches, and including it
    would attribute a one-off cost to every query.

    Raises ``ValueError`` on an empty call list rather than returning a zeroed
    :class:`Timing`, because a silently empty benchmark reports "instant".
    """
    materialised = list(calls)
    if not materialised:
        raise ValueError("nothing to measure: {}".format(label))
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must not be negative")

    for _ in range(warmup):
        for call in materialised:
            call()

    samples: List[float] = []
    for _ in range(repeats):
        for call in materialised:
            start = time.perf_counter()
            call()
            samples.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return Timing(
        label=label,
        samples=len(samples),
        median_ms=median,
        p95_ms=percentile(ordered, 0.95),
        mean_ms=sum(ordered) / len(ordered),
        min_ms=ordered[0],
    )


def measure_one(
    label: str, call: Callable[[], object], repeats: int = 3, warmup: int = 1
) -> Timing:
    """:func:`measure` for a single operation, such as fitting an index."""
    return measure(label, [call], repeats=repeats, warmup=warmup)


def format_timings(
    timings: Sequence[Timing], baseline: Optional[Timing] = None, label: str = "stage"
) -> str:
    """Render timings as a fixed-width table, with a ratio column when useful.

    ``baseline`` adds an ``xbase`` column; without it the table is raw
    milliseconds, which is the right shape for index-build costs where there
    is no natural denominator.
    """
    if not timings:
        return ""
    width = max(len(timing.label) for timing in timings) + 2
    width = max(width, len(label) + 2)
    columns = ["median_ms", "p95_ms", "mean_ms", "n"]
    header = "{:<{width}}".format(label, width=width) + "".join(
        "{:>12}".format(name) for name in columns
    )
    if baseline is not None:
        header += "{:>10}".format("xbase")
    lines = [header, "-" * len(header)]
    for timing in timings:
        row = "{:<{width}}".format(timing.label, width=width) + "".join(
            [
                "{:>12.2f}".format(timing.median_ms),
                "{:>12.2f}".format(timing.p95_ms),
                "{:>12.2f}".format(timing.mean_ms),
                "{:>12d}".format(timing.samples),
            ]
        )
        if baseline is not None:
            row += "{:>10.2f}".format(timing.ratio_to(baseline))
        lines.append(row)
    return "\n".join(lines)
