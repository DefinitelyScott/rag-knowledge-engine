"""Tests for the timing harness.

Timing tests have an obvious trap: assert on a duration and the suite fails on
a loaded machine for reasons that have nothing to do with the code. So nothing
here asserts how long anything took. What is asserted is the arithmetic around
the clock - sample counts, percentile selection, ordering invariants that hold
for any distribution, and the failure modes - because those are deterministic
and are where a benchmark actually goes wrong.
"""

from __future__ import annotations

import pytest

from ragkb.bench import Timing, format_timings, measure, measure_one, percentile


def test_percentile_returns_an_observed_value():
    values = [5.0, 1.0, 4.0, 2.0, 3.0]
    # Nearest-rank never interpolates, so every result is one of the inputs.
    for q in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0):
        assert percentile(values, q) in values


def test_percentile_bounds():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 1.0) == 40.0
    assert percentile(values, 0.5) == 20.0
    assert percentile([], 0.5) == 0.0


def test_percentile_rejects_out_of_range_q():
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def test_measure_pools_every_call_across_repeats():
    calls = [lambda: None for _ in range(4)]
    timing = measure("noop", calls, repeats=3, warmup=1)
    assert timing.samples == 12
    assert timing.label == "noop"


def test_measure_runs_warmup_but_does_not_count_it():
    runs = []
    timing = measure("counted", [lambda: runs.append(1)], repeats=2, warmup=3)
    assert timing.samples == 2
    assert len(runs) == 5  # 3 warmup + 2 timed


def test_measure_orders_its_summary_statistics():
    timing = measure("noop", [lambda: sum(range(50)) for _ in range(5)], repeats=3)
    assert timing.min_ms <= timing.median_ms <= timing.p95_ms
    assert timing.min_ms <= timing.mean_ms
    assert timing.median_ms >= 0.0


def test_measure_rejects_an_empty_benchmark():
    # A silently empty benchmark would report "instant", which is worse than
    # no measurement at all.
    with pytest.raises(ValueError):
        measure("nothing", [])


def test_measure_validates_repeats_and_warmup():
    with pytest.raises(ValueError):
        measure("noop", [lambda: None], repeats=0)
    with pytest.raises(ValueError):
        measure("noop", [lambda: None], warmup=-1)


def test_measure_one_produces_a_single_sample_per_repeat():
    timing = measure_one("fit", lambda: [x * x for x in range(100)], repeats=4)
    assert timing.samples == 4


def test_ratio_to_is_a_median_multiple():
    fast = Timing("fast", 1, 2.0, 3.0, 2.0, 2.0)
    slow = Timing("slow", 1, 6.0, 7.0, 6.0, 6.0)
    assert slow.ratio_to(fast) == pytest.approx(3.0)
    assert fast.ratio_to(fast) == pytest.approx(1.0)


def test_ratio_to_a_zero_baseline_is_infinite_not_a_crash():
    zero = Timing("zero", 1, 0.0, 0.0, 0.0, 0.0)
    assert Timing("x", 1, 1.0, 1.0, 1.0, 1.0).ratio_to(zero) == float("inf")


def test_format_timings_adds_a_ratio_column_only_with_a_baseline():
    base = Timing("base", 10, 1.0, 2.0, 1.1, 0.9)
    other = Timing("other", 10, 2.0, 3.0, 2.1, 1.9)
    plain = format_timings([base, other])
    assert "xbase" not in plain
    assert "median_ms" in plain
    ratioed = format_timings([base, other], baseline=base)
    assert "xbase" in ratioed
    assert "2.00" in ratioed


def test_format_timings_handles_no_rows():
    assert format_timings([]) == ""
