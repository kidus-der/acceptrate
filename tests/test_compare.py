"""bench/compare.py — the P1 gate: two runs of the same config agree on median tok/s within 2%."""

from __future__ import annotations

import polars as pl
import pytest

from acceptrate.bench.compare import GATE_TOLERANCE, Comparison, compare_runs, per_generation_tok_s
from acceptrate.trace.schema import frame_from_rows, make_row


def _run(run_id: str, window_ms: float, n_gens: int = 5, n_windows: int = 4) -> pl.DataFrame:
    rows = [
        make_row(run_id, w, w + 1, 0, 0, 0.0, window_ms, "code", 0, 0, 0, f"p{g}", 0, window_ms)
        for g in range(n_gens)
        for w in range(n_windows)
    ]
    return frame_from_rows(rows)


def test_per_generation_tok_s_groups_by_prompt_and_rep() -> None:
    rates = per_generation_tok_s(_run("a", 50.0, n_gens=3))

    assert len(rates) == 3
    assert all(r == pytest.approx(20.0) for r in rates)


def test_identical_runs_pass_the_gate() -> None:
    result = compare_runs(_run("a", 50.0), _run("b", 50.0))

    assert isinstance(result, Comparison)
    assert result.rel_diff == 0.0
    assert result.passed


def test_runs_within_tolerance_pass() -> None:
    result = compare_runs(_run("a", 50.0), _run("b", 50.9))  # ~1.8% slower

    assert result.rel_diff < GATE_TOLERANCE
    assert result.passed


def test_runs_outside_tolerance_fail() -> None:
    result = compare_runs(_run("a", 50.0), _run("b", 52.0))  # ~3.8% slower

    assert result.rel_diff > GATE_TOLERANCE
    assert not result.passed


def test_dirty_windows_are_excluded_before_comparing() -> None:
    clean = _run("a", 50.0)
    dirty_rows = [
        make_row("b", w, w + 1, 0, 0, 0.0, 500.0, "code", 0, 3, 0, "p9", 0, 500.0) for w in range(4)
    ]
    polluted = pl.concat([_run("b", 50.0), frame_from_rows(dirty_rows)])

    result = compare_runs(clean, polluted)

    assert result.passed
    assert result.dirty_b == 4


def test_gate_tolerance_is_two_percent() -> None:
    assert GATE_TOLERANCE == 0.02
