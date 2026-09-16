"""bench/stats.py — median and IQR, never the mean; per-generation throughput."""

from __future__ import annotations

import pytest

from acceptrate.bench.stats import MedianIQR, median_iqr, throughput_tok_s
from acceptrate.trace.schema import make_row


def test_median_iqr_of_odd_sample() -> None:
    result = median_iqr([5.0, 1.0, 3.0, 2.0, 4.0])

    assert result == MedianIQR(median=3.0, q1=2.0, q3=4.0, n=5)


def test_median_iqr_is_robust_to_one_throttle_event() -> None:
    clean = [20.0] * 9
    with_spike = [*clean, 2.0]

    assert median_iqr(clean).median == median_iqr(with_spike).median == 20.0


def test_median_iqr_rejects_empty() -> None:
    with pytest.raises(ValueError):
        median_iqr([])


def _row(window_ms: float, n_accepted: int = 0) -> tuple:
    return make_row("r", 0, 1, 0, n_accepted, 0.0, window_ms, "code", 0, 0, 0, "p", 0, window_ms)


def test_throughput_counts_one_emitted_token_per_window_plus_accepted() -> None:
    rows = [_row(50.0), _row(50.0), _row(50.0), _row(50.0)]  # 4 windows, 200 ms, K=0

    assert throughput_tok_s(rows) == pytest.approx(20.0)


def test_throughput_credits_accepted_draft_tokens() -> None:
    rows = [_row(100.0, n_accepted=3)]  # one window: 3 accepted + 1 target token in 100 ms

    assert throughput_tok_s(rows) == pytest.approx(40.0)


def test_throughput_of_no_windows_is_zero() -> None:
    assert throughput_tok_s([]) == 0.0
