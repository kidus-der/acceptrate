"""verify/lossless.py — greedy equivalence: exact token match, no tolerance."""

from __future__ import annotations

import pytest

from acceptrate.trace.schema import make_row
from acceptrate.verify.lossless import (
    LosslessReport,
    compare_generation,
    first_divergence,
    summarize,
)


def test_first_divergence_is_none_for_identical_sequences() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None


def test_first_divergence_is_the_first_differing_index() -> None:
    assert first_divergence([1, 2, 3, 4], [1, 2, 9, 4]) == 2


def test_length_mismatch_diverges_at_the_shorter_length() -> None:
    assert first_divergence([1, 2, 3], [1, 2]) == 2
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def _rows(accepts: list[int], k: int) -> tuple:
    return tuple(
        make_row("r", i, i, k, n, 5.0, 50.0, "code", 0, 0, 0, "p", 0, 60.0)
        for i, n in enumerate(accepts)
    )


def test_compare_generation_reports_match_and_alpha() -> None:
    rows = _rows([4, 2, 0], 4)

    report = compare_generation("code-001", k=4, plain=[1, 2, 3], spec=[1, 2, 3], rows=rows)

    assert isinstance(report, LosslessReport)
    assert report.matched
    assert report.first_divergence is None
    assert report.n_tokens == 3
    assert report.windows == 3
    assert report.alpha == pytest.approx(6 / 12)


def test_compare_generation_reports_the_divergence_and_both_tokens() -> None:
    rows = _rows([1], 4)

    report = compare_generation("code-001", k=4, plain=[1, 2, 3], spec=[1, 7, 3], rows=rows)

    assert not report.matched
    assert report.first_divergence == 1
    assert report.plain_token == 2
    assert report.spec_token == 7


def test_alpha_is_zero_with_no_windows() -> None:
    report = compare_generation("p", k=4, plain=[1], spec=[1], rows=())

    assert report.alpha == 0.0


def test_summarize_counts_matches() -> None:
    reports = [
        compare_generation("a", 4, [1], [1], ()),
        compare_generation("b", 4, [1], [2], ()),
        compare_generation("c", 4, [1, 2], [1, 2], ()),
    ]

    passed, total = summarize(reports)

    assert (passed, total) == (2, 3)
