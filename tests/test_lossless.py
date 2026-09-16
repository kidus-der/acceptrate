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


def _floor():
    from acceptrate.verify.noise_floor import NoiseFloor

    return NoiseFloor("M4", "0.32.2", 400, 0.03, 0.075, 0.1, 0)


def test_classify_uses_the_noise_floor_for_near_ties() -> None:
    from acceptrate.verify.lossless import Verdict, classify

    identical = compare_generation("a", 4, [1, 2], [1, 2], ())
    near = compare_generation("b", 4, [1, 2], [1, 3], (), margin=0.0)
    far = compare_generation("c", 4, [1, 2], [1, 3], (), margin=0.5)

    assert classify(identical, _floor()) is Verdict.IDENTICAL
    assert classify(near, _floor()) is Verdict.NEAR_TIE
    assert classify(far, _floor()) is Verdict.DIVERGENT


def test_classify_treats_a_divergence_without_margin_as_divergent() -> None:
    from acceptrate.verify.lossless import Verdict, classify

    report = compare_generation("b", 4, [1, 2], [1, 3], ())

    assert classify(report, _floor()) is Verdict.DIVERGENT


def test_sequential_margin_at_replays_plain_decoding_to_the_index() -> None:
    from acceptrate.verify.lossless import sequential_margin_at
    from tests.fakes import FakeBackend

    backend = FakeBackend(32)
    plain = [7, 17, 23]  # the fake's logits are one-hot, so the margin is exactly 1.0

    margin = sequential_margin_at(backend, [3], plain, index=2)

    assert margin == 1.0
    assert backend.position == 1 + 2


def test_gate_passes_when_every_divergence_is_a_near_tie() -> None:
    from acceptrate.verify.lossless import Verdict, gate_verdicts

    reports = [
        compare_generation("a", 4, [1, 2], [1, 2], ()),
        compare_generation("b", 4, [1, 2], [1, 3], (), margin=0.02),
    ]

    passed, counts = gate_verdicts(reports, _floor())

    assert passed
    assert counts == {Verdict.IDENTICAL: 1, Verdict.NEAR_TIE: 1, Verdict.DIVERGENT: 0}


def test_gate_fails_on_a_real_divergence() -> None:
    from acceptrate.verify.lossless import Verdict, gate_verdicts

    reports = [compare_generation("b", 4, [1, 2], [1, 3], (), margin=0.4)]

    passed, counts = gate_verdicts(reports, _floor())

    assert not passed
    assert counts[Verdict.DIVERGENT] == 1
