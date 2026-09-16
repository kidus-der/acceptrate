"""bench/p5.py — the P5 evaluation plan: adaptive vs every fixed K on the held-out mixed workload.

Pure planning + scoring; the composition root builds the generators. The
held-out split was fixed before any measurement existed (bench/workloads).
"""

from __future__ import annotations

from acceptrate.bench.p5 import P5Verdict, p5_arms, score_p5
from acceptrate.bench.runner import ArmSummary
from acceptrate.bench.stats import MedianIQR


def _arm(median: float, n: int = 20) -> ArmSummary:
    return ArmSummary(
        generations=n,
        windows=n * 30,
        dirty_windows=0,
        tok_s=MedianIQR(median, median - 1, median + 1, n),
    )


def test_arms_are_adaptive_plus_every_fixed_k() -> None:
    arms = p5_arms((1, 2, 4, 8), "org/target", "org/draft", 200, ["p1"], reps=1, warmup=2)

    assert [a.name for a in arms] == ["adaptive", "spec-k1", "spec-k2", "spec-k4", "spec-k8"]
    assert arms[0].k is None and arms[0].config["arm"] == "adaptive"
    assert len({a.run_id for a in arms}) == 5


def test_score_passes_when_adaptive_beats_best_fixed_by_five_percent() -> None:
    summaries = {"adaptive": _arm(31.6), "spec-k2": _arm(30.0), "spec-k4": _arm(29.0)}

    verdict = score_p5(summaries)

    assert isinstance(verdict, P5Verdict)
    assert verdict.best_fixed_name == "spec-k2"
    assert verdict.gain == 31.6 / 30.0 - 1
    assert verdict.passed


def test_score_fails_below_five_percent() -> None:
    summaries = {"adaptive": _arm(30.9), "spec-k2": _arm(30.0)}

    verdict = score_p5(summaries)

    assert not verdict.passed
    assert verdict.gain < 0.05


def test_score_requires_an_adaptive_arm_and_at_least_one_fixed() -> None:
    import pytest

    with pytest.raises(ValueError):
        score_p5({"spec-k2": _arm(30.0)})
    with pytest.raises(ValueError):
        score_p5({"adaptive": _arm(30.0)})
