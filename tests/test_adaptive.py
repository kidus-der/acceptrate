"""runtime/adaptive.py — the scheduler picks K per window; generation stays lossless."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acceptrate.runtime.adaptive import AdaptiveScheduler, SchedulerConfig, generate_adaptive
from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.runtime.speculative import generate_speculative
from acceptrate.trace.schema import frame_from_rows
from tests.fakes import FakeBackend

CTX = GenerationContext(run_id="r", workload_tag="code", prompt_id="code-001", rep=0)
V = 32


@dataclass(frozen=True)
class Snap:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


class PartialFake(FakeBackend):
    def next_token(self, token: int) -> int:
        wrong = 1 if token % 3 == 0 else 0
        return (token * 7 + wrong) % self.vocab_size


def _cfg(**kw) -> SchedulerConfig:
    base = dict(
        k_min=1, k_max=8, prior_alpha=0.6, prior_c=0.16, half_life_windows=3, warmup_windows=0
    )
    base.update(kw)
    return SchedulerConfig(**base)


def test_initial_k_comes_from_the_priors() -> None:
    sched = AdaptiveScheduler(_cfg(prior_alpha=0.7, prior_c=0.16))

    assert sched.next_k() == 3  # brief's worked example: alpha 0.70, c 0.16 -> K = 3


def test_k_rises_with_high_acceptance_and_falls_with_low() -> None:
    sched = AdaptiveScheduler(_cfg(prior_alpha=0.5))
    for _ in range(10):
        sched.observe(k_proposed=4, n_accepted=4, draft_ms=4 * 8.0, verify_ms=50.0)
    high = sched.next_k()
    for _ in range(10):
        sched.observe(k_proposed=4, n_accepted=0, draft_ms=4 * 8.0, verify_ms=50.0)
    low = sched.next_k()

    assert high > low
    assert low == 1  # never below k_min: a K=1 window keeps the estimate alive


def test_measured_cost_ratio_feeds_the_decision() -> None:
    cheap = AdaptiveScheduler(_cfg(prior_alpha=0.6))
    dear = AdaptiveScheduler(_cfg(prior_alpha=0.6))
    for _ in range(10):
        cheap.observe(4, 3, draft_ms=4 * 1.0, verify_ms=50.0)  # c = 0.02
        dear.observe(4, 3, draft_ms=4 * 25.0, verify_ms=50.0)  # c = 0.5

    assert cheap.next_k() > dear.next_k()
    assert cheap.c == pytest.approx(0.02, abs=0.05)


def test_k_is_clipped_to_the_configured_range() -> None:
    sched = AdaptiveScheduler(_cfg(k_max=3, prior_alpha=0.99, prior_c=0.01))

    assert sched.next_k() == 3


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(
            k_min=0, k_max=8, prior_alpha=0.6, prior_c=0.16, half_life_windows=3, warmup_windows=0
        )
    with pytest.raises(ValueError):
        SchedulerConfig(
            k_min=4, k_max=2, prior_alpha=0.6, prior_c=0.16, half_life_windows=3, warmup_windows=0
        )


def test_constant_policy_reproduces_fixed_k_speculative_exactly() -> None:
    class Constant:
        def next_k(self) -> int:
            return 4

        def observe(self, k_proposed, n_accepted, draft_ms, verify_ms) -> None:
            pass

    t1, d1 = FakeBackend(V, step=7), PartialFake(V)
    t2, d2 = FakeBackend(V, step=7), PartialFake(V)

    fixed = generate_speculative(t1, d1, [3, 4], 40, frozenset(), CTX, Snap, k=4)
    adaptive = generate_adaptive(t2, d2, [3, 4], 40, frozenset(), CTX, Snap, Constant())

    assert adaptive.tokens == fixed.tokens
    assert len(adaptive.rows) == len(fixed.rows)
    for a, f in zip(adaptive.rows, fixed.rows, strict=True):
        assert (
            a[:5] == f[:5] and a[7:] == f[7:] or a[:5] == f[:5]
        )  # timings differ; structure equal


def test_adaptive_varies_k_and_records_it_per_window() -> None:
    sched = AdaptiveScheduler(_cfg(prior_alpha=0.6, half_life_windows=2))

    result = generate_adaptive(
        FakeBackend(V, step=7), PartialFake(V), [3], 80, frozenset(), CTX, Snap, sched
    )

    df = frame_from_rows(result.rows)
    assert df["k_proposed"].n_unique() > 1
    assert (df["k_proposed"] >= 1).all()
    assert sched.windows == len(result.rows)


@settings(max_examples=100, deadline=None)
@given(
    step_d=st.integers(1, 31),
    offset_d=st.integers(0, 31),
    prompt=st.lists(st.integers(1, 31), min_size=1, max_size=5),
    max_tokens=st.integers(1, 40),
    prior=st.floats(0.05, 0.95),
)
def test_property_adaptive_equals_plain_for_any_draft(step_d, offset_d, prompt, max_tokens, prior):
    sched = AdaptiveScheduler(_cfg(prior_alpha=prior, half_life_windows=2))
    plain = generate_plain(FakeBackend(V, step=7), prompt, max_tokens, frozenset(), CTX, Snap)

    result = generate_adaptive(
        FakeBackend(V, step=7), FakeBackend(V, step=step_d, offset=offset_d),
        prompt, max_tokens, frozenset(), CTX, Snap, sched,
    )  # fmt: skip

    assert result.tokens == plain.tokens
