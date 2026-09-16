"""runtime/streaming.generate_adaptive_streaming — per-window K in the streaming path.

Must equal runtime/adaptive.generate_adaptive token-for-token and row-for-row
(timings masked) so the served runtime is the same runtime the bench measured.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from acceptrate.runtime.adaptive import AdaptiveScheduler, SchedulerConfig, generate_adaptive
from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.runtime.streaming import generate_adaptive_streaming
from tests.fakes import FakeBackend

CTX = GenerationContext(run_id="r", workload_tag="code", prompt_id="code-001", rep=0)
V = 32
_TIMING = {5, 6, 13}


@dataclass(frozen=True)
class Snap:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


class Cycling:
    """Deterministic K sequence so batch and streaming can be compared exactly.

    The real AdaptiveScheduler feeds its cost estimate from measured timings,
    which on fakes are microseconds of Python noise, so K may legitimately
    differ between two runs. Equality is asserted with this instead.
    """

    def __init__(self, pattern: tuple[int, ...]) -> None:
        self.pattern = pattern
        self.i = 0

    def next_k(self) -> int:
        k = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return k

    def observe(self, k_proposed, n_accepted, draft_ms, verify_ms) -> None:
        return None


def _cfg(prior: float) -> SchedulerConfig:
    return SchedulerConfig(
        k_min=1, k_max=6, prior_alpha=prior, prior_c=0.16, half_life_windows=2, warmup_windows=0
    )


def _mask(row):
    return tuple(v for i, v in enumerate(row) if i not in _TIMING)


@settings(max_examples=60, deadline=None)
@given(
    step_d=st.integers(1, 31),
    offset_d=st.integers(0, 31),
    prompt=st.lists(st.integers(1, 31), min_size=1, max_size=5),
    max_tokens=st.integers(1, 40),
    pattern=st.lists(st.integers(1, 6), min_size=1, max_size=4),
)
def test_streaming_adaptive_equals_batch_adaptive(step_d, offset_d, prompt, max_tokens, pattern):
    batch = generate_adaptive(
        FakeBackend(V, step=7),
        FakeBackend(V, step=step_d, offset=offset_d),
        prompt,
        max_tokens,
        frozenset(),
        CTX,
        Snap,
        Cycling(tuple(pattern)),
    )

    events = list(
        generate_adaptive_streaming(
            FakeBackend(V, step=7),
            FakeBackend(V, step=step_d, offset=offset_d),
            prompt,
            max_tokens,
            frozenset(),
            CTX,
            Snap,
            Cycling(tuple(pattern)),
        )
    )

    tokens = tuple(t for toks, _ in events for t in toks)
    rows = [row for _, row in events if row is not None]
    assert tokens == batch.tokens
    assert [_mask(r) for r in rows] == [_mask(r) for r in batch.rows]
    assert events[0][1] is None  # prefill token carries no row


def test_first_event_is_the_prefill_token() -> None:
    events = list(
        generate_adaptive_streaming(
            FakeBackend(V, step=7),
            FakeBackend(V, step=7),
            [3],
            10,
            frozenset(),
            CTX,
            Snap,
            AdaptiveScheduler(_cfg(0.6)),
        )
    )

    assert events[0] == (((3 * 7) % V,), None)


def test_real_scheduler_is_lossless_and_observes_every_window() -> None:
    plain = generate_plain(FakeBackend(V, step=7), [3], 80, frozenset(), CTX, Snap)
    sched = AdaptiveScheduler(_cfg(0.9))

    events = list(
        generate_adaptive_streaming(
            FakeBackend(V, step=7), FakeBackend(V, step=3), [3], 80, frozenset(), CTX, Snap, sched
        )
    )

    tokens = tuple(t for toks, _ in events for t in toks)
    assert tokens == plain.tokens
    assert sched.windows == sum(1 for _, row in events if row is not None)
