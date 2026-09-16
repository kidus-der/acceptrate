"""The scheduler: picks draft depth K for the next window, live.

Layer 1 (model/speedup.best_k) turns an alpha and a cost ratio into K.
Layer 2 (model/estimator) supplies alpha from recent windows. The cost
ratio c is measured the same way: an EWMA of (draft_ms / K) / verify_ms per
window, never assumed from parameter counts. K never drops below k_min, so
every window still yields an observation and the estimate cannot freeze.

generate_adaptive mirrors runtime/speculative.py exactly except that K is
asked of the scheduler before each window and the window is reported back
to it afterwards. Same cache discipline, same row contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from acceptrate.backend.protocol import Backend
from acceptrate.model.estimator import AcceptanceEstimator
from acceptrate.model.speedup import best_k
from acceptrate.runtime.engine import MS, GenerationContext, GenerationResult, GuardReader
from acceptrate.trace.schema import WindowRow


@dataclass(frozen=True)
class SchedulerConfig:
    k_min: int
    k_max: int
    prior_alpha: float
    prior_c: float
    half_life_windows: float
    warmup_windows: int

    def __post_init__(self) -> None:
        if self.k_min < 1:
            raise ValueError("k_min must be >= 1 so every window observes acceptance")
        if self.k_max < self.k_min:
            raise ValueError("k_max must be >= k_min")
        if self.prior_c < 0:
            raise ValueError("prior_c must be non-negative")


class Scheduler(Protocol):
    def next_k(self) -> int: ...

    def observe(
        self, k_proposed: int, n_accepted: int, draft_ms: float, verify_ms: float
    ) -> None: ...


class AdaptiveScheduler:
    """Stateful wrapper over immutable estimators; observe() swaps them by reference."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._alpha = AcceptanceEstimator(
            prior=config.prior_alpha,
            half_life_windows=config.half_life_windows,
            warmup_windows=config.warmup_windows,
        )
        self._c = config.prior_c
        self._c_decay = self._alpha.decay
        self.windows = 0

    @property
    def alpha(self) -> float:
        return self._alpha.alpha

    @property
    def c(self) -> float:
        return self._c

    def next_k(self) -> int:
        k = best_k(self._alpha.alpha, self._c, self.config.k_max)
        return min(self.config.k_max, max(self.config.k_min, k))

    def observe(self, k_proposed: int, n_accepted: int, draft_ms: float, verify_ms: float) -> None:
        self._alpha = self._alpha.update(k_proposed, n_accepted)
        if k_proposed > 0 and verify_ms > 0.0:
            measured = (draft_ms / k_proposed) / verify_ms
            self._c = self._c_decay * self._c + (1.0 - self._c_decay) * measured
        self.windows += 1


def generate_adaptive(
    target: Backend,
    draft: Backend,
    prompt: Sequence[int],
    max_tokens: int,
    eos: frozenset[int],
    ctx: GenerationContext,
    guard: GuardReader,
    scheduler: Scheduler,
) -> GenerationResult:
    """Greedy speculative decoding with K chosen per window by the scheduler."""
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    run_id, tag, prompt_id, rep = ctx.run_id, ctx.workload_tag, ctx.prompt_id, ctx.rep
    tokens: list[int] = []
    rows: list[WindowRow] = []

    started = perf_counter()
    logits = target.prefill(prompt)
    draft.prefill(prompt)
    prefill_ms = (perf_counter() - started) * MS
    last = int(np.argmax(logits))
    tokens.append(last)
    if last in eos or len(tokens) >= max_tokens:
        return GenerationResult(tuple(tokens), tuple(rows), prefill_ms, last in eos)

    pending = [last]
    window_idx = 0
    stopped_on_eos = False
    while len(tokens) < max_tokens:
        k = scheduler.next_k()
        before = guard()
        w0 = perf_counter()
        d_logits = draft.verify(pending)[-1]
        drafts = [int(np.argmax(d_logits))]
        for i in range(1, k):
            d_logits = draft.decode_step(drafts[i - 1])
            drafts.append(int(np.argmax(d_logits)))
        w1 = perf_counter()
        verify_rows = target.verify([last, *drafts])
        w2 = perf_counter()
        argmaxes = np.argmax(verify_rows, axis=1)
        n = 0
        while n < k and drafts[n] == argmaxes[n]:
            n += 1
        correction = int(argmaxes[n])
        target.trim(k - n)
        draft.trim(max(0, k - 1 - n))
        w3 = perf_counter()
        after = guard()
        draft_ms = (w1 - w0) * MS
        verify_ms = (w2 - w1) * MS
        rows.append(
            (
                run_id, window_idx, len(tokens), k, n, draft_ms, verify_ms, tag,
                max(before.mem_pressure, after.mem_pressure),
                after.page_ins - before.page_ins,
                max(before.thermal_level, after.thermal_level),
                prompt_id, rep, (w3 - w0) * MS,
            )
        )  # fmt: skip
        scheduler.observe(k, n, draft_ms, verify_ms)
        window_idx += 1
        pending = [drafts[k - 1], correction] if n == k else [correction]
        last = correction
        for token in (*drafts[:n], correction):
            tokens.append(token)
            if token in eos:
                stopped_on_eos = True
                break
            if len(tokens) >= max_tokens:
                break
        if stopped_on_eos:
            break
    return GenerationResult(tuple(tokens), tuple(rows), prefill_ms, stopped_on_eos)
