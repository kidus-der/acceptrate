"""The engine: drives a Backend and emits ONE trace row per draft window.

P1 ships plain decoding only (K = 0): every window is a single decode step,
n_accepted is 0 and draft_ms is 0. Speculation arrives in P2/P3 through the
same row shape, so the baseline is simply the K = 0 cell of the grid.

Hot-loop rules (CLAUDE.md trap 4): no logging, no f-strings, no dicts. Rows
are tuples appended to a list; the caller types them once per generation.
The guard reader is injected by the composition root so this module never
imports bench/.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from acceptrate.backend.protocol import Backend
from acceptrate.trace.schema import WindowRow

MS = 1000.0


class GuardLike(Protocol):
    mem_pressure: int
    page_ins: int
    thermal_level: int


GuardReader = Callable[[], GuardLike]
"""Returns the latest system snapshot. Must not allocate or block."""


@dataclass(frozen=True)
class GenerationContext:
    run_id: str
    workload_tag: str
    prompt_id: str
    rep: int


@dataclass(frozen=True)
class GenerationResult:
    tokens: tuple[int, ...]
    rows: tuple[WindowRow, ...]
    prefill_ms: float
    stopped_on_eos: bool


def generate_plain(
    backend: Backend,
    prompt: Sequence[int],
    max_tokens: int,
    eos: frozenset[int],
    ctx: GenerationContext,
    guard: GuardReader,
) -> GenerationResult:
    """Greedy-decode up to max_tokens with K = 0, one row per decode step."""
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    run_id, tag, prompt_id, rep = ctx.run_id, ctx.workload_tag, ctx.prompt_id, ctx.rep
    tokens: list[int] = []
    rows: list[WindowRow] = []

    started = perf_counter()
    logits = backend.prefill(prompt)
    prefill_ms = (perf_counter() - started) * MS
    token = int(np.argmax(logits))
    tokens.append(token)
    if token in eos:
        return GenerationResult(tuple(tokens), tuple(rows), prefill_ms, True)

    window_idx = 0
    while len(tokens) < max_tokens:
        before = guard()
        w0 = perf_counter()
        logits = backend.decode_step(token)
        verify_ms = (perf_counter() - w0) * MS
        token = int(np.argmax(logits))
        w1 = perf_counter()
        after = guard()
        rows.append(
            (
                run_id, window_idx, len(tokens), 0, 0, 0.0, verify_ms, tag,
                max(before.mem_pressure, after.mem_pressure),
                after.page_ins - before.page_ins,
                max(before.thermal_level, after.thermal_level),
                prompt_id, rep, (w1 - w0) * MS,
            )
        )  # fmt: skip
        tokens.append(token)
        window_idx += 1
        if token in eos:
            return GenerationResult(tuple(tokens), tuple(rows), prefill_ms, True)
    return GenerationResult(tuple(tokens), tuple(rows), prefill_ms, False)
