"""Streaming twins of generate_plain / generate_speculative.

`acceptrate serve` must hand tokens to the client as they are committed, but
the engine functions return only at the end of a generation. These generators
have the same body, the same cache discipline and the same row shape as their
non-streaming twins; they only differ in *yielding* each window instead of
accumulating it. tests/test_streaming.py asserts token-for-token and
row-for-row equality on fakes. Keep both copies in lockstep: a divergence here
is a losslessness bug, never a feature.

Each event is `(tokens committed this window, the window's trace row)`. The
first event carries the prefill token and no row, mirroring the engine, which
does not write a row for the prefill.

Hot-loop rules (CLAUDE.md trap 4) still apply: no logging, no f-strings, no
dicts; tuples only.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from time import perf_counter

import numpy as np

from acceptrate.backend.protocol import Backend
from acceptrate.runtime.engine import MS, GenerationContext, GuardReader
from acceptrate.trace.schema import WindowRow

StreamEvent = tuple[tuple[int, ...], WindowRow | None]


def generate_plain_streaming(
    backend: Backend,
    prompt: Sequence[int],
    max_tokens: int,
    eos: frozenset[int],
    ctx: GenerationContext,
    guard: GuardReader,
) -> Iterator[StreamEvent]:
    """Greedy-decode with K = 0, yielding one token (and one row) per decode step."""
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    run_id, tag, prompt_id, rep = ctx.run_id, ctx.workload_tag, ctx.prompt_id, ctx.rep

    logits = backend.prefill(prompt)
    token = int(np.argmax(logits))
    produced = 1
    yield (token,), None
    if token in eos:
        return

    window_idx = 0
    while produced < max_tokens:
        before = guard()
        w0 = perf_counter()
        logits = backend.decode_step(token)
        verify_ms = (perf_counter() - w0) * MS
        token = int(np.argmax(logits))
        w1 = perf_counter()
        after = guard()
        row: WindowRow = (
            run_id, window_idx, produced, 0, 0, 0.0, verify_ms, tag,
            max(before.mem_pressure, after.mem_pressure),
            after.page_ins - before.page_ins,
            max(before.thermal_level, after.thermal_level),
            prompt_id, rep, (w1 - w0) * MS,
        )  # fmt: skip
        produced += 1
        window_idx += 1
        yield (token,), row
        if token in eos:
            return


def generate_speculative_streaming(
    target: Backend,
    draft: Backend,
    prompt: Sequence[int],
    max_tokens: int,
    eos: frozenset[int],
    ctx: GenerationContext,
    guard: GuardReader,
    k: int,
) -> Iterator[StreamEvent]:
    """Greedy speculative decoding with fixed K >= 1, yielding each window as it commits."""
    if k < 1:
        raise ValueError("draft depth K must be >= 1; use generate_plain_streaming for K = 0")
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    run_id, tag, prompt_id, rep = ctx.run_id, ctx.workload_tag, ctx.prompt_id, ctx.rep

    logits = target.prefill(prompt)
    draft.prefill(prompt)
    last = int(np.argmax(logits))
    produced = 1
    yield (last,), None
    if last in eos or produced >= max_tokens:
        return

    pending = [last]  # committed tokens the draft has not consumed yet
    window_idx = 0
    while produced < max_tokens:
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
        row: WindowRow = (
            run_id, window_idx, produced, k, n, (w1 - w0) * MS, (w2 - w1) * MS, tag,
            max(before.mem_pressure, after.mem_pressure),
            after.page_ins - before.page_ins,
            max(before.thermal_level, after.thermal_level),
            prompt_id, rep, (w3 - w0) * MS,
        )  # fmt: skip
        window_idx += 1
        pending = [drafts[k - 1], correction] if n == k else [correction]
        last = correction
        committed: list[int] = []
        stopped_on_eos = False
        for token in (*drafts[:n], correction):
            committed.append(token)
            produced += 1
            if token in eos:
                stopped_on_eos = True
                break
            if produced >= max_tokens:
                break
        yield tuple(committed), row
        if stopped_on_eos:
            return
