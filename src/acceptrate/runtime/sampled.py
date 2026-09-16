"""Temperature > 0 generation: plain sampling and rejection-sampling speculative decoding.

Speculative sampling (Leviathan et al. 2023, Chen et al. 2023): the draft
proposes x_i ~ q_i; the target scores the whole window in one pass; x_i is
accepted with probability min(1, p_i(x_i) / q_i(x_i)). On the first
rejection the correction is drawn from normalise(max(0, p_i - q_i)); if
every draft is accepted the bonus is drawn from p_K. The committed tokens
are then distributed exactly as plain sampling from p. At T -> 0 this
collapses to runtime/speculative.py's greedy rule.

Cache discipline is copied from runtime/speculative.py verbatim (t = last
committed token, d = drafts):
  target.verify([t, d0..dK-1])  -> K+1 logit rows; cache = ... t d0..dK-1
  accept n; target.trim(K - n)  -> cache = ... t d0..dn-1
  draft consumed `pending` + d0..dK-2; draft.trim(max(0, K-1-n))
  next window feeds the draft [dK-1, c] if n == K else [c].

One trace row per window. Hot-loop rules (trap 4): no logging, no
f-strings, no dicts; tuples appended to a list.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from acceptrate.backend.protocol import Backend
from acceptrate.model.sampling import (
    accept_probability,
    residual_distribution,
    sample,
    softmax_with_temperature,
)
from acceptrate.runtime.engine import MS, GenerationContext, GenerationResult, GuardReader
from acceptrate.trace.schema import WindowRow


def _validate(prompt: Sequence[int], temperature: float) -> None:
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    if not temperature > 0.0:
        raise ValueError(f"temperature must be > 0 for sampling, got {temperature}")


def generate_plain_sampled(
    backend: Backend,
    prompt: Sequence[int],
    max_tokens: int,
    eos: frozenset[int],
    ctx: GenerationContext,
    guard: GuardReader,
    temperature: float,
    rng: np.random.Generator,
) -> GenerationResult:
    """Sample up to max_tokens with K = 0, one row per decode step."""
    _validate(prompt, temperature)
    run_id, tag, prompt_id, rep = ctx.run_id, ctx.workload_tag, ctx.prompt_id, ctx.rep
    tokens: list[int] = []
    rows: list[WindowRow] = []

    started = perf_counter()
    logits = backend.prefill(prompt)
    prefill_ms = (perf_counter() - started) * MS
    token = sample(softmax_with_temperature(logits, temperature), rng)
    tokens.append(token)
    if token in eos:
        return GenerationResult(tuple(tokens), tuple(rows), prefill_ms, True)

    window_idx = 0
    while len(tokens) < max_tokens:
        before = guard()
        w0 = perf_counter()
        logits = backend.decode_step(token)
        verify_ms = (perf_counter() - w0) * MS
        token = sample(softmax_with_temperature(logits, temperature), rng)
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


def _verify_window(
    verify_rows: np.ndarray,
    drafts: list[int],
    q_rows: list[np.ndarray],
    temperature: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """(n_accepted, next token): rejection-sample the drafts against the target rows."""
    k = len(drafts)
    n = 0
    while n < k:
        p = softmax_with_temperature(verify_rows[n], temperature)
        q = q_rows[n]
        x = drafts[n]
        if rng.random() < accept_probability(p[x], q[x]):
            n += 1
            continue
        return n, sample(residual_distribution(p, q), rng)
    return k, sample(softmax_with_temperature(verify_rows[k], temperature), rng)


def generate_speculative_sampled(
    target: Backend,
    draft: Backend,
    prompt: Sequence[int],
    max_tokens: int,
    eos: frozenset[int],
    ctx: GenerationContext,
    guard: GuardReader,
    k: int,
    temperature: float,
    rng: np.random.Generator,
) -> GenerationResult:
    """Rejection-sampling speculative decoding with a fixed draft depth K >= 1."""
    if k < 1:
        raise ValueError("draft depth K must be >= 1; use generate_plain_sampled for K = 0")
    _validate(prompt, temperature)
    run_id, tag, prompt_id, rep = ctx.run_id, ctx.workload_tag, ctx.prompt_id, ctx.rep
    tokens: list[int] = []
    rows: list[WindowRow] = []

    started = perf_counter()
    logits = target.prefill(prompt)
    draft.prefill(prompt)
    prefill_ms = (perf_counter() - started) * MS
    last = sample(softmax_with_temperature(logits, temperature), rng)
    tokens.append(last)
    if last in eos or len(tokens) >= max_tokens:
        return GenerationResult(tuple(tokens), tuple(rows), prefill_ms, last in eos)

    pending = [last]  # committed tokens the draft has not consumed yet
    window_idx = 0
    stopped_on_eos = False
    while len(tokens) < max_tokens:
        before = guard()
        w0 = perf_counter()
        q = softmax_with_temperature(draft.verify(pending)[-1], temperature)
        q_rows = [q]
        drafts = [sample(q, rng)]
        for i in range(1, k):
            q = softmax_with_temperature(draft.decode_step(drafts[i - 1]), temperature)
            q_rows.append(q)
            drafts.append(sample(q, rng))
        w1 = perf_counter()
        verify_rows = target.verify([last, *drafts])
        w2 = perf_counter()
        n, correction = _verify_window(verify_rows, drafts, q_rows, temperature, rng)
        target.trim(k - n)
        draft.trim(max(0, k - 1 - n))
        w3 = perf_counter()
        after = guard()
        rows.append(
            (
                run_id, window_idx, len(tokens), k, n, (w1 - w0) * MS, (w2 - w1) * MS, tag,
                max(before.mem_pressure, after.mem_pressure),
                after.page_ins - before.page_ins,
                max(before.thermal_level, after.thermal_level),
                prompt_id, rep, (w3 - w0) * MS,
            )
        )  # fmt: skip
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
