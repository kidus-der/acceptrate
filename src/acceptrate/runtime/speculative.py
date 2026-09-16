"""Fixed-K speculative generation: draft K tokens, verify in one target pass.

One trace row per draft window (CLAUDE.md). Greedy acceptance: a draft
token is accepted iff it equals the target's argmax at that position, so the
committed sequence is exactly what plain greedy decoding would produce.

Cache discipline per window (t = last committed token, d = drafts):
  target.verify([t, d0..dK-1])  -> K+1 logit rows; cache = ... t d0..dK-1
  accept n; target.trim(K - n)  -> cache = ... t d0..dn-1
  draft consumed `pending` + d0..dK-2; draft.trim(max(0, K-1-n))
  next window feeds the draft whatever it has not seen: [dK-1, c] if n == K
  else [c], where c is the correction (n < K) or bonus (n == K) token.

Hot-loop rules (trap 4): no logging, no f-strings, no dicts; tuples appended
to a list and typed once per generation by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from acceptrate.backend.protocol import Backend
from acceptrate.runtime.engine import MS, GenerationContext, GenerationResult, GuardReader
from acceptrate.trace.schema import WindowRow


def generate_speculative(
    target: Backend,
    draft: Backend,
    prompt: Sequence[int],
    max_tokens: int,
    eos: frozenset[int],
    ctx: GenerationContext,
    guard: GuardReader,
    k: int,
) -> GenerationResult:
    """Greedy speculative decoding with a fixed draft depth K >= 1."""
    if k < 1:
        raise ValueError("draft depth K must be >= 1; use generate_plain for K = 0")
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

    pending = [last]  # committed tokens the draft has not consumed yet
    window_idx = 0
    stopped_on_eos = False
    while len(tokens) < max_tokens:
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
