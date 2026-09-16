"""P0 smoke path: plain greedy streaming from a Backend.

Deliberately naive — no speculation, no timing, no trace. Its only job is to
prove that a backend loads and streams tokens end to end.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from acceptrate.backend.protocol import Backend


@dataclass(frozen=True)
class SmokeResult:
    tokens: list[int] = field(default_factory=list)
    stopped_on_eos: bool = False


def stream_greedy(
    backend: Backend,
    prompt_tokens: Sequence[int],
    max_tokens: int,
    eos: frozenset[int],
    on_token: Callable[[int], None] | None = None,
) -> SmokeResult:
    """Greedy-decode up to `max_tokens` after `prompt_tokens`, stopping at EOS."""
    tokens: list[int] = []
    logits = backend.prefill(prompt_tokens)
    for _ in range(max_tokens):
        token = int(np.argmax(logits))
        tokens.append(token)
        if on_token is not None:
            on_token(token)
        if token in eos:
            return SmokeResult(tokens=tokens, stopped_on_eos=True)
        logits = backend.decode_step(token)
    return SmokeResult(tokens=tokens, stopped_on_eos=False)
