"""Incremental detokenisation for streaming responses.

Decode only the tokens accumulated since the last emitted text and emit the
delta. Byte-level BPE (Llama 3's tokenizer) can split a multibyte character
across tokens; decoding the partial sequence yields U+FFFD, so that text is
held back until the next token completes it. `flush` releases whatever is
left at the end of a generation.

State is an immutable value; `step` returns a new one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

Decode = Callable[[Sequence[int]], str]

REPLACEMENT_CHAR = "�"


@dataclass(frozen=True)
class Detok:
    pending: tuple[int, ...] = ()
    """Tokens received but not yet emitted as text (a partial multibyte character)."""


def step(state: Detok, tokens: tuple[int, ...], decode: Decode) -> tuple[Detok, str]:
    """Absorb `tokens`; return the new state and the text that is safe to emit now."""
    if not tokens:
        return state, ""
    pending = (*state.pending, *tokens)
    text = decode(pending)
    if text.endswith(REPLACEMENT_CHAR):
        return Detok(pending), ""
    return Detok(), text


def flush(state: Detok, decode: Decode) -> str:
    """Emit whatever is still held, replacement characters and all."""
    return decode(state.pending) if state.pending else ""
