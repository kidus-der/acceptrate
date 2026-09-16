"""The portability seam.

Everything above this file — runtime/, model/, bench/, verify/ — talks to a
`Backend` and a `Tokenizer` and nothing else. Only backend/mlx_backend.py may
import mlx. A CUDA or llama.cpp backend is a new file implementing these two
protocols and nothing more.

Contract every implementation must honour:

* Every method returns only after its computation has fully completed on the
  device. Callers time around these calls; a lazy return measures queueing,
  not compute (CLAUDE.md trap 1).
* Logits are float32 numpy rows over the vocabulary, never framework tensors.
* `position` is the number of tokens currently held in the KV cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Logits = npt.NDArray[np.float32]
"""Shape (vocab_size,). Raw logits for the next position."""

LogitRows = npt.NDArray[np.float32]
"""Shape (n_tokens, vocab_size). Row i is the logits after consuming token i."""


@runtime_checkable
class Backend(Protocol):
    """A causal LM with a KV cache that can be rolled back."""

    vocab_size: int

    @property
    def position(self) -> int:
        """Tokens currently in the cache."""
        ...

    def prefill(self, tokens: Sequence[int]) -> Logits:
        """Reset the cache, consume the prompt, return logits for the next token."""
        ...

    def decode_step(self, token: int) -> Logits:
        """Append one token, return logits for the token after it."""
        ...

    def verify(self, tokens: Sequence[int]) -> LogitRows:
        """Append `tokens` in one forward pass; return logits after each of them.

        This is the speculative verify: one target pass scores a whole draft
        window. Row i lets the caller check draft token i+1 (or, for the last
        row, produce the bonus token).
        """
        ...

    def trim(self, n: int) -> None:
        """Drop the last `n` tokens from the cache (rejected draft tokens)."""
        ...


@runtime_checkable
class Tokenizer(Protocol):
    eos_token_ids: frozenset[int]

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...
