"""A deterministic, model-free Backend for unit tests.

Next-token rule: argmax(logits after token t) == (t * STEP) % vocab_size.
That is enough to exercise prefill / decode_step / verify / trim exactly,
with no weights, no GPU and no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

STEP = 7


class FakeBackend:
    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self._tokens: list[int] = []

    @property
    def position(self) -> int:
        return len(self._tokens)

    def _logits_for(self, token: int) -> npt.NDArray[np.float32]:
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        logits[(token * STEP) % self.vocab_size] = 1.0
        return logits

    def prefill(self, tokens: Sequence[int]) -> npt.NDArray[np.float32]:
        self._tokens = list(tokens)
        return self._logits_for(self._tokens[-1])

    def decode_step(self, token: int) -> npt.NDArray[np.float32]:
        self._tokens.append(token)
        return self._logits_for(token)

    def verify(self, tokens: Sequence[int]) -> npt.NDArray[np.float32]:
        rows = [self._logits_for(t) for t in tokens]
        self._tokens.extend(tokens)
        return np.stack(rows)

    def trim(self, n: int) -> None:
        if n:
            del self._tokens[-n:]

    def reset(self) -> None:
        self._tokens = []


class FakeTokenizer:
    eos_token_ids: frozenset[int] = frozenset({0})

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 or 1 for c in text]

    def encode_chat(self, messages: Sequence[dict[str, str]]) -> list[int]:
        return [2, *self.encode(" ".join(m["content"] for m in messages)), 3]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(97 + (t % 26)) for t in tokens)
