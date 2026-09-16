"""A deterministic, model-free Backend for unit tests.

Next-token rule: argmax(logits after token t) == (t * step + offset) % vocab_size.
Two fakes with the same rule agree on every token (a perfect draft); with
different rules they disagree on most tokens (a bad draft). That is enough
to exercise prefill / decode_step / verify / trim and the speculative
accept/reject path exactly, with no weights, no GPU and no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

STEP = 7


class FakeBackend:
    def __init__(self, vocab_size: int = 32, step: int = STEP, offset: int = 0) -> None:
        self.vocab_size = vocab_size
        self.step = step
        self.offset = offset
        self._tokens: list[int] = []
        self.calls: list[str] = []  # method names, for tests that check cache discipline

    @property
    def position(self) -> int:
        return len(self._tokens)

    def next_token(self, token: int) -> int:
        return (token * self.step + self.offset) % self.vocab_size

    def _logits_for(self, token: int) -> npt.NDArray[np.float32]:
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        logits[self.next_token(token)] = 1.0
        return logits

    def prefill(self, tokens: Sequence[int]) -> npt.NDArray[np.float32]:
        self.calls.append("prefill")
        self._tokens = list(tokens)
        return self._logits_for(self._tokens[-1])

    def decode_step(self, token: int) -> npt.NDArray[np.float32]:
        self.calls.append("decode_step")
        self._tokens.append(token)
        return self._logits_for(token)

    def verify(self, tokens: Sequence[int]) -> npt.NDArray[np.float32]:
        self.calls.append("verify")
        rows = [self._logits_for(t) for t in tokens]
        self._tokens.extend(tokens)
        return np.stack(rows)

    def trim(self, n: int) -> None:
        self.calls.append("trim")
        if n:
            del self._tokens[-n:]

    @property
    def tokens(self) -> tuple[int, ...]:
        """What the cache currently holds — lets tests assert rollback exactness."""
        return tuple(self._tokens)

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
