"""Prompt-lookup drafting as a Backend: n-gram matching, no draft model, c ~ 0.

The most recent `max_ngram` tokens are searched for earlier in the sequence
(prompt + committed output); the token that followed the longest, most
recent match is the draft. Structured text (code, JSON, quoted passages)
repeats itself and drafts nearly for free; novel prose mostly falls back to
a token that the target will reject, costing one cheap window.

Implements the Backend protocol so runtime/speculative.py and
runtime/adaptive.py use it unchanged. Logits are one-hot rows: every method
returns synchronously, so timing around it is honest (and tiny).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from acceptrate.backend.protocol import LogitRows, Logits


class LookupBackend:
    def __init__(self, vocab_size: int, max_ngram: int = 3, fallback: int = 0) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if max_ngram < 1:
            raise ValueError("max_ngram must be >= 1")
        if not 0 <= fallback < vocab_size:
            raise ValueError("fallback must be a valid token id")
        self.vocab_size = vocab_size
        self.max_ngram = max_ngram
        self.fallback = fallback
        self._tokens: list[int] = []

    @property
    def position(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(self._tokens)

    def _predict(self) -> int:
        seq = self._tokens
        n_max = min(self.max_ngram, len(seq) - 1)
        for n in range(n_max, 0, -1):
            key = seq[-n:]
            # most recent earlier occurrence of the key, followed by at least one token
            for start in range(len(seq) - n - 1, -1, -1):
                if seq[start : start + n] == key:
                    return seq[start + n]
        return self.fallback

    def _one_hot(self, token: int) -> Logits:
        row = np.zeros(self.vocab_size, dtype=np.float32)
        row[token] = 1.0
        return row

    def prefill(self, tokens: Sequence[int]) -> Logits:
        self._tokens = list(tokens)
        return self._one_hot(self._predict())

    def decode_step(self, token: int) -> Logits:
        self._tokens.append(token)
        return self._one_hot(self._predict())

    def verify(self, tokens: Sequence[int]) -> LogitRows:
        rows = np.zeros((len(tokens), self.vocab_size), dtype=np.float32)
        for i, token in enumerate(tokens):
            self._tokens.append(token)
            rows[i, self._predict()] = 1.0
        return rows

    def trim(self, n: int) -> None:
        if n > 0:
            del self._tokens[-n:]
