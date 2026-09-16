"""A deterministic Backend whose logits are genuinely spread out, for sampling tests.

The logits after token t are row t of a fixed pseudo-random table drawn
from N(0, scale^2) with a seeded generator: reproducible, Markov (they
depend only on the last token, like tests/fakes.py) and *not* one-hot, so
softmax sampling is non-trivial and the rejection-sampling verify step has
real work to do.

Draft models: `ProbFake(vocab_size, seed=other)` is an unrelated draft
(low acceptance); `ProbFake.perturbed(target, seed, noise)` is the target's
table plus Gaussian noise, a draft that is correlated with the target the
way a small model is with a large one (moderate acceptance).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

DEFAULT_SCALE = 2.0
"""Logit spread. 2.0 gives peaked-but-not-degenerate next-token laws over a 32 vocab."""


class ProbFake:
    def __init__(
        self,
        vocab_size: int = 32,
        seed: int = 0,
        scale: float = DEFAULT_SCALE,
        table: npt.NDArray[np.float32] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        if table is None:
            rng = np.random.default_rng(seed)
            table = (rng.normal(size=(vocab_size, vocab_size)) * scale).astype(np.float32)
        if table.shape != (vocab_size, vocab_size):
            raise ValueError(f"table must be ({vocab_size}, {vocab_size}), got {table.shape}")
        self.table = table
        self._tokens: list[int] = []
        self.calls: list[str] = []

    @classmethod
    def perturbed(cls, target: ProbFake, seed: int, noise: float) -> ProbFake:
        """A draft whose logits are the target's plus N(0, noise^2)."""
        rng = np.random.default_rng(seed)
        jitter = (rng.normal(size=target.table.shape) * noise).astype(np.float32)
        return cls(target.vocab_size, table=target.table + jitter)

    @property
    def position(self) -> int:
        return len(self._tokens)

    def _logits_for(self, token: int) -> npt.NDArray[np.float32]:
        return self.table[token].copy()

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
        return tuple(self._tokens)
