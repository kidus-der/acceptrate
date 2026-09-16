"""Fakes shared by the serve tests: a backend that can be held mid-generation."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from tests.fakes import FakeBackend, FakeTokenizer

HOLD_TIMEOUT_S = 5.0


class HoldableBackend(FakeBackend):
    """A FakeBackend whose decode/verify block on `gate` while it is cleared.

    `entered` is set the first time a forward call blocks, so a test can wait
    until the generation is provably in flight before probing for 429.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def _wait(self) -> None:
        if not self.gate.is_set():
            self.entered.set()
            if not self.gate.wait(HOLD_TIMEOUT_S):
                raise TimeoutError("HoldableBackend gate was never released")

    def decode_step(self, token: int) -> npt.NDArray[np.float32]:
        self._wait()
        return super().decode_step(token)

    def verify(self, tokens: Sequence[int]) -> npt.NDArray[np.float32]:
        self._wait()
        return super().verify(tokens)


class EosTokenizer(FakeTokenizer):
    """FakeTokenizer with a configurable end-of-sequence set."""

    def __init__(self, eos: frozenset[int]) -> None:
        self.eos_token_ids = eos
