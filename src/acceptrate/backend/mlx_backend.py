"""The single Apple-specific file in the repo.

Wraps an mlx-lm model and its KV cache behind the Backend protocol. Every
public method forces evaluation with mx.eval before returning, so a caller's
clock measures compute rather than lazy queueing (CLAUDE.md trap 1). Nothing
outside this file may import mlx or mlx_lm; tests/test_boundaries.py enforces
that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from acceptrate.backend.protocol import LogitRows, Logits


@dataclass(frozen=True)
class MLXTokenizer:
    """Adapts mlx-lm's TokenizerWrapper to the Tokenizer protocol."""

    _wrapped: Any
    eos_token_ids: frozenset[int]

    @classmethod
    def from_wrapper(cls, wrapper: Any) -> MLXTokenizer:
        return cls(_wrapped=wrapper, eos_token_ids=frozenset(int(t) for t in wrapper.eos_token_ids))

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in self._wrapped.encode(text, add_special_tokens=False)]

    def decode(self, tokens: Sequence[int]) -> str:
        return str(self._wrapped.decode(list(tokens)))


class MLXBackend:
    """A causal LM on Metal with a rewindable KV cache."""

    def __init__(self, model: Any, tokenizer: MLXTokenizer, vocab_size: int) -> None:
        self._model = model
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self._cache: list[Any] = make_prompt_cache(model)
        self._position = 0

    @classmethod
    def load(cls, repo: str) -> MLXBackend:
        model, wrapper, config = mlx_lm.load(repo, return_config=True)
        vocab_size = int(config["vocab_size"])
        return cls(model, MLXTokenizer.from_wrapper(wrapper), vocab_size)

    @property
    def position(self) -> int:
        return self._position

    def _forward(self, tokens: Sequence[int]) -> mx.array:
        inputs = mx.array(list(tokens), dtype=mx.int32)[None]
        logits = self._model(inputs, cache=self._cache)[0]
        mx.eval(logits)
        self._position += len(tokens)
        return logits

    def prefill(self, tokens: Sequence[int]) -> Logits:
        self._cache = make_prompt_cache(self._model)
        self._position = 0
        logits = self._forward(tokens)[-1]
        return np.array(logits.astype(mx.float32), copy=False)

    def decode_step(self, token: int) -> Logits:
        logits = self._forward([token])[-1]
        return np.array(logits.astype(mx.float32), copy=False)

    def verify(self, tokens: Sequence[int]) -> LogitRows:
        logits = self._forward(tokens)
        return np.array(logits.astype(mx.float32), copy=False)

    def trim(self, n: int) -> None:
        if n <= 0:
            return
        trimmed = trim_prompt_cache(self._cache, n)
        if trimmed != n:
            raise RuntimeError(f"cache trimmed {trimmed} tokens, expected {n}")
        self._position -= n
