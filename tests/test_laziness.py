"""CLAUDE.md trap 1: MLX is lazy. Timing without forcing evaluation measures queueing.

This file does two things. First it *demonstrates* the trap with raw mlx so
nobody can claim it is theoretical. Then it asserts MLXBackend does not fall
into it: every public method returns only after real compute has happened.
Needs weights; never runs in CI.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from acceptrate.config import DEFAULT_PAIR

pytestmark = pytest.mark.model

DRAFT_REPO = DEFAULT_PAIR.draft.repo  # type: ignore[union-attr]
PROMPT_TOKENS = list(range(100, 132))
# A 1B 4-bit decode step on an M4 is several milliseconds of real work. Anything
# under this is a queued graph, not a computed one.
REAL_WORK_MS = 1.0


@pytest.fixture(scope="module")
def backend():
    from acceptrate.backend.mlx_backend import MLXBackend

    b = MLXBackend.load(DRAFT_REPO)
    b.prefill(PROMPT_TOKENS)
    b.decode_step(5)  # warm the kernels so timings below are steady-state
    return b


def _ms(fn) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000.0


def test_the_trap_is_real_a_lazy_forward_pass_times_as_nearly_free(backend) -> None:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model = backend._model
    cache = make_prompt_cache(model)
    inputs = mx.array(PROMPT_TOKENS, dtype=mx.int32)[None]
    model(inputs, cache=cache)
    mx.eval(cache[0].keys)  # settle the prefill so only the decode step is measured
    step = mx.array([5], dtype=mx.int32)[None]

    lazy_ms = _ms(lambda: model(step, cache=cache))
    eager_ms = _ms(lambda: mx.eval(model(step, cache=cache)))

    assert lazy_ms < eager_ms, (lazy_ms, eager_ms)
    assert eager_ms > REAL_WORK_MS


def test_decode_step_returns_only_after_real_compute(backend) -> None:
    elapsed = _ms(lambda: backend.decode_step(7))

    assert elapsed > REAL_WORK_MS


def test_verify_returns_only_after_real_compute(backend) -> None:
    elapsed = _ms(lambda: backend.verify([1, 2, 3, 4]))

    assert elapsed > REAL_WORK_MS


def test_prefill_returns_only_after_real_compute(backend) -> None:
    elapsed = _ms(lambda: backend.prefill(PROMPT_TOKENS))

    assert elapsed > REAL_WORK_MS


def test_returned_logits_are_host_numpy_not_a_lazy_handle(backend) -> None:
    logits = backend.decode_step(9)

    assert isinstance(logits, np.ndarray)
    assert np.isfinite(logits).all()
