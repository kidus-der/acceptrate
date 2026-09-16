"""MLXBackend against the real 1B draft model. Needs weights; never runs in CI."""

from __future__ import annotations

import numpy as np
import pytest

from acceptrate.backend.protocol import Backend, Tokenizer
from acceptrate.config import DEFAULT_PAIR

pytestmark = pytest.mark.model

PROMPT = "The quick brown fox"
DRAFT_REPO = DEFAULT_PAIR.draft.repo  # type: ignore[union-attr]


@pytest.fixture(scope="module")
def backend():
    from acceptrate.backend.mlx_backend import MLXBackend

    return MLXBackend.load(DRAFT_REPO)


def test_backend_satisfies_the_protocol(backend) -> None:
    assert isinstance(backend, Backend)
    assert isinstance(backend.tokenizer, Tokenizer)
    assert backend.vocab_size == 128256


def test_prefill_returns_float32_row_over_the_vocab(backend) -> None:
    tokens = backend.tokenizer.encode(PROMPT)

    logits = backend.prefill(tokens)

    assert logits.shape == (backend.vocab_size,)
    assert logits.dtype == np.float32
    assert backend.position == len(tokens)


def test_prefill_is_deterministic(backend) -> None:
    tokens = backend.tokenizer.encode(PROMPT)

    first = backend.prefill(tokens)
    second = backend.prefill(tokens)

    np.testing.assert_array_equal(first, second)


def test_verify_rows_agree_with_sequential_decode_steps(backend) -> None:
    tokens = backend.tokenizer.encode(PROMPT)
    logits = backend.prefill(tokens)
    continuation = [int(np.argmax(logits))]
    for _ in range(3):
        continuation.append(int(np.argmax(backend.decode_step(continuation[-1]))))
    sequential = np.stack([backend.decode_step(t) for t in continuation[:0]] or [logits])
    backend.prefill(tokens)
    expected = np.stack([backend.decode_step(t) for t in continuation])
    backend.prefill(tokens)

    rows = backend.verify(continuation)

    assert rows.shape == (len(continuation), backend.vocab_size)
    assert backend.position == len(tokens) + len(continuation)
    np.testing.assert_array_equal(rows.argmax(axis=1), expected.argmax(axis=1))
    np.testing.assert_allclose(rows, expected, rtol=1e-2, atol=1e-2)
    del sequential


def test_trim_then_decode_matches_the_untrimmed_path(backend) -> None:
    tokens = backend.tokenizer.encode(PROMPT)
    logits = backend.prefill(tokens)
    good = int(np.argmax(logits))
    backend.decode_step(good)
    expected = backend.decode_step(good)
    backend.prefill(tokens)
    backend.verify([good, 1, 2, 3])  # three rejected drafts

    backend.trim(3)
    actual = backend.decode_step(good)

    assert backend.position == len(tokens) + 2
    np.testing.assert_array_equal(actual.argmax(), expected.argmax())
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


def test_tokenizer_roundtrips_text(backend) -> None:
    tokens = backend.tokenizer.encode(PROMPT)

    assert backend.tokenizer.decode(tokens).strip() == PROMPT
    assert isinstance(backend.tokenizer.eos_token_ids, frozenset)
    assert backend.tokenizer.eos_token_ids
