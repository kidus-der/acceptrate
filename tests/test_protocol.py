"""The Backend protocol is the horizontal seam. This pins its shape."""

from __future__ import annotations

import numpy as np

from acceptrate.backend.protocol import Backend, Tokenizer
from tests.fakes import FakeBackend, FakeTokenizer


def test_fake_backend_satisfies_the_runtime_checkable_protocol() -> None:
    assert isinstance(FakeBackend(), Backend)


def test_fake_tokenizer_satisfies_the_tokenizer_protocol() -> None:
    assert isinstance(FakeTokenizer(), Tokenizer)


def test_an_object_missing_verify_is_not_a_backend() -> None:
    class Partial:
        vocab_size = 4

        def prefill(self, tokens):
            return np.zeros(4, dtype=np.float32)

        def decode_step(self, token):
            return np.zeros(4, dtype=np.float32)

        def trim(self, n):
            return None

    assert not isinstance(Partial(), Backend)


def test_prefill_returns_one_logit_row_and_sets_position() -> None:
    backend = FakeBackend(vocab_size=32)

    logits = backend.prefill([1, 2, 3])

    assert logits.shape == (32,)
    assert logits.dtype == np.float32
    assert backend.position == 3


def test_verify_returns_one_row_per_token_and_advances_position() -> None:
    backend = FakeBackend(vocab_size=32)
    backend.prefill([1])

    rows = backend.verify([4, 5, 6])

    assert rows.shape == (3, 32)
    assert backend.position == 4


def test_verify_rows_match_sequential_decode_steps() -> None:
    sequential = FakeBackend()
    sequential.prefill([1])
    expected = np.stack([sequential.decode_step(t) for t in (4, 5, 6)])
    batched = FakeBackend()
    batched.prefill([1])

    rows = batched.verify([4, 5, 6])

    np.testing.assert_array_equal(rows, expected)


def test_trim_rolls_back_position_so_decode_continues_from_the_kept_prefix() -> None:
    backend = FakeBackend()
    backend.prefill([1])
    backend.verify([4, 5, 6])

    backend.trim(2)

    assert backend.position == 2
    np.testing.assert_array_equal(backend.decode_step(9), FakeBackend()._logits_for(9))
    assert backend.position == 3


def test_tokenizer_protocol_requires_encode_chat() -> None:
    class NoChat:
        eos_token_ids = frozenset({0})

        def encode(self, text):
            return [1]

        def decode(self, tokens):
            return ""

    assert not isinstance(NoChat(), Tokenizer)


def test_fake_tokenizer_encode_chat_returns_token_ids() -> None:
    tokens = FakeTokenizer().encode_chat([{"role": "user", "content": "hello"}])

    assert tokens
    assert all(isinstance(t, int) for t in tokens)
