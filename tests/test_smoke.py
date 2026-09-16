"""The P0 smoke path: stream N tokens from a backend, plain greedy decode.

No speculation, no timing claims — just proof that a backend loads, runs,
and streams. Model-free here via FakeBackend; the CLI wires MLXBackend.
"""

from __future__ import annotations

from acceptrate.bench.smoke import SmokeResult, stream_greedy
from tests.fakes import STEP, FakeBackend, FakeTokenizer


def test_streams_exactly_n_tokens() -> None:
    backend = FakeBackend(vocab_size=32)

    result = stream_greedy(backend, prompt_tokens=[1, 2, 3], max_tokens=128, eos=frozenset())

    assert isinstance(result, SmokeResult)
    assert len(result.tokens) == 128
    assert backend.position == 3 + 128


def test_greedy_follows_the_backend_argmax() -> None:
    backend = FakeBackend(vocab_size=32)

    result = stream_greedy(backend, prompt_tokens=[3], max_tokens=4, eos=frozenset())

    expected = [(3 * STEP) % 32]
    for _ in range(3):
        expected.append((expected[-1] * STEP) % 32)
    assert result.tokens == expected


def test_stops_early_on_eos() -> None:
    backend = FakeBackend(vocab_size=32)
    first = (3 * STEP) % 32

    result = stream_greedy(backend, prompt_tokens=[3], max_tokens=50, eos=frozenset({first}))

    assert result.tokens == [first]
    assert result.stopped_on_eos


def test_calls_on_token_for_each_streamed_token() -> None:
    backend = FakeBackend(vocab_size=32)
    seen: list[int] = []

    result = stream_greedy(
        backend, prompt_tokens=[3], max_tokens=5, eos=frozenset(), on_token=seen.append
    )

    assert seen == result.tokens


def test_fake_tokenizer_is_usable_end_to_end() -> None:
    tokenizer = FakeTokenizer()
    backend = FakeBackend(vocab_size=32)

    result = stream_greedy(
        backend, tokenizer.encode("hi"), max_tokens=3, eos=tokenizer.eos_token_ids
    )

    assert isinstance(tokenizer.decode(result.tokens), str)
