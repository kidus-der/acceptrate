"""serve/detok.py — incremental detokenisation with partial-multibyte hold-back.

The server decodes only the tokens accumulated since the last emitted text and
emits the delta. A trailing U+FFFD means the byte-level tokenizer is mid-way
through a multibyte character, so that text is held until the next token
completes it; `flush` releases whatever is left at the end of a generation.
"""

from __future__ import annotations

from collections.abc import Sequence

from acceptrate.serve.detok import Detok, flush, step
from tests.fakes import FakeTokenizer

REPLACEMENT = "�"


class ByteTokenizer:
    """Tokens are UTF-8 bytes; decoding a partial sequence yields U+FFFD, like tiktoken."""

    eos_token_ids: frozenset[int] = frozenset()

    def decode(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


def _drive(tokenizer, chunks):
    state = Detok()
    deltas = []
    for chunk in chunks:
        state, delta = step(state, tuple(chunk), tokenizer.decode)
        deltas.append(delta)
    return state, deltas


def test_emits_the_decoded_delta_for_each_chunk() -> None:
    tok = FakeTokenizer()

    _, deltas = _drive(tok, [[0, 1], [2], [3, 4, 5]])

    assert deltas == ["ab", "c", "def"]
    assert "".join(deltas) == tok.decode([0, 1, 2, 3, 4, 5])


def test_holds_back_a_partial_multibyte_character_until_it_completes() -> None:
    euro = "€".encode()  # three bytes

    _, deltas = _drive(ByteTokenizer(), [[ord("a"), euro[0]], [euro[1]], [euro[2], ord("b")]])

    assert deltas == ["", "", "a€b"]  # the whole chunk is held: text cannot map back to tokens
    assert REPLACEMENT not in "".join(deltas)


def test_pending_tokens_carry_over_and_state_is_immutable() -> None:
    euro = "€".encode()
    state0 = Detok()

    state1, delta1 = step(state0, (euro[0],), ByteTokenizer().decode)
    state2, delta2 = step(state1, (euro[1], euro[2]), ByteTokenizer().decode)

    assert (delta1, delta2) == ("", "€")
    assert state0.pending == ()
    assert state1.pending == (euro[0],)
    assert state2.pending == ()


def test_flush_releases_held_bytes_even_if_they_never_completed() -> None:
    euro = "€".encode()
    state, _ = step(Detok(), (ord("x"), euro[0]), ByteTokenizer().decode)

    assert flush(state, ByteTokenizer().decode) == "x" + REPLACEMENT
    assert flush(Detok(), ByteTokenizer().decode) == ""


def test_step_with_no_tokens_emits_nothing() -> None:
    state, delta = step(Detok(), (), FakeTokenizer().decode)

    assert delta == ""
    assert state == Detok()
