"""runtime/streaming.py — streaming twins of generate_plain / generate_speculative.

The losslessness contract for `acceptrate serve`: the streaming variants must
commit exactly the tokens, in exactly the windows, with exactly the rows, that
the non-streaming engine produces. Timing fields are the only allowed
difference, so rows are compared with draft_ms / verify_ms / window_ms masked.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.runtime.speculative import generate_speculative
from acceptrate.runtime.streaming import generate_plain_streaming, generate_speculative_streaming
from acceptrate.trace.schema import FIELD_NAMES
from tests.fakes import FakeBackend

CTX = GenerationContext(run_id="r", workload_tag="chat", prompt_id="p", rep=0)
V = 32
TIMING_FIELDS = frozenset({"draft_ms", "verify_ms", "window_ms"})
TIMING_INDICES = tuple(i for i, name in enumerate(FIELD_NAMES) if name in TIMING_FIELDS)


@dataclass(frozen=True)
class Snap:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


def _mask(row):
    return tuple(None if i in TIMING_INDICES else v for i, v in enumerate(row))


def _collect(events):
    tokens: list[int] = []
    rows = []
    for committed, row in events:
        tokens.extend(committed)
        if row is not None:
            rows.append(row)
    return tokens, rows


def test_first_event_is_the_prefill_token_without_a_row() -> None:
    events = list(generate_plain_streaming(FakeBackend(V), [3], 5, frozenset(), CTX, Snap))

    assert events[0] == ((FakeBackend(V).next_token(3),), None)
    assert all(row is not None for _, row in events[1:])


def test_plain_streaming_matches_plain_token_for_token_and_row_for_row() -> None:
    reference = generate_plain(FakeBackend(V), [3, 4], 20, frozenset(), CTX, Snap)

    tokens, rows = _collect(
        generate_plain_streaming(FakeBackend(V), [3, 4], 20, frozenset(), CTX, Snap)
    )

    assert tokens == list(reference.tokens)
    assert [_mask(r) for r in rows] == [_mask(r) for r in reference.rows]
    assert len(rows) == len(tokens) - 1


def test_plain_streaming_stops_at_eos_and_keeps_it() -> None:
    plain = generate_plain(FakeBackend(V), [3], 30, frozenset(), CTX, Snap).tokens
    eos = frozenset({plain[5]})
    reference = generate_plain(FakeBackend(V), [3], 30, eos, CTX, Snap)

    tokens, _ = _collect(generate_plain_streaming(FakeBackend(V), [3], 30, eos, CTX, Snap))

    assert tokens == list(reference.tokens)
    assert tokens[-1] in eos


def test_speculative_streaming_yields_one_event_per_window_with_its_tokens() -> None:
    events = list(
        generate_speculative_streaming(
            FakeBackend(V), FakeBackend(V), [3], 13, frozenset(), CTX, Snap, k=4
        )
    )

    assert events[0][1] is None
    committed_sizes = [len(tokens) for tokens, row in events[1:]]
    assert committed_sizes == [5, 5, 2]  # 1 + 5 + 5 + 2 == 13, truncated mid-window
    assert [row[4] for _, row in events[1:]] == [4, 4, 4]  # n_accepted per window


def test_speculative_streaming_matches_speculative_with_a_partial_draft() -> None:
    class PartialFake(FakeBackend):
        def next_token(self, token: int) -> int:
            return (token * 7 + (1 if token % 3 == 0 else 0)) % self.vocab_size

    reference = generate_speculative(
        FakeBackend(V), PartialFake(V), [3, 4], 40, frozenset(), CTX, Snap, k=3
    )

    tokens, rows = _collect(
        generate_speculative_streaming(
            FakeBackend(V), PartialFake(V), [3, 4], 40, frozenset(), CTX, Snap, k=3
        )
    )

    assert tokens == list(reference.tokens)
    assert [_mask(r) for r in rows] == [_mask(r) for r in reference.rows]


def test_speculative_streaming_cache_discipline_matches_the_non_streaming_engine() -> None:
    ref_target, ref_draft = FakeBackend(V), FakeBackend(V, step=3)
    generate_speculative(ref_target, ref_draft, [3], 25, frozenset(), CTX, Snap, k=4)
    target, draft = FakeBackend(V), FakeBackend(V, step=3)

    list(generate_speculative_streaming(target, draft, [3], 25, frozenset(), CTX, Snap, k=4))

    assert target.calls == ref_target.calls
    assert draft.calls == ref_draft.calls
    assert target.tokens == ref_target.tokens
    assert draft.tokens == ref_draft.tokens


@pytest.mark.parametrize("k", [0, -1])
def test_speculative_streaming_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError):
        list(
            generate_speculative_streaming(
                FakeBackend(V), FakeBackend(V), [3], 5, frozenset(), CTX, Snap, k=k
            )
        )


def test_streaming_variants_reject_an_empty_prompt() -> None:
    with pytest.raises(ValueError):
        list(generate_plain_streaming(FakeBackend(V), [], 5, frozenset(), CTX, Snap))
    with pytest.raises(ValueError):
        list(
            generate_speculative_streaming(
                FakeBackend(V), FakeBackend(V), [], 5, frozenset(), CTX, Snap, k=2
            )
        )


@settings(max_examples=100, deadline=None)
@given(
    step_d=st.integers(min_value=1, max_value=31),
    offset_d=st.integers(min_value=0, max_value=31),
    prompt=st.lists(st.integers(min_value=1, max_value=31), min_size=1, max_size=6),
    k=st.integers(min_value=1, max_value=6),
    max_tokens=st.integers(min_value=1, max_value=40),
    eos_token=st.integers(min_value=0, max_value=31),
)
def test_property_streaming_equals_non_streaming(
    step_d, offset_d, prompt, k, max_tokens, eos_token
):
    eos = frozenset({eos_token})
    reference = generate_speculative(
        FakeBackend(V), FakeBackend(V, step_d, offset_d), prompt, max_tokens, eos, CTX, Snap, k
    )

    tokens, rows = _collect(
        generate_speculative_streaming(
            FakeBackend(V), FakeBackend(V, step_d, offset_d), prompt, max_tokens, eos, CTX, Snap, k
        )
    )

    assert tokens == list(reference.tokens)
    assert [_mask(r) for r in rows] == [_mask(r) for r in reference.rows]
