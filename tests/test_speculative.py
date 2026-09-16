"""runtime/speculative.py — fixed-K draft/verify generation, one trace row per window.

Losslessness at the logic level: with deterministic fakes, speculative
output must equal plain greedy output for every draft, good or bad.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.runtime.speculative import generate_speculative
from acceptrate.trace.schema import FIELD_NAMES, frame_from_rows
from tests.fakes import FakeBackend

CTX = GenerationContext(run_id="r", workload_tag="code", prompt_id="code-001", rep=0)
V = 32


@dataclass(frozen=True)
class Snap:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


class PartialFake(FakeBackend):
    """Agrees with FakeBackend(step=7) except after tokens divisible by 3."""

    def next_token(self, token: int) -> int:
        wrong = 1 if token % 3 == 0 else 0
        return (token * 7 + wrong) % self.vocab_size


def _plain(prompt, max_tokens, eos=frozenset()):
    return list(generate_plain(FakeBackend(V, step=7), prompt, max_tokens, eos, CTX, Snap).tokens)


def _spec(draft, prompt, max_tokens, k, eos=frozenset()):
    target = FakeBackend(V, step=7)
    result = generate_speculative(target, draft, prompt, max_tokens, eos, CTX, Snap, k=k)
    return target, draft, result


def test_perfect_draft_accepts_every_token_and_matches_plain() -> None:
    _, _, result = _spec(FakeBackend(V, step=7), [3], max_tokens=20, k=4)

    assert list(result.tokens) == _plain([3], 20)
    df = frame_from_rows(result.rows)
    assert set(df["k_proposed"]) == {4}
    assert set(df["n_accepted"]) == {4}


def test_always_wrong_draft_accepts_nothing_and_still_matches_plain() -> None:
    _, _, result = _spec(FakeBackend(V, step=3), [3], max_tokens=20, k=4)

    assert list(result.tokens) == _plain([3], 20)
    df = frame_from_rows(result.rows)
    assert set(df["n_accepted"]) == {0}
    assert len(result.rows) == 19  # one correction token per window after the prefill token


def test_partial_draft_mixes_accept_counts_and_matches_plain() -> None:
    _, _, result = _spec(PartialFake(V), [3], max_tokens=40, k=4)

    assert list(result.tokens) == _plain([3], 40)
    counts = set(frame_from_rows(result.rows)["n_accepted"])
    assert len(counts) > 1
    assert counts <= {0, 1, 2, 3, 4}


def test_rows_carry_positions_timings_and_context() -> None:
    _, _, result = _spec(PartialFake(V), [3], max_tokens=24, k=3)

    df = frame_from_rows(result.rows)
    assert list(df["window_idx"]) == list(range(len(result.rows)))
    positions = list(df["token_pos"])
    assert positions[0] == 1
    assert all(b > a for a, b in pairwise(positions))
    assert (df["draft_ms"] >= 0).all()
    assert (df["verify_ms"] >= 0).all()
    assert (df["window_ms"] >= df["draft_ms"] + df["verify_ms"]).all()
    assert set(df["workload_tag"]) == {"code"}
    assert type(result.rows[0]) is tuple and len(result.rows[0]) == len(FIELD_NAMES)


def test_stops_at_max_tokens_exactly_even_mid_window() -> None:
    _, _, result = _spec(FakeBackend(V, step=7), [3], max_tokens=10, k=4)

    assert len(result.tokens) == 10
    assert list(result.tokens) == _plain([3], 10)


def test_stops_at_eos_and_keeps_the_eos_token() -> None:
    plain = _plain([3], 30)
    eos_token = plain[6]
    first_hit = plain.index(eos_token)  # the fake's rule cycles, so it may appear earlier

    eos = frozenset({eos_token})

    _, _, result = _spec(FakeBackend(V, step=7), [3], max_tokens=30, k=4, eos=eos)

    assert list(result.tokens) == plain[: first_hit + 1]
    assert result.stopped_on_eos


def test_caches_hold_exactly_the_committed_prefix() -> None:
    target, draft, result = _spec(PartialFake(V), [3, 4], max_tokens=25, k=4)

    committed = [3, 4, *result.tokens]
    assert target.tokens[: len(committed) - 1] == tuple(committed[:-1])
    assert "trim" in target.calls
    assert draft.tokens == tuple(committed[: len(draft.tokens)])


def test_draft_costs_k_forward_calls_per_window_and_target_one() -> None:
    target, draft, result = _spec(FakeBackend(V, step=3), [3], max_tokens=9, k=4)

    windows = len(result.rows)
    assert target.calls.count("verify") == windows
    draft_forwards = draft.calls.count("verify") + draft.calls.count("decode_step")
    assert draft_forwards == windows * 4


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError):
        _spec(FakeBackend(V), [3], max_tokens=5, k=k)


@settings(max_examples=150, deadline=None)
@given(
    step_d=st.integers(min_value=1, max_value=31),
    offset_d=st.integers(min_value=0, max_value=31),
    prompt=st.lists(st.integers(min_value=1, max_value=31), min_size=1, max_size=6),
    k=st.integers(min_value=1, max_value=6),
    max_tokens=st.integers(min_value=1, max_value=40),
)
def test_property_speculative_equals_plain_for_any_draft(step_d, offset_d, prompt, k, max_tokens):
    draft = FakeBackend(V, step=step_d, offset=offset_d)

    _, _, result = _spec(draft, prompt, max_tokens, k)

    assert list(result.tokens) == _plain(prompt, max_tokens)
