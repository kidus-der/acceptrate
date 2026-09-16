"""runtime/engine.py — plain (K=0) generation that emits one trace row per window."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from acceptrate.runtime.engine import GenerationContext, GenerationResult, generate_plain
from acceptrate.trace.schema import FIELD_NAMES, frame_from_rows
from tests.fakes import STEP, FakeBackend

CTX = GenerationContext(run_id="run-a", workload_tag="code", prompt_id="code-001", rep=0)


@dataclass(frozen=True)
class Snap:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


class ScriptedGuard:
    """Returns snapshots from a script, repeating the last one forever."""

    def __init__(self, *snaps: Snap) -> None:
        self._snaps = list(snaps)
        self._i = 0

    def __call__(self) -> Snap:
        snap = self._snaps[min(self._i, len(self._snaps) - 1)]
        self._i += 1
        return snap


def test_streams_max_tokens_and_returns_a_result() -> None:
    backend = FakeBackend(vocab_size=32)

    result = generate_plain(backend, [3], max_tokens=10, eos=frozenset(), ctx=CTX, guard=Snap)

    assert isinstance(result, GenerationResult)
    assert len(result.tokens) == 10
    assert not result.stopped_on_eos
    assert result.prefill_ms >= 0.0


def test_tokens_follow_greedy_argmax() -> None:
    backend = FakeBackend(vocab_size=32)

    result = generate_plain(backend, [3], max_tokens=4, eos=frozenset(), ctx=CTX, guard=Snap)

    expected = [(3 * STEP) % 32]
    for _ in range(3):
        expected.append((expected[-1] * STEP) % 32)
    assert list(result.tokens) == expected


def test_one_row_per_decode_window_not_per_request() -> None:
    backend = FakeBackend(vocab_size=32)

    result = generate_plain(backend, [3], max_tokens=10, eos=frozenset(), ctx=CTX, guard=Snap)

    # first token comes from prefill (not a window); each later token is one window
    assert len(result.rows) == 9
    df = frame_from_rows(result.rows)
    assert list(df["window_idx"]) == list(range(9))
    assert list(df["token_pos"]) == list(range(1, 10))
    assert set(df["k_proposed"]) == {0}
    assert set(df["n_accepted"]) == {0}
    assert set(df["draft_ms"]) == {0.0}
    assert (df["verify_ms"] >= 0.0).all()
    assert (df["window_ms"] >= df["verify_ms"]).all()


def test_rows_carry_context_and_are_plain_tuples() -> None:
    backend = FakeBackend(vocab_size=32)

    result = generate_plain(backend, [3], max_tokens=3, eos=frozenset(), ctx=CTX, guard=Snap)

    row = result.rows[0]
    assert type(row) is tuple
    assert len(row) == len(FIELD_NAMES)
    assert row[FIELD_NAMES.index("run_id")] == "run-a"
    assert row[FIELD_NAMES.index("workload_tag")] == "code"
    assert row[FIELD_NAMES.index("prompt_id")] == "code-001"
    assert row[FIELD_NAMES.index("rep")] == 0


def test_page_ins_is_the_delta_across_the_window_and_pressure_is_the_max() -> None:
    backend = FakeBackend(vocab_size=32)
    guard = ScriptedGuard(
        Snap(page_ins=100),  # before window 0
        Snap(page_ins=100),  # after window 0
        Snap(page_ins=100, mem_pressure=0),  # before window 1
        Snap(page_ins=103, mem_pressure=2, thermal_level=1),  # after window 1
    )

    result = generate_plain(backend, [3], max_tokens=3, eos=frozenset(), ctx=CTX, guard=guard)

    df = frame_from_rows(result.rows)
    assert list(df["page_ins"]) == [0, 3]
    assert list(df["mem_pressure"]) == [0, 2]
    assert list(df["thermal_level"]) == [0, 1]


def test_stops_on_eos_and_emits_no_row_for_the_eos_window_after_it() -> None:
    backend = FakeBackend(vocab_size=32)
    first = (3 * STEP) % 32
    second = (first * STEP) % 32

    result = generate_plain(
        backend, [3], max_tokens=50, eos=frozenset({second}), ctx=CTX, guard=Snap
    )

    assert list(result.tokens) == [first, second]
    assert result.stopped_on_eos
    assert len(result.rows) == 1


def test_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError):
        generate_plain(FakeBackend(), [], max_tokens=1, eos=frozenset(), ctx=CTX, guard=Snap)
