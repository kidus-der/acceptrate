"""runtime/sampled.py — temperature > 0 plain and speculative (rejection-sampling) generation.

Contract: same trace-row shape and cache discipline as runtime/speculative.py;
at temperature -> 0 both sampled paths collapse to their greedy twins exactly.
The distributional check itself lives in tests/test_distributional.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pytest

from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.runtime.sampled import generate_plain_sampled, generate_speculative_sampled
from acceptrate.runtime.speculative import generate_speculative
from acceptrate.trace.schema import FIELD_NAMES, frame_from_rows
from tests.fakes import FakeBackend
from tests.fakes_prob import ProbFake

CTX = GenerationContext(run_id="r", workload_tag="code", prompt_id="code-001", rep=0)
V = 32
COLD = 1e-6
"""A temperature at which softmax is one-hot to float64 precision: sampling == argmax."""


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


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _plain_sampled(backend, prompt, max_tokens, temperature, seed=0, eos=frozenset()):
    return generate_plain_sampled(
        backend, prompt, max_tokens, eos, CTX, Snap, temperature, _rng(seed)
    )


def _spec_sampled(target, draft, prompt, max_tokens, k, temperature, seed=0, eos=frozenset()):
    return generate_speculative_sampled(
        target, draft, prompt, max_tokens, eos, CTX, Snap, k, temperature, _rng(seed)
    )


# --- temperature -> 0 reproduces greedy exactly -----------------------------


def test_cold_plain_sampled_equals_greedy_plain_on_one_hot_fake() -> None:
    greedy = generate_plain(FakeBackend(V), [3], 20, frozenset(), CTX, Snap)

    sampled = _plain_sampled(FakeBackend(V), [3], 20, COLD)

    assert sampled.tokens == greedy.tokens


def test_cold_plain_sampled_equals_greedy_plain_on_prob_fake() -> None:
    greedy = generate_plain(ProbFake(V, seed=0), [3], 40, frozenset(), CTX, Snap)

    sampled = _plain_sampled(ProbFake(V, seed=0), [3], 40, COLD)

    assert sampled.tokens == greedy.tokens


@pytest.mark.parametrize("k", [1, 3, 4])
def test_cold_speculative_sampled_equals_greedy_speculative_on_partial_draft(k: int) -> None:
    greedy = generate_speculative(
        FakeBackend(V), PartialFake(V), [3], 40, frozenset(), CTX, Snap, k=k
    )

    sampled = _spec_sampled(FakeBackend(V), PartialFake(V), [3], 40, k, COLD)

    assert sampled.tokens == greedy.tokens
    assert [r[4] for r in sampled.rows] == [r[4] for r in greedy.rows]  # n_accepted per window


def test_cold_speculative_sampled_equals_greedy_speculative_on_prob_draft() -> None:
    target = ProbFake(V, seed=0)
    draft = ProbFake.perturbed(target, seed=1, noise=0.5)
    greedy = generate_speculative(
        ProbFake(V, seed=0), ProbFake.perturbed(target, seed=1, noise=0.5),
        [3], 60, frozenset(), CTX, Snap, k=4,
    )  # fmt: skip

    sampled = _spec_sampled(target, draft, [3], 60, 4, COLD)

    assert sampled.tokens == greedy.tokens


# --- trace rows -------------------------------------------------------------


def test_rows_have_schema_arity_one_per_window_and_sane_counts() -> None:
    target = ProbFake(V, seed=0)
    draft = ProbFake.perturbed(target, seed=1, noise=1.0)

    result = _spec_sampled(target, draft, [3], 40, 4, 1.0)

    df = frame_from_rows(result.rows)
    assert type(result.rows[0]) is tuple and len(result.rows[0]) == len(FIELD_NAMES)
    assert list(df["window_idx"]) == list(range(len(result.rows)))
    assert set(df["k_proposed"]) == {4}
    assert set(df["n_accepted"]) <= {0, 1, 2, 3, 4}
    positions = list(df["token_pos"])
    assert positions[0] == 1
    assert all(b > a for a, b in pairwise(positions))
    assert (df["window_ms"] >= df["draft_ms"] + df["verify_ms"]).all()
    assert set(df["workload_tag"]) == {"code"}
    # committed tokens = prefill token + sum over windows of (accepted + 1)
    assert len(result.tokens) == min(40, 1 + int((df["n_accepted"] + 1).sum()))


def test_plain_sampled_rows_are_k_zero_with_nothing_accepted() -> None:
    result = _plain_sampled(ProbFake(V, seed=0), [3], 12, 1.0)

    df = frame_from_rows(result.rows)
    assert len(result.rows) == 11
    assert set(df["k_proposed"]) == {0}
    assert set(df["n_accepted"]) == {0}
    assert (df["draft_ms"] == 0.0).all()


# --- cache discipline -------------------------------------------------------


def test_caches_hold_exactly_the_committed_prefix() -> None:
    target = ProbFake(V, seed=0)
    draft = ProbFake.perturbed(target, seed=1, noise=1.0)

    result = _spec_sampled(target, draft, [3, 4], 25, 4, 1.0)

    committed = [3, 4, *result.tokens]
    assert target.tokens[: len(committed) - 1] == tuple(committed[:-1])
    assert "trim" in target.calls
    assert draft.tokens == tuple(committed[: len(draft.tokens)])


def test_draft_costs_k_forward_calls_per_window_and_target_one() -> None:
    target = ProbFake(V, seed=0)
    draft = ProbFake(V, seed=9)

    result = _spec_sampled(target, draft, [3], 9, 4, 1.0)

    windows = len(result.rows)
    assert target.calls.count("verify") == windows
    assert draft.calls.count("verify") + draft.calls.count("decode_step") == windows * 4


# --- stopping and validation ------------------------------------------------


def test_stops_at_max_tokens_exactly_even_mid_window() -> None:
    target = ProbFake(V, seed=0)

    result = _spec_sampled(target, ProbFake.perturbed(target, 1, 0.1), [3], 10, 4, 1.0)

    assert len(result.tokens) == 10
    assert not result.stopped_on_eos


def test_stops_at_eos_and_keeps_the_eos_token() -> None:
    target = ProbFake(V, seed=0)
    draft = ProbFake.perturbed(target, 1, 0.5)
    unbounded = _spec_sampled(
        ProbFake(V, seed=0), ProbFake.perturbed(target, 1, 0.5), [3], 50, 4, 1.0
    )
    eos_token = unbounded.tokens[7]
    first_hit = unbounded.tokens.index(eos_token)

    result = _spec_sampled(target, draft, [3], 50, 4, 1.0, eos=frozenset({eos_token}))

    assert result.tokens == unbounded.tokens[: first_hit + 1]
    assert result.stopped_on_eos


def test_same_seed_reproduces_and_different_seed_diverges() -> None:
    def run(seed: int):
        target = ProbFake(V, seed=0)
        return _spec_sampled(
            target, ProbFake.perturbed(target, 1, 0.5), [3], 40, 4, 1.0, seed
        ).tokens

    assert run(5) == run(5)
    assert run(5) != run(6)


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError):
        _spec_sampled(ProbFake(V), ProbFake(V, seed=1), [3], 5, k, 1.0)


def test_rejects_empty_prompt_and_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        _plain_sampled(ProbFake(V), [], 5, 1.0)
    with pytest.raises(ValueError):
        _spec_sampled(ProbFake(V), ProbFake(V, seed=1), [], 5, 2, 1.0)
    with pytest.raises(ValueError):
        _plain_sampled(ProbFake(V), [3], 5, 0.0)
