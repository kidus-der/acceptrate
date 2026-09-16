"""model/sampling.py — temperature softmax, categorical sampling, residual and accept prob.

Pure numpy; property-tested because every formula here has an invariant
(a distribution sums to 1, a probability sits in [0, 1]) that must hold for
any input, not just the handful of vectors a unit test would think of.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from acceptrate.model.sampling import (
    accept_probability,
    residual_distribution,
    sample,
    softmax_with_temperature,
)

LOGITS = hnp.arrays(
    dtype=np.float32,
    shape=st.integers(min_value=1, max_value=64),
    elements=st.floats(min_value=-50.0, max_value=50.0, width=32),
)
TEMPERATURES = st.floats(min_value=1e-6, max_value=10.0, allow_nan=False)
PROBS = LOGITS.map(lambda z: softmax_with_temperature(z, 1.0))


@settings(max_examples=200, deadline=None)
@given(logits=LOGITS, temperature=TEMPERATURES)
def test_softmax_is_a_float64_distribution(logits, temperature) -> None:
    probs = softmax_with_temperature(logits, temperature)

    assert probs.dtype == np.float64
    assert probs.shape == logits.shape
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0, atol=1e-9)


def test_softmax_preserves_argmax_and_sharpens_as_temperature_falls() -> None:
    logits = np.array([1.0, 3.0, 2.0], dtype=np.float32)

    hot = softmax_with_temperature(logits, 2.0)
    cold = softmax_with_temperature(logits, 0.1)

    assert int(np.argmax(hot)) == int(np.argmax(cold)) == 1
    assert cold[1] > hot[1]


def test_softmax_is_stable_for_huge_logits_and_tiny_temperature() -> None:
    logits = np.array([1000.0, -1000.0, 999.0], dtype=np.float32)

    probs = softmax_with_temperature(logits, 1e-6)

    assert np.isfinite(probs).all()
    assert probs[0] == pytest.approx(1.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
def test_softmax_rejects_non_positive_temperature(temperature: float) -> None:
    with pytest.raises(ValueError):
        softmax_with_temperature(np.zeros(3, dtype=np.float32), temperature)


@settings(max_examples=200, deadline=None)
@given(probs=PROBS, seed=st.integers(min_value=0, max_value=2**31))
def test_sample_returns_an_index_with_positive_probability(probs, seed) -> None:
    token = sample(probs, np.random.default_rng(seed))

    assert isinstance(token, int)
    assert 0 <= token < len(probs)
    assert probs[token] > 0.0


def test_sample_frequencies_track_the_distribution() -> None:
    probs = np.array([0.1, 0.6, 0.3])
    rng = np.random.default_rng(0)

    counts = np.bincount([sample(probs, rng) for _ in range(20_000)], minlength=3)

    assert np.allclose(counts / counts.sum(), probs, atol=0.02)


def test_sample_never_picks_a_zero_probability_token() -> None:
    probs = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(1)

    assert all(sample(probs, rng) == 1 for _ in range(1000))


@settings(max_examples=200, deadline=None)
@given(logits=LOGITS, temperature=TEMPERATURES, seed=st.integers(min_value=0, max_value=2**16))
def test_residual_is_a_distribution_for_any_target_and_draft(logits, temperature, seed) -> None:
    p_target = softmax_with_temperature(logits, temperature)
    q_draft = softmax_with_temperature(np.random.default_rng(seed).normal(size=logits.shape), 1.0)

    residual = residual_distribution(p_target, q_draft)

    assert residual.shape == p_target.shape
    assert np.all(residual >= 0.0)
    assert np.isclose(residual.sum(), 1.0, atol=1e-9)


def test_residual_drops_mass_where_the_draft_over_proposed() -> None:
    p_target = np.array([0.5, 0.3, 0.2])
    q_draft = np.array([0.1, 0.6, 0.3])

    residual = residual_distribution(p_target, q_draft)

    assert residual[1] == 0.0 and residual[2] == 0.0
    assert residual[0] == pytest.approx(1.0)


def test_residual_falls_back_to_target_when_draft_equals_target() -> None:
    p_target = np.array([0.5, 0.3, 0.2])

    residual = residual_distribution(p_target, p_target.copy())

    assert np.array_equal(residual, p_target)


@settings(max_examples=200, deadline=None)
@given(
    p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    q=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_accept_probability_is_a_probability(p: float, q: float) -> None:
    assert 0.0 <= accept_probability(p, q) <= 1.0


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        (0.5, 0.25, 1.0),  # target likes it more: always accept
        (0.25, 0.5, 0.5),  # p / q
        (0.0, 0.5, 0.0),  # target never produces it
        (0.3, 0.0, 1.0),  # draft could not have proposed it; p / 0 clamps to 1
        (0.0, 0.0, 0.0),  # neither side has mass: reject
    ],
)
def test_accept_probability_edge_cases(p: float, q: float, expected: float) -> None:
    assert accept_probability(p, q) == pytest.approx(expected)
