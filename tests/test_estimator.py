"""model/estimator.py — Layer 2: an EWMA acceptance tracker with a warmup guard.

Tracks accepted / proposed per window. Blind for the first few windows (the
cold-start prior covers that in P6); afterwards it adapts inside a response
as text moves between easy and hard stretches.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from acceptrate.model.estimator import AcceptanceEstimator


def test_starts_unready_with_the_prior_as_its_estimate() -> None:
    est = AcceptanceEstimator(prior=0.6, half_life_windows=4, warmup_windows=3)

    assert not est.ready
    assert est.alpha == 0.6
    assert est.windows == 0


def test_becomes_ready_after_warmup_windows() -> None:
    est = AcceptanceEstimator(prior=0.6, half_life_windows=4, warmup_windows=2)

    est = est.update(k_proposed=4, n_accepted=4)
    assert not est.ready
    est = est.update(k_proposed=4, n_accepted=2)
    assert est.ready


def test_update_returns_a_new_estimator_and_leaves_the_old_one_untouched() -> None:
    before = AcceptanceEstimator(prior=0.5, half_life_windows=4, warmup_windows=0)

    after = before.update(k_proposed=4, n_accepted=4)

    assert before.alpha == 0.5
    assert after.alpha > 0.5


def test_converges_towards_the_observed_rate() -> None:
    est = AcceptanceEstimator(prior=0.2, half_life_windows=3, warmup_windows=0)

    for _ in range(40):
        est = est.update(k_proposed=4, n_accepted=3)

    assert est.alpha == pytest.approx(0.75, abs=0.01)


def test_half_life_means_half_the_weight_after_that_many_windows() -> None:
    est = AcceptanceEstimator(prior=0.0, half_life_windows=2, warmup_windows=0)

    est = est.update(k_proposed=4, n_accepted=4)  # observation 1.0
    est = est.update(k_proposed=4, n_accepted=4)

    # after two windows at rate 1.0 from a prior of 0.0, weight on the prior is 0.5
    assert est.alpha == pytest.approx(0.5, abs=1e-9)


def test_alpha_is_per_token_so_a_rejection_counts_as_one_examined_token() -> None:
    est = AcceptanceEstimator(prior=0.0, half_life_windows=1e-9, warmup_windows=0)

    est = est.update(k_proposed=4, n_accepted=2)  # 2 accepted, 1 rejected, 1 never examined

    assert est.alpha == pytest.approx(2 / 3)


def test_zero_proposed_window_is_ignored() -> None:
    est = AcceptanceEstimator(prior=0.6, half_life_windows=4, warmup_windows=0)

    same = est.update(k_proposed=0, n_accepted=0)

    assert same.alpha == 0.6
    assert same.windows == 0


def test_rejects_impossible_counts() -> None:
    est = AcceptanceEstimator(prior=0.6, half_life_windows=4, warmup_windows=0)

    with pytest.raises(ValueError):
        est.update(k_proposed=4, n_accepted=5)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_rejects_prior_outside_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError):
        AcceptanceEstimator(prior=bad, half_life_windows=4, warmup_windows=0)


@given(
    obs=st.lists(st.tuples(st.integers(1, 8), st.integers(0, 8)), min_size=1, max_size=50),
    prior=st.floats(0.0, 1.0),
)
def test_alpha_always_stays_in_the_unit_interval(obs, prior) -> None:
    est = AcceptanceEstimator(prior=prior, half_life_windows=4, warmup_windows=0)

    for k, n in obs:
        est = est.update(k_proposed=k, n_accepted=min(n, k))

    assert 0.0 <= est.alpha <= 1.0
