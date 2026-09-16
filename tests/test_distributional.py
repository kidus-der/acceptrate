"""verify/distributional.py — total variation, empirical laws and the two-sample TV bound.

The end-to-end check against the sampled runtimes lives in
tests/test_distributional_fakes.py; this file pins the pure functions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acceptrate.model.sampling import softmax_with_temperature
from acceptrate.verify.distributional import (
    DistributionalReport,
    compare_next_token_distributions,
    empirical_distribution,
    passes,
    total_variation,
    tv_bound,
)


def test_total_variation_of_identical_laws_is_zero() -> None:
    p = np.array([0.2, 0.3, 0.5])

    assert total_variation(p, p.copy()) == 0.0


def test_total_variation_of_disjoint_laws_is_one() -> None:
    assert total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_total_variation_is_half_the_l1_distance() -> None:
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.1, 0.6, 0.3])

    assert total_variation(p, q) == pytest.approx(0.5 * (0.4 + 0.3 + 0.1))


def test_total_variation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        total_variation(np.array([1.0]), np.array([0.5, 0.5]))


def test_empirical_distribution_counts_and_normalises() -> None:
    dist = empirical_distribution([0, 2, 2, 3], vocab_size=5)

    assert dist.tolist() == [0.25, 0.0, 0.5, 0.25, 0.0]


@pytest.mark.parametrize("samples", [[], [5], [-1]])
def test_empirical_distribution_rejects_empty_or_out_of_range(samples: list[int]) -> None:
    with pytest.raises(ValueError):
        empirical_distribution(samples, vocab_size=5)


def test_compare_reports_tv_and_sample_counts() -> None:
    report = compare_next_token_distributions([0, 0, 1, 1], [0, 1, 1, 1, 1, 1], vocab_size=2)

    assert report == DistributionalReport(tv=pytest.approx(1 / 3), n_plain=4, n_spec=6)


def test_passes_compares_tv_to_the_bound() -> None:
    assert passes(DistributionalReport(tv=0.05, n_plain=10, n_spec=10), bound=0.1)
    assert not passes(DistributionalReport(tv=0.15, n_plain=10, n_spec=10), bound=0.1)


# --- the bound --------------------------------------------------------------


def test_tv_bound_matches_the_bretagnolle_huber_carol_formula() -> None:
    n, k, confidence = 4000, 32, 0.99
    alpha = 1.0 - confidence
    per_side = math.sqrt((k * math.log(2.0) + math.log(2.0 / alpha)) / (2.0 * n))

    assert tv_bound(n, n, k, confidence) == pytest.approx(2.0 * per_side)
    assert tv_bound(n, n, k, confidence) == pytest.approx(0.1172, abs=1e-4)


def test_tv_bound_shrinks_with_samples_and_grows_with_support_and_confidence() -> None:
    assert tv_bound(8000, 8000, 32) < tv_bound(4000, 4000, 32)
    assert tv_bound(4000, 4000, 64) > tv_bound(4000, 4000, 32)
    assert tv_bound(4000, 4000, 32, confidence=0.999) > tv_bound(4000, 4000, 32, confidence=0.99)
    assert tv_bound(4000, 400, 32) > tv_bound(4000, 4000, 32)


@pytest.mark.parametrize(
    ("n_plain", "n_spec", "support", "confidence"),
    [(0, 10, 4, 0.99), (10, 0, 4, 0.99), (10, 10, 0, 0.99), (10, 10, 4, 1.0), (10, 10, 4, 0.0)],
)
def test_tv_bound_rejects_degenerate_inputs(n_plain, n_spec, support, confidence) -> None:
    with pytest.raises(ValueError):
        tv_bound(n_plain, n_spec, support, confidence)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000), scale=st.floats(0.1, 4.0))
def test_two_samples_of_the_same_law_fall_under_the_bound(seed: int, scale: float) -> None:
    rng = np.random.default_rng(seed)
    vocab = 32
    law = softmax_with_temperature(rng.normal(size=vocab) * scale, 1.0)
    n = 2000

    a = rng.choice(vocab, size=n, p=law)
    b = rng.choice(vocab, size=n, p=law)
    report = compare_next_token_distributions(a, b, vocab)

    assert passes(report, tv_bound(n, n, vocab))
