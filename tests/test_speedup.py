"""model/speedup.py — the governing equation and the argmax over K.

speedup(alpha, K, c) = (1 - alpha^(K+1)) / ((1 - alpha) * (K*c + 1))
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from acceptrate.model.speedup import K_MAX, best_k, break_even_alpha, speedup

ALPHA = st.floats(min_value=0.0, max_value=0.99)
COST = st.floats(min_value=0.001, max_value=1.0)
K = st.integers(min_value=0, max_value=K_MAX)


def test_k_zero_is_plain_decoding() -> None:
    assert speedup(0.7, 0, 0.16) == 1.0


def test_matches_the_briefs_worked_example() -> None:
    # brief calculator default: alpha 0.70, c 0.16. The caption says "1.71x at
    # K = 4"; the equation gives 1.71x at K = 3 and 1.69x at K = 4. The
    # equation wins: the value is right, the depth in the caption is not.
    assert speedup(0.70, 3, 0.16) == pytest.approx(1.71, abs=0.01)
    assert speedup(0.70, 4, 0.16) == pytest.approx(1.69, abs=0.01)
    assert best_k(0.70, 0.16) == 3


def test_alpha_zero_always_loses_for_any_positive_k() -> None:
    assert speedup(0.0, 4, 0.16) == pytest.approx(1 / (4 * 0.16 + 1))


def test_alpha_one_is_the_limit_k_plus_one_over_cost() -> None:
    assert speedup(1.0, 4, 0.16) == pytest.approx(5 / (4 * 0.16 + 1))


@given(alpha=ALPHA, k=K, c=COST)
def test_speedup_is_finite_and_positive(alpha: float, k: int, c: float) -> None:
    value = speedup(alpha, k, c)

    assert math.isfinite(value)
    assert value > 0


@given(k=st.integers(min_value=1, max_value=K_MAX), c=COST)
def test_speedup_is_monotone_in_alpha(k: int, c: float) -> None:
    grid = [i / 20 for i in range(20)]

    values = [speedup(a, k, c) for a in grid]

    assert all(x <= y + 1e-12 for x, y in pairwise(values))


@given(alpha=ALPHA, k=st.integers(min_value=1, max_value=K_MAX))
def test_free_drafts_never_lose(alpha: float, k: int) -> None:
    assert speedup(alpha, k, 0.0) >= 1.0 - 1e-12


@given(alpha=ALPHA, c=COST)
def test_best_k_is_the_argmax_over_the_grid(alpha: float, c: float) -> None:
    k = best_k(alpha, c)

    assert 0 <= k <= K_MAX
    assert all(speedup(alpha, k, c) >= speedup(alpha, j, c) - 1e-12 for j in range(K_MAX + 1))


def test_best_k_is_zero_when_no_depth_wins() -> None:
    assert best_k(0.1, 0.5) == 0


def test_break_even_alpha_is_where_speedup_crosses_one() -> None:
    alpha = break_even_alpha(k=4, c=0.16)

    assert alpha is not None
    assert speedup(alpha, 4, 0.16) == pytest.approx(1.0, abs=0.02)


def test_break_even_is_none_when_drafting_is_too_expensive() -> None:
    assert break_even_alpha(k=1, c=1.0) is None


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_alpha_outside_unit_interval_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        speedup(bad, 4, 0.16)


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValueError):
        speedup(0.5, 4, -0.1)
