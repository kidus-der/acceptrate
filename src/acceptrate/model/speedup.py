"""Layer 1: the closed-form speculative-decoding speedup and the argmax over K.

    speedup(alpha, K, c) = (1 - alpha^(K+1)) / ((1 - alpha) * (K*c + 1))

alpha: per-token acceptance rate; K: draft depth; c: draft cost / target cost.
Pure arithmetic, microseconds, no training. If alpha were known this layer
would be the entire product (brief tab 4).
"""

from __future__ import annotations

K_MAX = 10
"""Largest draft depth the optimiser considers."""

_ALPHA_CAP = 0.999999
"""alpha == 1 is the limit (K+1)/(Kc+1); computed via the geometric sum, not the ratio."""

_BREAK_EVEN_STEP = 0.001


def _validate(alpha: float, k: int, c: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if k < 0:
        raise ValueError(f"K must be non-negative, got {k}")
    if c < 0.0:
        raise ValueError(f"cost ratio must be non-negative, got {c}")


def expected_tokens(alpha: float, k: int) -> float:
    """Expected tokens harvested per round: 1 + alpha + ... + alpha^K."""
    if alpha >= _ALPHA_CAP:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    _validate(alpha, k, c)
    if k == 0:
        return 1.0
    return expected_tokens(alpha, k) / (k * c + 1.0)


def best_k(alpha: float, c: float, k_max: int = K_MAX) -> int:
    """The draft depth that maximises speedup; 0 when no depth beats plain decoding."""
    _validate(alpha, 0, c)
    best, best_value = 0, 1.0
    for k in range(1, k_max + 1):
        value = speedup(alpha, k, c)
        if value > best_value:
            best, best_value = k, value
    return best


def break_even_alpha(k: int, c: float) -> float | None:
    """Smallest alpha < 1 at which depth K stops losing, or None if it never does.

    alpha == 1 is excluded: reaching parity only with every draft accepted is
    not a break-even any real workload can rely on.
    """
    _validate(0.0, k, c)
    steps = int(1.0 / _BREAK_EVEN_STEP)
    for i in range(steps):
        alpha = i * _BREAK_EVEN_STEP
        if speedup(alpha, k, c) >= 1.0:
            return alpha
    return None
