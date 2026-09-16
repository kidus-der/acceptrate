"""Temperature sampling primitives for the sampled (T > 0) speculative loop.

The Leviathan et al. / Chen et al. rejection-sampling verify step needs
exactly four operations, all pure numpy and framework-free:

    p = softmax(logits / T)                    target or draft next-token law
    x ~ p                                      one categorical draw
    accept x with prob min(1, p(x) / q(x))     draft token x proposed under q
    on rejection, x ~ normalise(max(0, p - q)) the residual distribution

Everything returns float64 so that ratios of small probabilities stay
finite. The hot loop calls these per position; keep them allocation-light.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Probs = npt.NDArray[np.float64]
"""Shape (vocab_size,), non-negative, sums to 1."""

RESIDUAL_MASS_FLOOR = 1e-12
"""Below this residual mass p ~= q everywhere; sample from p itself instead."""


def softmax_with_temperature(logits: npt.ArrayLike, temperature: float) -> Probs:
    """softmax(logits / temperature) in float64, shifted by the max for stability."""
    if not temperature > 0.0:  # also rejects NaN
        raise ValueError(f"temperature must be > 0 for sampling, got {temperature}")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def sample(probs: Probs, rng: np.random.Generator) -> int:
    """One categorical draw by inverse CDF; never returns a zero-probability index."""
    cdf = np.cumsum(probs)
    u = rng.random() * cdf[-1]
    index = int(np.searchsorted(cdf, u, side="right"))
    return min(index, len(probs) - 1)


def residual_distribution(p_target: Probs, q_draft: Probs) -> Probs:
    """normalise(max(0, p - q)); falls back to p_target when the residual mass vanishes."""
    residual = np.maximum(p_target - q_draft, 0.0)
    mass = residual.sum()
    if mass <= RESIDUAL_MASS_FLOOR:
        return np.array(p_target, dtype=np.float64, copy=True)
    return residual / mass


def accept_probability(p_target_token: float, q_draft_token: float) -> float:
    """min(1, p / q) for the drafted token; q == 0 clamps to 1 and p == 0 rejects."""
    if p_target_token <= 0.0:
        return 0.0
    if q_draft_token <= 0.0:
        return 1.0
    return min(1.0, p_target_token / q_draft_token)
