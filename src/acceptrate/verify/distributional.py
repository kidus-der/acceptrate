"""Distributional losslessness above temperature 0 (brief tab 6).

Greedy equivalence is an exact token match. Above T = 0 the claim becomes
"speculative sampling draws from the same next-token law as plain
sampling", which cannot be checked on one generation: instead the same
prefix is sampled N times both ways and the two empirical laws are
compared in total variation against a bound that two samples of *one* law
would respect with the stated confidence.

The bound (`tv_bound`) is Bretagnolle-Huber-Carol (van der Vaart & Wellner,
*Weak Convergence and Empirical Processes*, Prop. A.6.6): for an empirical
law P_n of n draws from P supported on k atoms,

    Pr( ||P_n - P||_1 >= eps ) <= 2^k exp(-n eps^2 / 2).

With TV = ||.||_1 / 2 this is Pr(TV(P_n, P) >= d) <= 2^k exp(-2 n d^2).
Solving 2^k exp(-2 n d^2) = alpha/2 for each sample and adding the two
radii (triangle inequality through P) gives, with probability >= 1 - alpha,

    TV(P_plain, P_spec) <= d(n_plain) + d(n_spec),
    d(n) = sqrt( (k ln 2 + ln(2 / alpha)) / (2 n) ).

`k` is the support size of the underlying law. Passing the full vocabulary
makes 2^k vacuous for real models, so callers pass the effective support:
the number of distinct tokens observed across both samples. Tokens with
mass below ~1/n are invisible to either sample and contribute negligible
TV, which is the approximation being made; it is stated here, not hidden.

Framework-free: token samples in, a report and a number out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from acceptrate.backend.protocol import Backend
from acceptrate.runtime.engine import GenerationContext
from acceptrate.runtime.sampled import generate_plain_sampled, generate_speculative_sampled

Probs = npt.NDArray[np.float64]

DEFAULT_CONFIDENCE = 0.99


@dataclass(frozen=True)
class DistributionalReport:
    tv: float
    """Total variation between the plain and speculative empirical next-token laws."""
    n_plain: int
    n_spec: int


def total_variation(p: npt.ArrayLike, q: npt.ArrayLike) -> float:
    """0.5 * sum |p - q| — the largest probability disagreement over any event."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError(f"distributions differ in shape: {p_arr.shape} vs {q_arr.shape}")
    return float(0.5 * np.abs(p_arr - q_arr).sum())


def empirical_distribution(samples: Sequence[int] | npt.ArrayLike, vocab_size: int) -> Probs:
    """Relative frequency of each token id over `samples`."""
    ids = np.asarray(samples, dtype=np.int64)
    if ids.size == 0:
        raise ValueError("cannot build an empirical distribution from zero samples")
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError(f"token ids must lie in [0, {vocab_size}), got {ids.min()}..{ids.max()}")
    return np.bincount(ids, minlength=vocab_size) / ids.size


def compare_next_token_distributions(
    plain_samples: Sequence[int] | npt.ArrayLike,
    spec_samples: Sequence[int] | npt.ArrayLike,
    vocab_size: int,
) -> DistributionalReport:
    p_plain = empirical_distribution(plain_samples, vocab_size)
    p_spec = empirical_distribution(spec_samples, vocab_size)
    return DistributionalReport(
        tv=total_variation(p_plain, p_spec),
        n_plain=int(np.asarray(plain_samples).size),
        n_spec=int(np.asarray(spec_samples).size),
    )


def _bhc_radius(n: int, support: int, alpha: float) -> float:
    return math.sqrt((support * math.log(2.0) + math.log(2.0 / alpha)) / (2.0 * n))


def tv_bound(
    n_plain: int, n_spec: int, vocab_support: int, confidence: float = DEFAULT_CONFIDENCE
) -> float:
    """TV two empirical laws of one distribution stay under with the given confidence.

    Bretagnolle-Huber-Carol per sample, summed by the triangle inequality;
    see the module docstring for the derivation.
    """
    if n_plain < 1 or n_spec < 1:
        raise ValueError(f"both sample sizes must be >= 1, got {n_plain} and {n_spec}")
    if vocab_support < 1:
        raise ValueError(f"support size must be >= 1, got {vocab_support}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence}")
    alpha = 1.0 - confidence
    return _bhc_radius(n_plain, vocab_support, alpha) + _bhc_radius(n_spec, vocab_support, alpha)


def passes(report: DistributionalReport, bound: float) -> bool:
    return report.tv <= bound


# --- driving the sampled runtimes -------------------------------------------

CHECK_CONTEXT = GenerationContext(
    run_id="distributional", workload_tag="prose", prompt_id="prefix", rep=0
)
"""Trace rows from the check are discarded; the context only satisfies the runtime."""

FIRST_VERIFIED_POSITION = 1
"""Token index 0 comes straight from the target's prefill logits in both paths and
would compare trivially; index 1 is the first token that passes accept/reject."""


@dataclass(frozen=True)
class _QuietGuard:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


def _observed_support(plain: Sequence[int], spec: Sequence[int]) -> int:
    return len(set(plain) | set(spec))


def run_distributional_check(
    target: Backend,
    draft: Backend,
    prompt_tokens: Sequence[int],
    k: int,
    temperature: float,
    n_samples: int,
    seed: int,
    vocab_size: int,
    *,
    position: int = FIRST_VERIFIED_POSITION,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[DistributionalReport, float]:
    """Sample `prompt_tokens` n times plain and n times speculatively; compare token `position`.

    The two paths get independent generators derived from `seed`, so the
    same seed reproduces the report and the two samples are independent.
    The bound uses the support observed across both samples.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    max_tokens = position + 1
    eos: frozenset[int] = frozenset()
    rng_plain = np.random.default_rng([seed, 0])
    rng_spec = np.random.default_rng([seed, 1])
    plain = [
        generate_plain_sampled(
            target, prompt_tokens, max_tokens, eos, CHECK_CONTEXT, _QuietGuard,
            temperature, rng_plain,
        ).tokens[position]
        for _ in range(n_samples)
    ]  # fmt: skip
    spec = [
        generate_speculative_sampled(
            target, draft, prompt_tokens, max_tokens, eos, CHECK_CONTEXT, _QuietGuard,
            k, temperature, rng_spec,
        ).tokens[position]
        for _ in range(n_samples)
    ]  # fmt: skip
    report = compare_next_token_distributions(plain, spec, vocab_size)
    bound = tv_bound(n_samples, n_samples, _observed_support(plain, spec), confidence)
    return report, bound
