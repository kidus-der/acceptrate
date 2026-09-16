"""Layer 2: the running acceptance estimate.

An exponentially weighted moving average of accepted / proposed per draft
window. It is the workhorse of the adaptive runtime: it adapts *inside* a
response as text moves between easy and hard stretches. It is blind for the
first `warmup_windows`; until then `alpha` is the prior (P6's cold-start
predictor supplies a better one).

Immutable: `update` returns a new estimator. Pure Python, no framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AcceptanceEstimator:
    prior: float
    half_life_windows: float
    warmup_windows: int
    alpha: float = float("nan")
    windows: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.prior <= 1.0:
            raise ValueError(f"prior must be in [0, 1], got {self.prior}")
        if self.half_life_windows <= 0:
            raise ValueError("half_life_windows must be positive")
        if self.warmup_windows < 0:
            raise ValueError("warmup_windows must be non-negative")
        if math.isnan(self.alpha):
            object.__setattr__(self, "alpha", self.prior)

    @property
    def ready(self) -> bool:
        return self.windows >= self.warmup_windows and self.windows > 0

    @property
    def decay(self) -> float:
        """Weight kept from the previous estimate per window: 0.5 ** (1 / half_life)."""
        return 0.5 ** (1.0 / self.half_life_windows)

    def update(self, k_proposed: int, n_accepted: int) -> AcceptanceEstimator:
        if k_proposed < 0 or n_accepted < 0 or n_accepted > k_proposed:
            raise ValueError(f"impossible window: {n_accepted} accepted of {k_proposed} proposed")
        if k_proposed == 0:
            return self
        observed = n_accepted / k_proposed
        decay = self.decay
        alpha = decay * self.alpha + (1.0 - decay) * observed
        return replace(self, alpha=min(1.0, max(0.0, alpha)), windows=self.windows + 1)
