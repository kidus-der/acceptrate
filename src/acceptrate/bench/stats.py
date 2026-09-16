"""Summary statistics for traces. Median and IQR only — one throttle event ruins a mean."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from acceptrate.trace.schema import FIELD_NAMES, WindowRow

_N_ACCEPTED = FIELD_NAMES.index("n_accepted")
_WINDOW_MS = FIELD_NAMES.index("window_ms")
MS_PER_S = 1000.0


@dataclass(frozen=True)
class MedianIQR:
    median: float
    q1: float
    q3: float
    n: int

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def median_iqr(values: Sequence[float]) -> MedianIQR:
    if not values:
        raise ValueError("median_iqr needs at least one value")
    ordered = sorted(values)
    mid = len(ordered) // 2
    lower = ordered[:mid]
    upper = ordered[mid + 1 :] if len(ordered) % 2 else ordered[mid:]
    q1 = median(lower) if lower else ordered[0]
    q3 = median(upper) if upper else ordered[-1]
    return MedianIQR(median=float(median(ordered)), q1=float(q1), q3=float(q3), n=len(ordered))


def throughput_tok_s(rows: Sequence[WindowRow]) -> float:
    """Tokens emitted per second across the given windows, from wall-clock window_ms.

    Each window emits n_accepted draft tokens plus one target token (the
    correction or bonus). With K = 0 that is exactly one token per window.
    """
    if not rows:
        return 0.0
    tokens = sum(row[_N_ACCEPTED] + 1 for row in rows)
    elapsed_ms = sum(row[_WINDOW_MS] for row in rows)
    if elapsed_ms <= 0:
        return 0.0
    return tokens / elapsed_ms * MS_PER_S
