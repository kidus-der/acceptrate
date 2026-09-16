"""Startup memory-fit guard.

On 16 GB unified memory the OS is a co-tenant. Crossing into memory pressure
does not degrade gracefully — page-outs fall off a cliff and invalidate every
measurement. So a pair that would leave less than MIN_HEADROOM_GB free is
refused before any weight is loaded (brief tab 3 / tab 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil

from acceptrate.config import ModelPairConfig

GB = 1024**3


@dataclass(frozen=True)
class MemoryFit:
    MIN_HEADROOM_GB = 2.0

    ok: bool
    required_gb: float
    available_gb: float
    headroom_gb: float
    reason: str


def available_bytes() -> int:
    """Memory the OS reports as available right now (free + reclaimable)."""
    return int(psutil.virtual_memory().available)


def assess_fit(pair: ModelPairConfig, available_bytes: int) -> MemoryFit:
    required = pair.estimated_gb
    available = available_bytes / GB
    headroom = available - required
    if headroom < 0:
        reason = f"needs ~{required:.1f} GB but only {available:.1f} GB is available"
        return MemoryFit(False, required, available, headroom, reason)
    if headroom < MemoryFit.MIN_HEADROOM_GB:
        reason = (
            f"would leave only {headroom:.1f} GB free; "
            f"need >= {MemoryFit.MIN_HEADROOM_GB:.1f} GB to stay out of memory pressure"
        )
        return MemoryFit(False, required, available, headroom, reason)
    return MemoryFit(True, required, available, headroom, "")
