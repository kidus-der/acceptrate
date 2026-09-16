"""Startup memory-fit guard.

On unified memory the OS is a co-tenant. Crossing into memory pressure does
not degrade gracefully — page-outs fall off a cliff and invalidate every
measurement. So a pair that would leave less than MIN_HEADROOM_GB of the
post-OS budget is refused before any weight is loaded (brief tab 3 / tab 4).

The budget is total physical memory minus OS_RESERVE_GB, deliberately not
psutil's "available": macOS keeps recently read weight files as active page
cache that "available" excludes, yet MLX maps those very pages. Whether the
run *actually* stays clean is decided per window by bench/guards.py.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import psutil

from acceptrate.config import ModelPairConfig

GB = 1024**3
OS_RESERVE_GB = 4.5
"""What macOS and the desktop keep for themselves on a typical machine (brief tab 3)."""

_PRESSURE_SYSCTL = "kern.memorystatus_vm_pressure_level"
_PRESSURE_MAP = {1: 0, 2: 1, 4: 2}


@dataclass(frozen=True)
class MemoryFit:
    MIN_HEADROOM_GB = 2.0

    ok: bool
    required_gb: float
    budget_gb: float
    headroom_gb: float
    reason: str


def total_bytes() -> int:
    return int(psutil.virtual_memory().total)


def current_pressure_level() -> int:
    """0 normal / 1 warn / 2 critical, from the kernel; 0 if the sysctl is unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", _PRESSURE_SYSCTL], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if out.returncode != 0:
        return 0
    return _PRESSURE_MAP.get(int(out.stdout.strip() or 1), 0)


def assess_fit(pair: ModelPairConfig, total_bytes: int, pressure_level: int) -> MemoryFit:
    required = pair.estimated_gb
    budget = total_bytes / GB - OS_RESERVE_GB
    headroom = budget - required
    if pressure_level > 0:
        reason = f"machine is already under memory pressure (level {pressure_level})"
        return MemoryFit(False, required, budget, headroom, reason)
    if headroom < 0:
        reason = (
            f"needs ~{required:.1f} GB but only {budget:.1f} GB remains "
            f"after the {OS_RESERVE_GB:.1f} GB OS reserve"
        )
        return MemoryFit(False, required, budget, headroom, reason)
    if headroom < MemoryFit.MIN_HEADROOM_GB:
        reason = (
            f"would leave only {headroom:.1f} GB of headroom; "
            f"need >= {MemoryFit.MIN_HEADROOM_GB:.1f} GB to stay out of memory pressure"
        )
        return MemoryFit(False, required, budget, headroom, reason)
    return MemoryFit(True, required, budget, headroom, "")
