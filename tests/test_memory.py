"""The startup memory-fit guard: refuse a pair that cannot load safely.

Budget = total physical memory minus the OS co-tenant reserve (brief tab 3:
~4.5 GB), not psutil's "available". macOS keeps recently read model files
as active page cache that psutil does not count as available, yet MLX maps
those same pages, so a byte-availability heuristic refuses runs that are
fine. Dynamic trouble (page-ins, pressure) is caught per window by guards.
"""

from __future__ import annotations

import pytest

from acceptrate.config import DEFAULT_PAIR, ModelPairConfig, ModelSpec
from acceptrate.memory import OS_RESERVE_GB, MemoryFit, assess_fit

GB = 1024**3
SIXTEEN = 16 * GB
FOURTEEN_B_PAIR = ModelPairConfig(
    target=ModelSpec(repo="org/14b-4bit", weights_gb=8.0),
    draft=DEFAULT_PAIR.draft,
    kv_cache_gb=1.4,
)


def test_default_pair_fits_a_16gb_machine_with_headroom() -> None:
    fit = assess_fit(DEFAULT_PAIR, total_bytes=SIXTEEN, pressure_level=0)

    assert fit.ok
    expected = 16 - OS_RESERVE_GB - DEFAULT_PAIR.estimated_gb
    assert fit.headroom_gb == pytest.approx(expected, abs=0.01)


def test_the_briefs_14b_boundary_is_refused_on_16gb() -> None:
    fit = assess_fit(FOURTEEN_B_PAIR, total_bytes=SIXTEEN, pressure_level=0)

    assert not fit.ok
    assert 0 < fit.headroom_gb < MemoryFit.MIN_HEADROOM_GB
    assert "GB" in fit.reason


def test_refuses_an_8b_target_on_an_8gb_machine() -> None:
    fit = assess_fit(DEFAULT_PAIR, total_bytes=8 * GB, pressure_level=0)

    assert not fit.ok
    assert fit.headroom_gb < 0


def test_refuses_when_the_machine_is_already_under_memory_pressure() -> None:
    fit = assess_fit(DEFAULT_PAIR, total_bytes=SIXTEEN, pressure_level=1)

    assert not fit.ok
    assert "pressure" in fit.reason


def test_allows_a_32b_pair_on_64gb() -> None:
    big = ModelPairConfig(
        target=ModelSpec(repo="org/32b-4bit", weights_gb=18.0), draft=DEFAULT_PAIR.draft
    )

    assert assess_fit(big, total_bytes=64 * GB, pressure_level=0).ok


def test_live_probes_return_sane_values() -> None:
    from acceptrate.memory import current_pressure_level, total_bytes

    assert total_bytes() > 1 * GB  # any machine that can run the suite; CI runners have 7 GB
    assert current_pressure_level() in (0, 1, 2)
