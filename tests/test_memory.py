"""The startup memory-fit guard: refuse a pair that cannot load safely."""

from __future__ import annotations

import pytest

from acceptrate.config import DEFAULT_PAIR
from acceptrate.memory import MemoryFit, assess_fit

GB = 1024**3


def test_fits_with_headroom_on_a_16gb_machine_with_ten_free() -> None:
    fit = assess_fit(DEFAULT_PAIR, available_bytes=10 * GB)

    assert fit.ok
    assert fit.headroom_gb == pytest.approx(10 - DEFAULT_PAIR.estimated_gb, abs=0.01)


def test_refuses_when_available_memory_is_below_estimate() -> None:
    fit = assess_fit(DEFAULT_PAIR, available_bytes=5 * GB)

    assert not fit.ok
    assert fit.headroom_gb < 0


def test_refuses_when_headroom_is_under_the_safety_margin() -> None:
    barely = int((DEFAULT_PAIR.estimated_gb + 0.5) * GB)

    fit = assess_fit(DEFAULT_PAIR, available_bytes=barely)

    assert not fit.ok
    assert 0 < fit.headroom_gb < MemoryFit.MIN_HEADROOM_GB


def test_reason_names_the_shortfall_when_refusing() -> None:
    fit = assess_fit(DEFAULT_PAIR, available_bytes=5 * GB)

    assert "GB" in fit.reason
    assert fit.reason != ""


def test_live_probe_returns_a_positive_byte_count() -> None:
    from acceptrate.memory import available_bytes

    assert available_bytes() > 0
