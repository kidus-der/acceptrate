"""System guards: the 1 Hz sampler thread and the parsers behind it.

The hot loop only ever reads ints from a GuardSnapshot. Everything that
shells out lives on the guard thread, so parsers are tested here against
canned command output, plus one live test against this Mac.
"""

from __future__ import annotations

import pytest

from acceptrate.bench.guards import (
    GuardParseError,
    GuardSnapshot,
    dirty_reason,
    is_clean,
    page_ins_from_swap_in,
    parse_mem_pressure,
    parse_thermal,
)

# --- pmset -g therm -----------------------------------------------------------

THERM_NOMINAL_NOTES = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)
THERM_NOMINAL_KEYS = (
    "Note: No thermal warning level has been recorded\n"
    "CPU_Scheduler_Limit \t= 100\n"
    "CPU_Available_CPUs \t= 10\n"
    "CPU_Speed_Limit \t= 100\n"
)
THERM_THROTTLED = (
    "Note: No thermal warning level has been recorded\n"
    "CPU_Scheduler_Limit \t= 80\n"
    "CPU_Available_CPUs \t= 10\n"
    "CPU_Speed_Limit \t= 60\n"
)


def test_thermal_is_nominal_when_pmset_has_recorded_nothing() -> None:
    assert parse_thermal(THERM_NOMINAL_NOTES) == 0


def test_thermal_is_nominal_when_both_limits_are_100() -> None:
    assert parse_thermal(THERM_NOMINAL_KEYS) == 0


def test_thermal_level_is_the_worst_percentage_lost() -> None:
    # speed 60 -> 40 lost; scheduler 80 -> 20 lost; report the worse of the two
    assert parse_thermal(THERM_THROTTLED) == 40


def test_thermal_rejects_output_with_neither_notes_nor_limits() -> None:
    with pytest.raises(GuardParseError):
        parse_thermal("")
    with pytest.raises(GuardParseError):
        parse_thermal("pmset: unrecognized argument\n")


def test_thermal_rejects_a_limit_outside_0_to_100() -> None:
    with pytest.raises(GuardParseError):
        parse_thermal("CPU_Speed_Limit \t= 150\nCPU_Scheduler_Limit \t= 100\n")


# --- sysctl -n kern.memorystatus_vm_pressure_level ----------------------------


@pytest.mark.parametrize(("raw", "level"), [("1\n", 0), ("2\n", 1), ("4\n", 2)])
def test_mem_pressure_maps_kernel_levels_to_0_1_2(raw: str, level: int) -> None:
    assert parse_mem_pressure(raw) == level


@pytest.mark.parametrize("raw", ["", "3\n", "abc\n", "sysctl: unknown oid\n"])
def test_mem_pressure_rejects_unknown_output(raw: str) -> None:
    with pytest.raises(GuardParseError):
        parse_mem_pressure(raw)


# --- page-ins from psutil.swap_memory().sin ----------------------------------


def test_page_ins_is_swap_in_bytes_divided_by_page_size() -> None:
    assert page_ins_from_swap_in(sin_bytes=49162421 * 16384, page_size=16384) == 49162421


def test_page_ins_rejects_a_byte_count_that_is_not_page_aligned() -> None:
    with pytest.raises(GuardParseError):
        page_ins_from_swap_in(sin_bytes=16385, page_size=16384)


# --- snapshot -----------------------------------------------------------------


def test_snapshot_is_immutable() -> None:
    snap = GuardSnapshot(mem_pressure=0, page_ins=1, thermal_level=0, sampled_at=0.0)

    with pytest.raises((AttributeError, TypeError)):
        snap.page_ins = 2  # type: ignore[misc]


# --- is_clean / dirty_reason --------------------------------------------------


def _snap(mem: int = 0, page_ins: int = 100, thermal: int = 0, at: float = 0.0) -> GuardSnapshot:
    return GuardSnapshot(mem_pressure=mem, page_ins=page_ins, thermal_level=thermal, sampled_at=at)


def test_clean_when_nothing_moved_and_both_ends_are_nominal() -> None:
    assert is_clean(_snap(at=0.0), _snap(at=1.0))
    assert dirty_reason(_snap(at=0.0), _snap(at=1.0)) == ""


def test_dirty_when_page_ins_advanced_over_the_window() -> None:
    before, after = _snap(page_ins=100), _snap(page_ins=103)

    assert not is_clean(before, after)
    assert "page_ins" in dirty_reason(before, after)
    assert "3" in dirty_reason(before, after)


@pytest.mark.parametrize(("before", "after"), [(_snap(mem=1), _snap()), (_snap(), _snap(mem=2))])
def test_dirty_when_memory_pressure_is_elevated_at_either_end(
    before: GuardSnapshot, after: GuardSnapshot
) -> None:
    assert not is_clean(before, after)
    assert "mem_pressure" in dirty_reason(before, after)


@pytest.mark.parametrize(
    ("before", "after"), [(_snap(thermal=5), _snap()), (_snap(), _snap(thermal=40))]
)
def test_dirty_when_throttled_at_either_end(before: GuardSnapshot, after: GuardSnapshot) -> None:
    assert not is_clean(before, after)
    assert "thermal_level" in dirty_reason(before, after)


def test_dirty_reason_lists_every_cause() -> None:
    reason = dirty_reason(_snap(mem=1, page_ins=1, thermal=1), _snap(mem=0, page_ins=2, thermal=0))

    assert "page_ins" in reason
    assert "mem_pressure" in reason
    assert "thermal_level" in reason
