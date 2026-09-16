"""System guards: the 1 Hz sampler thread and the parsers behind it.

The hot loop only ever reads ints from a GuardSnapshot. Everything that
shells out lives on the guard thread, so parsers are tested here against
canned command output, plus one live test against this Mac.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import time
from collections.abc import Callable

import pytest

from acceptrate.bench.guards import (
    GuardParseError,
    GuardSnapshot,
    SystemGuard,
    dirty_reason,
    is_clean,
    page_ins_from_swap_in,
    parse_mem_pressure,
    parse_thermal,
    take_snapshot,
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


# --- SystemGuard thread -------------------------------------------------------

FAST_INTERVAL_S = 0.005
WAIT_DEADLINE_S = 2.0


def _counting_sampler() -> Callable[[], GuardSnapshot]:
    ticks = itertools.count()
    return lambda: _snap(page_ins=next(ticks), at=time.perf_counter())


def _wait_until(predicate: Callable[[], bool]) -> None:
    deadline = time.perf_counter() + WAIT_DEADLINE_S
    while not predicate():
        if time.perf_counter() > deadline:
            raise AssertionError("guard thread never satisfied the condition")
        time.sleep(FAST_INTERVAL_S)


def test_latest_raises_before_any_sample_is_taken() -> None:
    guard = SystemGuard(sampler=_counting_sampler(), interval_s=FAST_INTERVAL_S)

    with pytest.raises(RuntimeError):
        _ = guard.latest


def test_sample_now_publishes_synchronously_without_a_thread() -> None:
    guard = SystemGuard(sampler=_counting_sampler(), interval_s=FAST_INTERVAL_S)

    snap = guard.sample_now()

    assert guard.latest is snap
    assert not guard.is_running


def test_start_publishes_a_first_snapshot_before_returning() -> None:
    guard = SystemGuard(sampler=_counting_sampler(), interval_s=FAST_INTERVAL_S)
    try:
        guard.start()
        assert isinstance(guard.latest, GuardSnapshot)
        assert guard.is_running
    finally:
        guard.stop()


def test_thread_keeps_publishing_fresh_snapshots() -> None:
    with SystemGuard(sampler=_counting_sampler(), interval_s=FAST_INTERVAL_S) as guard:
        first = guard.latest

        _wait_until(lambda: guard.latest.page_ins >= first.page_ins + 3)

        assert guard.latest.sampled_at > first.sampled_at


def test_stop_joins_the_thread_and_is_idempotent() -> None:
    guard = SystemGuard(sampler=_counting_sampler(), interval_s=FAST_INTERVAL_S)
    guard.start()

    guard.stop()
    settled = guard.latest
    time.sleep(FAST_INTERVAL_S * 10)

    assert not guard.is_running
    assert guard.latest is settled
    guard.stop()


def test_sampler_errors_are_counted_and_keep_the_last_good_snapshot() -> None:
    calls = itertools.count()

    def flaky() -> GuardSnapshot:
        n = next(calls)
        if n % 2 == 1:
            raise GuardParseError("boom")
        return _snap(page_ins=n, at=time.perf_counter())

    with SystemGuard(sampler=flaky, interval_s=FAST_INTERVAL_S) as guard:
        _wait_until(lambda: guard.sample_errors >= 2)

        assert isinstance(guard.latest, GuardSnapshot)
        assert isinstance(guard.last_error, GuardParseError)
        assert guard.is_running


def test_sample_now_raises_instead_of_counting() -> None:
    def broken() -> GuardSnapshot:
        raise GuardParseError("boom")

    guard = SystemGuard(sampler=broken, interval_s=FAST_INTERVAL_S)

    with pytest.raises(GuardParseError):
        guard.sample_now()
    assert guard.sample_errors == 0


def test_start_twice_is_an_error() -> None:
    with (
        SystemGuard(sampler=_counting_sampler(), interval_s=FAST_INTERVAL_S) as guard,
        pytest.raises(RuntimeError),
    ):
        guard.start()


# --- live, on this Mac (CI is macOS; no marker needed) ------------------------

VM_STAT_PAGEINS = re.compile(r"^Pageins:\s+(\d+)\.", re.MULTILINE)
PAGEINS_DRIFT_TOLERANCE = 0.01


def test_live_samplers_return_ints_in_range() -> None:
    snap = take_snapshot()

    assert snap.mem_pressure in (0, 1, 2)
    assert snap.thermal_level >= 0
    assert snap.page_ins > 0
    assert all(isinstance(v, int) for v in (snap.mem_pressure, snap.page_ins, snap.thermal_level))
    assert snap.sampled_at > 0


def test_live_page_ins_agree_with_vm_stat() -> None:
    snap = take_snapshot()
    vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    match = VM_STAT_PAGEINS.search(vm_stat)
    assert match is not None, vm_stat

    reported = int(match.group(1))

    assert reported >= snap.page_ins
    assert reported - snap.page_ins <= reported * PAGEINS_DRIFT_TOLERANCE
