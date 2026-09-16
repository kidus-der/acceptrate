"""System guards: memory pressure, page-ins and thermal throttling.

Trap 4 of CLAUDE.md: the hot loop allocates nothing and never shells out.
So sampling happens on a 1 Hz daemon thread that publishes an immutable
GuardSnapshot; the loop reads `SystemGuard.latest` (a single attribute read)
and compares ints. No sudo, no powermetrics.

Sources, verified on the M4 dev box (macOS 26 / Darwin 25.5):

- thermal:  `pmset -g therm`. When nothing has been recorded it prints only
  "Note: No ... recorded" lines; when throttling has occurred it prints
  `CPU_Speed_Limit = N` and `CPU_Scheduler_Limit = N` (100 = unthrottled).
  thermal_level = max(100 - speed, 100 - scheduler): 0 when nominal,
  otherwise the worst percentage of capacity lost.
- memory pressure: `sysctl -n kern.memorystatus_vm_pressure_level`, which
  reports 1 normal / 2 warn / 4 critical and is mapped to 0 / 1 / 2.
- page-ins: `psutil.swap_memory().sin`, which psutil computes from
  host_statistics64 as pageins * page_size; dividing by mmap.PAGESIZE gives
  vm_stat's "Pageins" without a shell-out.
"""

from __future__ import annotations

import mmap
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from typing import NamedTuple, Self

import psutil

THERMAL_COMMAND = ("pmset", "-g", "therm")
MEM_PRESSURE_COMMAND = ("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
COMMAND_TIMEOUT_S = 5.0
DEFAULT_INTERVAL_S = 1.0

THERMAL_LIMIT_KEYS = ("CPU_Speed_Limit", "CPU_Scheduler_Limit")
THERMAL_NOMINAL_NOTE = re.compile(r"^Note: No .* recorded", re.MULTILINE)
THERMAL_LIMIT_LINE = re.compile(r"^(CPU_\w+)\s*=\s*(\d+)\s*$", re.MULTILINE)
THERMAL_UNTHROTTLED = 100

KERNEL_PRESSURE_TO_LEVEL = {1: 0, 2: 1, 4: 2}

Runner = Callable[[Sequence[str]], str]


class GuardParseError(ValueError):
    """A guard source returned output that cannot be trusted as a reading."""


class GuardSnapshot(NamedTuple):
    mem_pressure: int
    """0 normal / 1 warn / 2 critical."""
    page_ins: int
    """Cumulative page-ins since boot."""
    thermal_level: int
    """0 nominal; otherwise the worst percentage of CPU capacity lost."""
    sampled_at: float
    """time.perf_counter() at sampling."""


def run_command(argv: Sequence[str]) -> str:
    """Run a guard command and return its stdout; any failure is an explicit error."""
    completed = subprocess.run(
        list(argv), capture_output=True, text=True, check=True, timeout=COMMAND_TIMEOUT_S
    )
    return completed.stdout


# --- thermal ------------------------------------------------------------------


def parse_thermal(text: str) -> int:
    """thermal_level from `pmset -g therm` output; see the module docstring."""
    limits = {key: int(value) for key, value in THERMAL_LIMIT_LINE.findall(text)}
    present = [key for key in THERMAL_LIMIT_KEYS if key in limits]
    if not present:
        if THERMAL_NOMINAL_NOTE.search(text):
            return 0
        raise GuardParseError(f"pmset -g therm output has no limits and no notes: {text!r}")
    lost = [THERMAL_UNTHROTTLED - limits[key] for key in present]
    if any(limits[key] > THERMAL_UNTHROTTLED for key in present):
        raise GuardParseError(f"pmset limit above {THERMAL_UNTHROTTLED}: {limits!r}")
    return max(lost)


def sample_thermal(run: Runner = run_command) -> int:
    return parse_thermal(run(THERMAL_COMMAND))


# --- memory pressure ----------------------------------------------------------


def parse_mem_pressure(text: str) -> int:
    """0/1/2 from the kernel's 1/2/4 `kern.memorystatus_vm_pressure_level`."""
    raw = text.strip()
    if not raw.isdigit() or int(raw) not in KERNEL_PRESSURE_TO_LEVEL:
        raise GuardParseError(f"unexpected kern.memorystatus_vm_pressure_level: {text!r}")
    return KERNEL_PRESSURE_TO_LEVEL[int(raw)]


def sample_mem_pressure(run: Runner = run_command) -> int:
    return parse_mem_pressure(run(MEM_PRESSURE_COMMAND))


# --- page-ins -----------------------------------------------------------------


def page_ins_from_swap_in(sin_bytes: int, page_size: int) -> int:
    """Cumulative page-ins from psutil's byte count (pageins * page_size)."""
    if page_size <= 0 or sin_bytes < 0 or sin_bytes % page_size:
        raise GuardParseError(f"swap-in bytes {sin_bytes} not a multiple of page size {page_size}")
    return sin_bytes // page_size


def read_swap_in_bytes() -> int:
    return int(psutil.swap_memory().sin)


def sample_page_ins(read_swap_in: Callable[[], int] = read_swap_in_bytes) -> int:
    return page_ins_from_swap_in(read_swap_in(), mmap.PAGESIZE)


# --- window verdict -----------------------------------------------------------


def dirty_reason(before: GuardSnapshot, after: GuardSnapshot) -> str:
    """Every reason the window between two snapshots is not a clean sample; "" if clean."""
    reasons: list[str] = []
    page_in_delta = after.page_ins - before.page_ins
    if page_in_delta > 0:
        reasons.append(f"page_ins advanced by {page_in_delta}")
    worst_pressure = max(before.mem_pressure, after.mem_pressure)
    if worst_pressure > 0:
        reasons.append(f"mem_pressure {worst_pressure}")
    worst_thermal = max(before.thermal_level, after.thermal_level)
    if worst_thermal > 0:
        reasons.append(f"thermal_level {worst_thermal}")
    return "; ".join(reasons)


def is_clean(before: GuardSnapshot, after: GuardSnapshot) -> bool:
    """True if no page-ins landed and neither end was under pressure or throttled."""
    return dirty_reason(before, after) == ""


# --- the 1 Hz guard -----------------------------------------------------------


def take_snapshot(run: Runner = run_command) -> GuardSnapshot:
    """One synchronous reading of all three sources."""
    return GuardSnapshot(
        mem_pressure=sample_mem_pressure(run),
        page_ins=sample_page_ins(),
        thermal_level=sample_thermal(run),
        sampled_at=time.perf_counter(),
    )


class SystemGuard:
    """Samples the machine on a daemon thread and publishes the latest GuardSnapshot.

    `latest` is a plain attribute read (atomic reference swap on the writer
    side), so the hot loop pays nothing and takes no lock. Sampler failures on
    the thread are counted in `sample_errors` and kept in `last_error`; the
    previous good snapshot stays published.
    """

    def __init__(
        self,
        sampler: Callable[[], GuardSnapshot] = take_snapshot,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._sampler = sampler
        self._interval_s = interval_s
        self._latest: GuardSnapshot | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sample_errors = 0
        self.last_error: Exception | None = None

    @property
    def latest(self) -> GuardSnapshot:
        snapshot = self._latest
        if snapshot is None:
            raise RuntimeError("SystemGuard has no snapshot yet; call start() or sample_now()")
        return snapshot

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def sample_now(self) -> GuardSnapshot:
        """Sample synchronously and publish; errors propagate to the caller."""
        snapshot = self._sampler()
        self._latest = snapshot
        return snapshot

    def start(self) -> None:
        """Take one sample synchronously (so `latest` is valid) and start the thread."""
        if self._thread is not None:
            raise RuntimeError("SystemGuard already started")
        self.sample_now()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="acceptrate-guard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread and join it. Safe to call more than once."""
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join()
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.sample_now()
            except Exception as exc:  # counted and surfaced, never swallowed
                self.sample_errors += 1
                self.last_error = exc

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
