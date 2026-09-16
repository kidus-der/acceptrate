"""Live statistics over the rolling window of recent draft windows.

Pure functions of (window, totals). The Session owns the window; GET /stats
and the SSE feed call `compute_stats` on a snapshot. `alpha_ewma` is the
same quantity the adaptive scheduler (P5) will drive K from, so the TUI shows
what the runtime sees.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from acceptrate.trace.schema import FIELD_NAMES, WindowRow

EWMA_DECAY = 0.1
"""Weight of the newest window in the acceptance-rate EWMA."""

LAST_WINDOWS = 16
"""How many recent windows /stats lists verbatim."""

MS_PER_S = 1000.0

_COL = {name: i for i, name in enumerate(FIELD_NAMES)}


class WindowStat(NamedTuple):
    k_proposed: int
    n_accepted: int
    n_committed: int
    draft_ms: float
    verify_ms: float
    window_ms: float


class Totals(NamedTuple):
    windows: int
    accepted: int
    proposed: int


class WindowSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    k_proposed: int
    n_accepted: int
    draft_ms: float
    verify_ms: float
    window_ms: float


class Stats(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    draft: str | None
    busy: bool
    k_current: int
    alpha_ewma: float | None
    tok_s_recent: float | None
    windows_total: int
    accepted_total: int
    proposed_total: int
    last_windows: list[WindowSummary]


def window_stat_from_row(row: WindowRow, n_committed: int) -> WindowStat:
    return WindowStat(
        k_proposed=row[_COL["k_proposed"]],
        n_accepted=row[_COL["n_accepted"]],
        n_committed=n_committed,
        draft_ms=row[_COL["draft_ms"]],
        verify_ms=row[_COL["verify_ms"]],
        window_ms=row[_COL["window_ms"]],
    )


def ewma_alpha(window: Sequence[WindowStat], decay: float = EWMA_DECAY) -> float | None:
    """EWMA of per-window acceptance fraction n/k over speculative windows; None if none."""
    ewma: float | None = None
    for stat in window:
        if stat.k_proposed <= 0:
            continue
        # per-token acceptance: a rejection is one examined token (same alpha the scheduler uses)
        examined = stat.n_accepted + 1 if stat.n_accepted < stat.k_proposed else stat.k_proposed
        rate = stat.n_accepted / examined
        ewma = rate if ewma is None else (1 - decay) * ewma + decay * rate
    return ewma


def tokens_per_second(window: Sequence[WindowStat]) -> float | None:
    """Committed tokens over wall clock of the window (window_ms, Python overhead included)."""
    total_ms = sum(stat.window_ms for stat in window)
    if total_ms <= 0:
        return None
    return sum(stat.n_committed for stat in window) / total_ms * MS_PER_S


def compute_stats(
    window: Sequence[WindowStat],
    totals: Totals,
    *,
    model: str,
    draft: str | None,
    busy: bool,
    k_current: int,
    last_n: int = LAST_WINDOWS,
) -> Stats:
    recent = window[-last_n:] if last_n > 0 else ()
    return Stats(
        model=model,
        draft=draft,
        busy=busy,
        k_current=k_current,
        alpha_ewma=ewma_alpha(window),
        tok_s_recent=tokens_per_second(window),
        windows_total=totals.windows,
        accepted_total=totals.accepted,
        proposed_total=totals.proposed,
        last_windows=[
            WindowSummary(
                k_proposed=s.k_proposed,
                n_accepted=s.n_accepted,
                draft_ms=s.draft_ms,
                verify_ms=s.verify_ms,
                window_ms=s.window_ms,
            )
            for s in recent
        ],
    )
