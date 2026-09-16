"""The trace schema: ONE ROW PER DRAFT WINDOW, never per request.

This is the only thing bench/ and runtime/ share. Per-request averages
destroy the signal the whole project depends on (alpha is high mid-code-block
and low at a sentence boundary) and it cannot be recovered afterwards.

Rows are plain tuples in FIELD_NAMES order so the hot loop appends without
building dicts (CLAUDE.md trap 4). Polars typing happens once, at flush.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

WORKLOAD_TAGS: tuple[str, ...] = ("code", "json", "prose", "chat", "summarize", "reason")

SCHEMA = pl.Schema(
    {
        # --- the brief's table (tab 4), in order ---
        "run_id": pl.Utf8,  # groups a sweep cell; carries the full config hash
        "window_idx": pl.Int32,  # draft/verify round within the generation
        "token_pos": pl.Int32,  # output position at window start; alpha drifts with depth
        "k_proposed": pl.Int8,  # draft depth used this window (0 = plain decode)
        "n_accepted": pl.Int8,  # the numerator of everything
        "draft_ms": pl.Float32,  # measured, with mx.eval before the clock stops
        "verify_ms": pl.Float32,  # measured, likewise
        "workload_tag": pl.Utf8,  # one of WORKLOAD_TAGS
        "mem_pressure": pl.UInt8,  # 0 normal / 1 warn / 2 critical; >0 excluded from headlines
        "page_ins": pl.Int64,  # delta over the window; non-zero invalidates the sample
        "thermal_level": pl.UInt8,  # from pmset; throttled run = discarded run
        # --- additions needed to make rows unambiguous and speedup honest ---
        "prompt_id": pl.Utf8,  # which corpus prompt
        "rep": pl.Int32,  # repeat index of this prompt within the run
        "window_ms": pl.Float32,  # wall clock for the whole window, Python overhead included
    }
)

FIELD_NAMES: tuple[str, ...] = tuple(SCHEMA.names())

WindowRow = tuple[str, int, int, int, int, float, float, str, int, int, int, str, int, float]


def make_row(
    run_id: str,
    window_idx: int,
    token_pos: int,
    k_proposed: int,
    n_accepted: int,
    draft_ms: float,
    verify_ms: float,
    workload_tag: str,
    mem_pressure: int,
    page_ins: int,
    thermal_level: int,
    prompt_id: str,
    rep: int,
    window_ms: float,
) -> WindowRow:
    """Build one row. Keyword use is for tests; the hot loop builds tuples directly."""
    return (
        run_id, window_idx, token_pos, k_proposed, n_accepted, draft_ms, verify_ms,
        workload_tag, mem_pressure, page_ins, thermal_level, prompt_id, rep, window_ms,
    )  # fmt: skip


def frame_from_rows(rows: Sequence[WindowRow]) -> pl.DataFrame:
    """Type a batch of rows exactly to SCHEMA. Called once per generation, never per window."""
    return pl.DataFrame(list(rows), schema=SCHEMA, orient="row")
