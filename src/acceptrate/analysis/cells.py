"""Per-cell metrics: one row per (target, max_tokens, session, draft, k, workload_tag).

The aggregation is DuckDB SQL over the registered Polars frame. Two acceptance
figures are reported and they are not the same thing:

  alpha        accepted / examined draft tokens. A window examines n_accepted
               tokens plus the one that was rejected (none if all K passed).
               This is the per-token acceptance rate the closed form takes.
  accept_frac  accepted / proposed. Tokens after the first rejection were
               drafted but never examined, so this is below alpha for K > 1;
               it measures wasted draft work, not acceptance.

Throughput is per generation (prompt_id, rep) from wall-clock window_ms, and
the cell reports its median and IQR — never a mean (CLAUDE.md, measurement
rules). Speedup is the cell median over the K=0 median of the same
(target, max_tokens, workload_tag), i.e. the baseline of the same sweep.
"""

from __future__ import annotations

import duckdb
import polars as pl

CELL_KEY: tuple[str, ...] = ("target", "max_tokens", "session", "draft", "k", "workload_tag")
BASELINE_KEY: tuple[str, ...] = ("target", "max_tokens", "session", "workload_tag")
ROWS_VIEW = "rows"
SPECULATION_PARITY = 1.0
"""Measured speedup at which speculation neither wins nor loses."""
DEFAULT_POSITION_BIN = 16

CELL_SQL = """
WITH generations AS (
    SELECT target, max_tokens, session, draft, k, workload_tag, prompt_id, rep,
           SUM(n_accepted + 1) AS tokens,
           SUM(window_ms) AS elapsed_ms
    FROM rows
    GROUP BY ALL
),
rates AS (
    SELECT target, max_tokens, session, draft, k, workload_tag,
           COUNT(*) AS n_generations,
           MEDIAN(tokens / elapsed_ms * 1000.0) AS tok_s_median,
           QUANTILE_CONT(tokens / elapsed_ms * 1000.0, 0.25) AS tok_s_q1,
           QUANTILE_CONT(tokens / elapsed_ms * 1000.0, 0.75) AS tok_s_q3
    FROM generations
    WHERE elapsed_ms > 0
    GROUP BY ALL
),
windows AS (
    SELECT target, max_tokens, session, draft, k, workload_tag,
           COUNT(*) AS n_windows,
           SUM(n_accepted)::DOUBLE
               / NULLIF(SUM(n_accepted + (n_accepted < k_proposed)::INTEGER), 0) AS alpha,
           SUM(n_accepted)::DOUBLE / NULLIF(SUM(k_proposed), 0) AS accept_frac,
           MEDIAN(CASE WHEN k_proposed > 0 AND verify_ms > 0
                       THEN draft_ms / k_proposed / verify_ms END) AS c,
           MEDIAN(verify_ms) AS verify_ms_median,
           MEDIAN(window_ms) AS window_ms_median
    FROM rows
    GROUP BY ALL
)
SELECT w.target, w.max_tokens, w.session, w.draft, w.k, w.workload_tag,
       w.n_windows, r.n_generations, w.alpha, w.accept_frac, w.c,
       w.verify_ms_median, w.window_ms_median,
       r.tok_s_median, r.tok_s_q1, r.tok_s_q3, r.tok_s_q3 - r.tok_s_q1 AS tok_s_iqr
FROM windows w
JOIN rates r
  ON w.target = r.target AND w.max_tokens = r.max_tokens
 AND w.session = r.session
 AND w.draft IS NOT DISTINCT FROM r.draft
 AND w.k = r.k AND w.workload_tag = r.workload_tag
ORDER BY w.target, w.max_tokens, w.draft NULLS FIRST, w.k, w.workload_tag
"""


def _aggregate(df: pl.DataFrame) -> pl.DataFrame:
    with duckdb.connect() as con:
        con.register(ROWS_VIEW, df)
        return con.execute(CELL_SQL).pl()


def _with_baseline(cells: pl.DataFrame) -> pl.DataFrame:
    """Join each cell to the K=0 cell of the same session and tag.

    Adds baseline_tok_s, measured_speedup, and v = the cell's median verify
    pass over the baseline's median plain step — the measured cost of one
    verification relative to one decode step, which the closed form assumes
    is exactly 1.
    """
    baseline = (
        cells.filter(pl.col("k") == 0)
        .select(
            [
                *BASELINE_KEY,
                pl.col("tok_s_median").alias("baseline_tok_s"),
                pl.col("window_ms_median").alias("baseline_step_ms"),
            ]
        )
        .unique(subset=list(BASELINE_KEY))
    )
    return cells.join(baseline, on=list(BASELINE_KEY), how="left").with_columns(
        measured_speedup=pl.col("tok_s_median") / pl.col("baseline_tok_s"),
        v=pl.col("verify_ms_median") / pl.col("baseline_step_ms"),
    )


def cell_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Per-cell alpha, cost ratio, throughput median/IQR and measured speedup.

    Pass a cleaned frame; dirty windows are excluded from headline numbers by
    the caller, never silently here. Cells with no K=0 baseline of the same
    (target, max_tokens, workload_tag) get a null measured_speedup.
    """
    return _with_baseline(_aggregate(df))


def alpha_by_position(df: pl.DataFrame, bin_size: int = DEFAULT_POSITION_BIN) -> pl.DataFrame:
    """Per-token alpha per (workload_tag, token position bin) over every K >= 1 window.

    The within-response drift the thesis rests on: alpha is not one number
    per prompt, it moves with where in the output the window sits.
    """
    if bin_size < 1:
        raise ValueError(f"bin_size must be >= 1, got {bin_size}")
    examined = pl.col("n_accepted") + (pl.col("n_accepted") < pl.col("k_proposed")).cast(pl.Int32)
    return (
        df.filter(pl.col("k_proposed") > 0)
        .with_columns(pos_bin=(pl.col("token_pos") // bin_size) * bin_size)
        .group_by("workload_tag", "pos_bin")
        .agg(
            n_windows=pl.len(),
            accepted=pl.col("n_accepted").sum(),
            examined=examined.sum(),
        )
        .with_columns(alpha=pl.col("accepted") / pl.col("examined"))
        .sort("workload_tag", "pos_bin")
    )
