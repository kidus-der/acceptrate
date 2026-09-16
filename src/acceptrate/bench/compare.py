"""Compare two runs of the same configuration — the P1 repeatability gate."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from acceptrate.bench.stats import MedianIQR, median_iqr

GATE_TOLERANCE = 0.02
"""Same config, two invocations: median tok/s must agree within this fraction."""

MS_PER_S = 1000.0


@dataclass(frozen=True)
class Comparison:
    a: MedianIQR
    b: MedianIQR
    rel_diff: float
    dirty_a: int
    dirty_b: int

    @property
    def passed(self) -> bool:
        return self.rel_diff <= GATE_TOLERANCE


def clean_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Drop windows with page-ins, memory pressure or thermal throttling."""
    return df.filter(
        (pl.col("page_ins") == 0) & (pl.col("mem_pressure") == 0) & (pl.col("thermal_level") == 0)
    )


def per_generation_tok_s(df: pl.DataFrame) -> list[float]:
    """One throughput figure per generation (prompt_id, rep), from wall-clock window_ms."""
    if df.height == 0:
        return []
    per_gen = df.group_by("prompt_id", "rep").agg(
        tokens=(pl.col("n_accepted") + 1).sum(), elapsed_ms=pl.col("window_ms").sum()
    )
    rates = per_gen.filter(pl.col("elapsed_ms") > 0).with_columns(
        tok_s=pl.col("tokens") / pl.col("elapsed_ms") * MS_PER_S
    )
    return [float(x) for x in rates["tok_s"]]


def compare_runs(a: pl.DataFrame, b: pl.DataFrame) -> Comparison:
    clean_a, clean_b = clean_rows(a), clean_rows(b)
    stats_a = median_iqr(per_generation_tok_s(clean_a))
    stats_b = median_iqr(per_generation_tok_s(clean_b))
    rel_diff = abs(stats_a.median - stats_b.median) / max(stats_a.median, stats_b.median)
    return Comparison(
        a=stats_a,
        b=stats_b,
        rel_diff=rel_diff,
        dirty_a=a.height - clean_a.height,
        dirty_b=b.height - clean_b.height,
    )
