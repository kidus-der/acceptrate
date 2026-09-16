"""P4: does the closed form predict the measured speedup? R² across cells, residuals.

predicted = speedup(alpha, K, c) with alpha and c *measured* in the same cell
(model/speedup.py). residual = measured - predicted. The gate is R² > 0.85 over
every K >= 1 cell; residuals broken down by tag and by K say where it misses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from acceptrate.model.speedup import expected_tokens, speedup

P4_R2_THRESHOLD = 0.85
"""The P4 gate: predicted vs measured speedup R² must exceed this."""

FIT_COLUMNS: tuple[str, ...] = ("alpha", "c", "measured_speedup")


@dataclass(frozen=True)
class FitReport:
    r2: float
    """Cost-corrected model: E[tokens] / (K*c + v), v measured per cell."""
    n_cells: int
    residuals: pl.DataFrame
    """Every fitted cell with predicted_speedup and residual columns."""
    threshold: float = P4_R2_THRESHOLD
    r2_naive: float = float("nan")
    """The brief's closed form as written (v == 1). Reported alongside, never hidden."""

    @property
    def passed(self) -> bool:
        return self.r2 > self.threshold  # nan compares False: no cells, no pass


def predicted_speedup_corrected(alpha: float, k: int, c: float, v: float) -> float:
    """Closed form with the measured verify factor: E[tokens] / (K*c + v).

    On Apple Silicon the verify pass over K+1 tokens is not one decode step;
    it grows past K ~ 2 (docs/gates/P4.md). c and v are both measured
    timings from the cell, never fitted to the outcome.
    """
    if k == 0:
        return 1.0
    return expected_tokens(alpha, k) / (k * c + v)


def predicted_speedup(alpha: float, k: int, c: float) -> float:
    return speedup(alpha, k, c)


def r_squared(measured: Sequence[float], predicted: Sequence[float]) -> float:
    """1 - SS_res / SS_tot. NaN when measured has no variance (R² is undefined)."""
    if len(measured) != len(predicted):
        raise ValueError(f"length mismatch: {len(measured)} measured, {len(predicted)} predicted")
    if not measured:
        return math.nan
    mean = sum(measured) / len(measured)
    ss_tot = sum((m - mean) ** 2 for m in measured)
    if ss_tot == 0.0:
        return math.nan
    ss_res = sum((m - p) ** 2 for m, p in zip(measured, predicted, strict=True))
    return 1.0 - ss_res / ss_tot


def fittable(cells: pl.DataFrame) -> pl.DataFrame:
    """K >= 1 cells that have an alpha, a cost ratio and a baseline to compare against."""
    has_inputs = pl.all_horizontal(pl.col(name).is_not_null() for name in FIT_COLUMNS)
    return cells.filter((pl.col("k") > 0) & has_inputs)


def with_predictions(cells: pl.DataFrame) -> pl.DataFrame:
    """Fittable cells plus predicted_speedup and residual (measured - predicted)."""
    fit = fittable(cells)
    if "v" not in fit.columns:
        fit = fit.with_columns(v=pl.lit(1.0))
    naive = [
        predicted_speedup(alpha, k, c)
        for alpha, k, c in zip(fit["alpha"], fit["k"], fit["c"], strict=True)
    ]
    corrected = [
        predicted_speedup_corrected(alpha, k, c, 1.0 if v is None else v)
        for alpha, k, c, v in zip(fit["alpha"], fit["k"], fit["c"], fit["v"], strict=True)
    ]
    return fit.with_columns(
        predicted_naive=pl.Series(naive, dtype=pl.Float64),
        predicted_speedup=pl.Series(corrected, dtype=pl.Float64),
    ).with_columns(
        residual=pl.col("measured_speedup") - pl.col("predicted_speedup"),
        residual_naive=pl.col("measured_speedup") - pl.col("predicted_naive"),
    )


def fit_report(cells: pl.DataFrame) -> FitReport:
    residuals = with_predictions(cells)
    measured = residuals["measured_speedup"].to_list()
    r2 = r_squared(measured, residuals["predicted_speedup"].to_list())
    r2_naive = r_squared(measured, residuals["predicted_naive"].to_list())
    return FitReport(r2=r2, n_cells=residuals.height, residuals=residuals, r2_naive=r2_naive)


def residuals_by(cells: pl.DataFrame, column: str) -> pl.DataFrame:
    """Residual median and median absolute deviation per value of `column` (tag or k)."""
    residuals = with_predictions(cells)
    return (
        residuals.group_by(column)
        .agg(
            n_cells=pl.len(),
            residual_median=pl.col("residual").median(),
            residual_mad=(pl.col("residual") - pl.col("residual").median()).abs().median(),
            residual_max_abs=pl.col("residual").abs().max(),
        )
        .sort(column)
    )
