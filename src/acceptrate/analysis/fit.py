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

from acceptrate.model.speedup import speedup

P4_R2_THRESHOLD = 0.85
"""The P4 gate: predicted vs measured speedup R² must exceed this."""

FIT_COLUMNS: tuple[str, ...] = ("alpha", "c", "measured_speedup")


@dataclass(frozen=True)
class FitReport:
    r2: float
    n_cells: int
    residuals: pl.DataFrame
    """Every fitted cell with predicted_speedup and residual columns."""
    threshold: float = P4_R2_THRESHOLD

    @property
    def passed(self) -> bool:
        return self.r2 > self.threshold  # nan compares False: no cells, no pass


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
    predicted = [
        predicted_speedup(alpha, k, c)
        for alpha, k, c in zip(fit["alpha"], fit["k"], fit["c"], strict=True)
    ]
    return fit.with_columns(predicted_speedup=pl.Series(predicted, dtype=pl.Float64)).with_columns(
        residual=pl.col("measured_speedup") - pl.col("predicted_speedup")
    )


def fit_report(cells: pl.DataFrame) -> FitReport:
    residuals = with_predictions(cells)
    r2 = r_squared(
        residuals["measured_speedup"].to_list(), residuals["predicted_speedup"].to_list()
    )
    return FitReport(r2=r2, n_cells=residuals.height, residuals=residuals)


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
