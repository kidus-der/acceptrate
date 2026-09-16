"""Markdown tables and a machine-readable summary.json for the P4 write-up.

Every cell is published, the losing ones marked — a configuration that did
not pay off is a result, not an embarrassment (CLAUDE.md).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl

from acceptrate.analysis.fit import FitReport, residuals_by, with_predictions

CELLS_TABLE = "cells.md"
SUMMARY_TABLE = "summary.md"
SUMMARY_JSON = "summary.json"
LOSES_MARK = "loses"
SPECULATION_PARITY = 1.0

CELL_COLUMNS: tuple[str, ...] = (
    "draft",
    "k",
    "workload_tag",
    "n_windows",
    "n_generations",
    "alpha",
    "accept_frac",
    "c",
    "tok_s_median",
    "tok_s_iqr",
    "measured_speedup",
    "predicted_speedup",
    "residual",
)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def _speedup_cell(value: float | None) -> str:
    if value is None:
        return "-"
    if value < SPECULATION_PARITY:
        return f"**{value:.3f}** ({LOSES_MARK})"
    return f"{value:.3f}"


def cells_with_predictions(cells: pl.DataFrame) -> pl.DataFrame:
    """All cells, baseline rows included, with predicted_speedup and residual where fittable."""
    fitted = with_predictions(cells).select(
        "target", "max_tokens", "draft", "k", "workload_tag", "predicted_speedup", "residual"
    )
    return cells.join(
        fitted,
        on=["target", "max_tokens", "draft", "k", "workload_tag"],
        how="left",
        nulls_equal=True,
    ).sort("target", "max_tokens", "draft", "k", "workload_tag", nulls_last=False)


def cells_table(cells: pl.DataFrame) -> str:
    full = cells_with_predictions(cells)
    rows = []
    for row in full.iter_rows(named=True):
        cells_out = [
            _speedup_cell(row[name]) if name == "measured_speedup" else _fmt(row[name])
            for name in CELL_COLUMNS
        ]
        rows.append(cells_out)
    return markdown_table(CELL_COLUMNS, rows)


def summary_by_tag(cells: pl.DataFrame) -> pl.DataFrame:
    """Per tag: the best K and its speedup, the worst cell, how many cells lose, alpha range."""
    spec = cells.filter((pl.col("k") > 0) & pl.col("measured_speedup").is_not_null())
    return (
        spec.sort("measured_speedup", descending=True)
        .group_by("workload_tag")
        .agg(
            n_cells=pl.len(),
            best_k=pl.col("k").first(),
            best_speedup=pl.col("measured_speedup").first(),
            worst_speedup=pl.col("measured_speedup").min(),
            losing_cells=(pl.col("measured_speedup") < SPECULATION_PARITY).sum(),
            alpha_min=pl.col("alpha").min(),
            alpha_max=pl.col("alpha").max(),
        )
        .sort("workload_tag")
    )


def summary_table(cells: pl.DataFrame, fit: FitReport) -> str:
    by_tag = summary_by_tag(cells)
    headers = tuple(by_tag.columns)
    table = markdown_table(headers, ([_fmt(v) for v in row] for row in by_tag.iter_rows()))
    verdict = "PASS" if fit.passed else "FAIL"
    heading = (
        f"P4 fit: R² = {_fmt(fit.r2)} over {fit.n_cells} cells "
        f"(gate > {fit.threshold}): **{verdict}**"
    )
    return f"{heading}\n\n{table}"


def _json_safe(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records(frame: pl.DataFrame) -> list[dict]:
    return [{k: _json_safe(v) for k, v in row.items()} for row in frame.iter_rows(named=True)]


def summary_dict(cells: pl.DataFrame, fit: FitReport) -> dict:
    full = cells_with_predictions(cells)
    return {
        "p4": {
            "r2": _json_safe(fit.r2),
            "n_cells": fit.n_cells,
            "threshold": fit.threshold,
            "passed": fit.passed,
        },
        "losing_cells": full.filter(pl.col("measured_speedup") < SPECULATION_PARITY).height,
        "by_tag": _records(summary_by_tag(cells)),
        "residuals_by_tag": _records(residuals_by(cells, "workload_tag")),
        "residuals_by_k": _records(residuals_by(cells, "k")),
        "cells": _records(full),
    }


def write_tables(cells: pl.DataFrame, fit: FitReport, out_dir: Path | str) -> tuple[Path, ...]:
    """cells.md, summary.md and summary.json under out_dir; returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = (
        (out / CELLS_TABLE, cells_table(cells) + "\n"),
        (out / SUMMARY_TABLE, summary_table(cells, fit) + "\n"),
        (out / SUMMARY_JSON, json.dumps(summary_dict(cells, fit), indent=2) + "\n"),
    )
    for path, text in targets:
        path.write_text(text)
    return tuple(path for path, _ in targets)
