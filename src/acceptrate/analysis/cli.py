"""`acceptrate analysis ...`: the P3 gate count, per-cell metrics, the P4 fit and its figures.

A Typer sub-app; the composition root (cli.py) mounts it. Nothing here loads
a model — it only reads parquet a sweep already wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from acceptrate.analysis.cells import cell_metrics
from acceptrate.analysis.figures import (
    alpha_by_position,
    heatmap_speedup,
    residuals_vs_pressure,
    scatter_predicted_vs_measured,
)
from acceptrate.analysis.fit import fit_report
from acceptrate.analysis.load import P3_MIN_CLEAN_WINDOWS, clean, load_sweep, p3_gate
from acceptrate.analysis.report import cells_table, write_tables

analysis_app = typer.Typer(no_args_is_help=True, help="P3/P4 analysis over recorded traces.")

DEFAULT_FIGURES_DIR = Path("docs/figures")
FIGURE_STEMS = {
    "heatmap": "speedup_heatmap",
    "scatter": "predicted_vs_measured",
    "pressure": "residuals_vs_pressure",
    "alpha": "alpha_by_position",
}

TracesRoot = Annotated[Path, typer.Argument(help="Directory holding one run directory per cell.")]
OutDir = Annotated[Path, typer.Option("--out", help="Where tables and figures are written.")]


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _load(traces_root: Path) -> pl.DataFrame:
    rows = load_sweep(traces_root)
    if rows.height == 0:
        _fail(f"no runs with rows under {traces_root}")
    return rows


@analysis_app.command("p3-gate")
def p3_gate_command(
    traces_root: TracesRoot,
    min_clean: Annotated[
        int, typer.Option(help="Clean windows required to pass.")
    ] = P3_MIN_CLEAN_WINDOWS,
) -> None:
    """P3 gate: enough clean draft windows, and no page-in rows among them."""
    gate = p3_gate(_load(traces_root), min_clean_windows=min_clean)
    typer.echo(f"clean windows: {gate.clean_windows} (need {gate.min_clean_windows})")
    typer.echo(f"dirty windows: {gate.dirty_windows} (page-in rows: {gate.page_in_rows})")
    typer.echo(f"page-in rows among clean: {gate.page_in_rows_in_clean}")
    if not gate.passed:
        _fail("P3 gate: FAIL")
    typer.echo("P3 gate: PASS")


@analysis_app.command("cells")
def cells_command(traces_root: TracesRoot) -> None:
    """Per (draft, K, tag) cell: alpha, cost ratio, tok/s median and IQR, measured speedup."""
    typer.echo(cells_table(cell_metrics(clean(_load(traces_root)))))


@analysis_app.command("fit")
def fit_command(traces_root: TracesRoot, out: OutDir = DEFAULT_FIGURES_DIR) -> None:
    """P4 gate: R² of predicted vs measured speedup across cells; writes the tables."""
    cells = cell_metrics(clean(_load(traces_root)))
    fit = fit_report(cells)
    written = write_tables(cells, fit, out)
    typer.echo(
        f"R² = {fit.r2:.4f} over {fit.n_cells} speculative cells (gate > {fit.threshold}); "
        f"closed form as written (v = 1): R² = {fit.r2_naive:.4f}"
    )
    for path in written:
        typer.echo(f"  -> {path}")
    if not fit.passed:
        _fail("P4 gate: FAIL")
    typer.echo("P4 gate: PASS")


@analysis_app.command("figures")
def figures_command(traces_root: TracesRoot, out: OutDir = DEFAULT_FIGURES_DIR) -> None:
    """Heatmap, predicted-vs-measured scatter, residuals vs pressure, alpha by position."""
    rows = _load(traces_root)
    cleaned = clean(rows)
    cells = cell_metrics(cleaned)
    written: tuple[Path, ...] = ()
    drafts = sorted(d for d in cells["draft"].unique().to_list() if d is not None)
    for draft in drafts:
        # one grid per draft, from its most recent session (a draft may have been swept twice)
        session = cells.filter(pl.col("draft") == draft)["session"].max()
        subset = cells.filter(
            (pl.col("session") == session)
            & (pl.col("draft").is_null() | (pl.col("draft") == draft))
        )
        slug = draft.rsplit("/", 1)[-1].replace(".", "_")
        stem = f"{FIGURE_STEMS['heatmap']}_{slug}"
        written += heatmap_speedup(subset, out / stem)
    written += (
        *scatter_predicted_vs_measured(cells, out / FIGURE_STEMS["scatter"]),
        *residuals_vs_pressure(rows, cells, out / FIGURE_STEMS["pressure"]),
        *alpha_by_position(cleaned, out / FIGURE_STEMS["alpha"]),
    )
    for path in written:
        typer.echo(f"  -> {path}")
