"""The P4 figures. Matplotlib on the Agg backend; each figure lands as vector PDF + PNG.

Negative cells (speedup < 1.0) are published, outlined and labelled — never
dropped (CLAUDE.md: a negative answer is a valid result). Colour never
carries meaning alone: every tag has a legend entry, every losing cell a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")  # chosen before pyplot loads: figures never need a display

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from acceptrate.analysis.cells import CELL_KEY, DEFAULT_POSITION_BIN, SPECULATION_PARITY
from acceptrate.analysis.cells import alpha_by_position as alpha_by_position_table
from acceptrate.analysis.fit import fit_report, with_predictions
from acceptrate.trace.schema import WORKLOAD_TAGS

PNG_DPI = 150
NORM_MIN_ARM = 0.01
"""Smallest half-range of the diverging colour scale, so a one-sided grid still centres on 1.0."""

# Categorical slots in fixed order (one per workload tag), sequential/diverging poles,
# status red for "loses", and ink tokens for text — the reference palette of the
# dataviz method, validated for adjacent-pair CVD separation on a light surface.
TAG_COLOURS: dict[str, str] = dict(
    zip(
        WORKLOAD_TAGS,
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
        strict=True,
    )
)
DIVERGING_LOSS, DIVERGING_MID, DIVERGING_GAIN = "#e34948", "#f0efec", "#2a78d6"
STATUS_CRITICAL = "#d03b3b"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#9a9891"
SURFACE = "#fcfcfb"
MARK_LINE_WIDTH = 2.0
MARK_SIZE = 64


@dataclass(frozen=True)
class SpeedupGrid:
    ks: tuple[int, ...]
    tags: tuple[str, ...]
    values: np.ndarray
    """shape (len(ks), len(tags)); NaN where the sweep has no such cell."""


def _stem(path: Path | str) -> Path:
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    return stem


def _save(fig: Figure, path: Path | str) -> tuple[Path, Path]:
    stem = _stem(path)
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _style(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=INK_MUTED, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, loc="left", fontsize=11)


def _tag_colour(tag: str) -> str:
    return TAG_COLOURS.get(tag, INK_MUTED)


def speedup_grid(cells: pl.DataFrame) -> SpeedupGrid:
    """K x tag matrix of measured speedup for the sweep's one speculative arm."""
    spec = cells.filter((pl.col("k") > 0) & pl.col("measured_speedup").is_not_null())
    pairs = spec.select("target", "max_tokens", "draft").unique()
    if pairs.height > 1:
        raise ValueError(
            f"{pairs.height} (target, max_tokens, draft) combinations among speculative cells; "
            "filter to one draft before drawing a heatmap"
        )
    ks = tuple(sorted(spec["k"].unique().to_list()))
    tags = tuple(sorted(spec["workload_tag"].unique().to_list()))
    values = np.full((len(ks), len(tags)), np.nan)
    for row in spec.iter_rows(named=True):
        values[ks.index(row["k"]), tags.index(row["workload_tag"])] = row["measured_speedup"]
    return SpeedupGrid(ks=ks, tags=tags, values=values)


def _diverging_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Gray at parity; each arm spans only the range actually present (a speedup is never < 0)."""
    finite = values[np.isfinite(values)]
    lo = float(finite.min()) if finite.size else SPECULATION_PARITY
    hi = float(finite.max()) if finite.size else SPECULATION_PARITY
    return TwoSlopeNorm(
        vmin=min(lo, SPECULATION_PARITY - NORM_MIN_ARM),
        vcenter=SPECULATION_PARITY,
        vmax=max(hi, SPECULATION_PARITY + NORM_MIN_ARM),
    )


def _annotate_grid(ax: Axes, grid: SpeedupGrid, norm: TwoSlopeNorm) -> None:
    for i, _k in enumerate(grid.ks):
        for j, _tag in enumerate(grid.tags):
            value = grid.values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "n/a", ha="center", va="center", color=INK_MUTED, fontsize=8)
                continue
            far_from_parity = abs(norm(value) - 0.5) > 0.3
            ink = "#ffffff" if far_from_parity else INK_PRIMARY
            label = f"{value:.2f}" + ("\nloses" if value < SPECULATION_PARITY else "")
            ax.text(j, i, label, ha="center", va="center", color=ink, fontsize=8)
            if value < SPECULATION_PARITY:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=STATUS_CRITICAL, linewidth=2
                    )
                )


def heatmap_speedup(cells: pl.DataFrame, path: Path | str) -> tuple[Path, Path]:
    """Measured speedup per (K, workload tag); cells below 1.0 outlined and labelled 'loses'."""
    grid = speedup_grid(cells)
    norm = _diverging_norm(grid.values)
    cmap = LinearSegmentedColormap.from_list(
        "speedup", [DIVERGING_LOSS, DIVERGING_MID, DIVERGING_GAIN]
    )
    fig, ax = plt.subplots(figsize=(1.4 * len(grid.tags) + 2, 0.8 * len(grid.ks) + 2))
    image = ax.imshow(grid.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(grid.tags)), grid.tags)
    ax.set_yticks(range(len(grid.ks)), [f"K={k}" for k in grid.ks])
    _style(ax, "workload tag", "draft depth", "Measured speedup over plain decoding")
    ax.grid(False)
    _annotate_grid(ax, grid, norm)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colourbar.set_label("tok/s ratio (1.0 = parity)", color=INK_SECONDARY)
    return _save(fig, path)


def scatter_predicted_vs_measured(cells: pl.DataFrame, path: Path | str) -> tuple[Path, Path]:
    """One point per speculative cell, coloured by tag, labelled by K, with the y = x line."""
    report = fit_report(cells)
    fitted = report.residuals
    fig, ax = plt.subplots(figsize=(6, 6))
    for tag in sorted(fitted["workload_tag"].unique().to_list()):
        rows = fitted.filter(pl.col("workload_tag") == tag)
        ax.scatter(
            rows["predicted_speedup"],
            rows["measured_speedup"],
            s=MARK_SIZE,
            color=_tag_colour(tag),
            edgecolor=SURFACE,
            linewidth=1,
            label=tag,
        )
        for x, y, k in zip(
            rows["predicted_speedup"], rows["measured_speedup"], rows["k"], strict=True
        ):
            ax.annotate(f"K={k}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    values = np.concatenate([fitted["predicted_speedup"], fitted["measured_speedup"]])
    lo, hi = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
    pad = 0.05 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=INK_MUTED, linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    _style(
        ax,
        "predicted speedup  (1 - alpha^(K+1)) / ((1 - alpha)(Kc + 1))",
        "measured speedup",
        f"Predicted vs measured speedup — R² = {report.r2:.3f}, n = {report.n_cells} cells",
    )
    ax.legend(title="workload", frameon=False, fontsize=8)
    return _save(fig, path)


def _cell_pressure(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.group_by(list(CELL_KEY)).agg(
        mean_mem_pressure=pl.col("mem_pressure").mean(), mean_page_ins=pl.col("page_ins").mean()
    )


def _scatter_by_tag(ax: Axes, frame: pl.DataFrame, x: str, y: str) -> None:
    for tag in sorted(frame["workload_tag"].unique().to_list()):
        rows = frame.filter(pl.col("workload_tag") == tag)
        ax.scatter(
            rows[x],
            rows[y],
            s=MARK_SIZE,
            color=_tag_colour(tag),
            edgecolor=SURFACE,
            linewidth=1,
            label=tag,
        )


def residuals_vs_pressure(
    rows: pl.DataFrame, cells: pl.DataFrame, path: Path | str
) -> tuple[Path, Path]:
    """Residual (measured - predicted) against each cell's mean memory pressure and page-ins.

    Computed over every row of the sweep, dirty ones included: the plot exists
    to show that pressure explains nothing once the cleaning rule is applied.
    """
    fitted = with_predictions(cells).join(
        _cell_pressure(rows), on=list(CELL_KEY), how="left", nulls_equal=True
    )
    fig, (ax_mem, ax_page) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, column, label in (
        (ax_mem, "mean_mem_pressure", "mean mem_pressure over the cell's windows"),
        (ax_page, "mean_page_ins", "mean page_ins over the cell's windows"),
    ):
        ax.axhline(0.0, color=INK_MUTED, linewidth=1, linestyle="--")
        _scatter_by_tag(ax, fitted, column, "residual")
        _style(ax, label, "residual (measured - predicted)", "")
    ax_mem.set_title("Fit residuals vs system pressure", color=INK_PRIMARY, loc="left")
    ax_page.legend(title="workload", frameon=False, fontsize=8)
    return _save(fig, path)


def alpha_by_position(
    df: pl.DataFrame, path: Path | str, bin_size: int = DEFAULT_POSITION_BIN
) -> tuple[Path, Path]:
    """Per-token alpha against output position, one line per tag: the within-response drift."""
    table = alpha_by_position_table(df, bin_size)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tag in sorted(table["workload_tag"].unique().to_list()):
        rows = table.filter(pl.col("workload_tag") == tag)
        ax.plot(
            rows["pos_bin"],
            rows["alpha"],
            color=_tag_colour(tag),
            linewidth=MARK_LINE_WIDTH,
            marker="o",
            markersize=4,
            label=tag,
        )
    ax.set_ylim(0.0, 1.0)
    _style(
        ax,
        f"token position in the output (bins of {bin_size})",
        "acceptance rate alpha",
        "Acceptance rate drifts within a response",
    )
    ax.legend(title="workload", frameon=False, fontsize=8, ncol=2)
    return _save(fig, path)
