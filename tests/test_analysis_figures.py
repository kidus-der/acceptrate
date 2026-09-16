"""Every figure writes a vector PDF and a PNG that exist and are non-empty; negatives stay in."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from acceptrate.analysis.cells import cell_metrics
from acceptrate.analysis.figures import (
    alpha_by_position,
    heatmap_speedup,
    residuals_vs_pressure,
    scatter_predicted_vs_measured,
    speedup_grid,
)
from acceptrate.analysis.load import clean, load_sweep
from tests.synth_sweep import DEFAULT_SPEC, SynthSpec, write_sweep


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> tuple[pl.DataFrame, pl.DataFrame]:
    root = tmp_path_factory.mktemp("traces") / "sweep"
    write_sweep(root, SynthSpec(tag_alpha=DEFAULT_SPEC.tag_alpha, dirty_every=9))
    rows = load_sweep(root)
    return rows, cell_metrics(clean(rows))


def _assert_written(paths: tuple[Path, ...]) -> None:
    assert {p.suffix for p in paths} == {".pdf", ".png"}
    for path in paths:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_speedup_grid_keeps_losing_cells(data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    _, cells = data

    grid = speedup_grid(cells)

    assert grid.ks == tuple(k for k in DEFAULT_SPEC.ks if k > 0)
    assert grid.tags == tuple(sorted(DEFAULT_SPEC.tag_alpha))
    assert grid.values.shape == (len(grid.ks), len(grid.tags))
    assert (grid.values < 1.0).any()


def test_speedup_grid_refuses_ambiguous_cells(data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    _, cells = data
    other_draft = cells.with_columns(draft=pl.lit("fake/other-draft"))

    with pytest.raises(ValueError, match="draft"):
        speedup_grid(pl.concat([cells, other_draft]))


def test_heatmap_writes_pdf_and_png(
    data: tuple[pl.DataFrame, pl.DataFrame], tmp_path: Path
) -> None:
    _, cells = data

    _assert_written(heatmap_speedup(cells, tmp_path / "heatmap"))


def test_scatter_writes_pdf_and_png(
    data: tuple[pl.DataFrame, pl.DataFrame], tmp_path: Path
) -> None:
    _, cells = data

    _assert_written(scatter_predicted_vs_measured(cells, tmp_path / "scatter"))


def test_residuals_vs_pressure_writes_pdf_and_png(
    data: tuple[pl.DataFrame, pl.DataFrame], tmp_path: Path
) -> None:
    rows, cells = data

    _assert_written(residuals_vs_pressure(rows, cells, tmp_path / "pressure"))


def test_alpha_by_position_writes_pdf_and_png(
    data: tuple[pl.DataFrame, pl.DataFrame], tmp_path: Path
) -> None:
    rows, _ = data

    _assert_written(alpha_by_position(clean(rows), tmp_path / "nested" / "alpha_pos"))


def test_figures_land_under_the_given_stem_even_with_a_suffix(
    data: tuple[pl.DataFrame, pl.DataFrame], tmp_path: Path
) -> None:
    _, cells = data

    paths = heatmap_speedup(cells, tmp_path / "heatmap.png")

    assert {p.name for p in paths} == {"heatmap.pdf", "heatmap.png"}
