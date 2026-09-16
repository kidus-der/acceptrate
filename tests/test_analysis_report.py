"""write_tables publishes every cell (negatives included), a per-tag summary and summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from acceptrate.analysis.cells import cell_metrics
from acceptrate.analysis.fit import fit_report
from acceptrate.analysis.load import clean, load_sweep
from acceptrate.analysis.report import (
    CELLS_TABLE,
    SUMMARY_JSON,
    SUMMARY_TABLE,
    markdown_table,
    summary_by_tag,
    write_tables,
)
from tests.synth_sweep import DEFAULT_SPEC, write_sweep


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pl.DataFrame]:
    root = tmp_path_factory.mktemp("traces") / "sweep"
    write_sweep(root)
    cells = cell_metrics(clean(load_sweep(root)))
    out = tmp_path_factory.mktemp("out")
    write_tables(cells, fit_report(cells), out)
    return out, cells


def test_markdown_table_renders_header_separator_and_rows() -> None:
    text = markdown_table(("a", "b"), [("1", "x"), ("2", "y")])

    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |", "| 2 | y |"]


def test_cells_table_lists_every_cell_and_flags_the_losing_ones(
    written: tuple[Path, pl.DataFrame],
) -> None:
    out, cells = written
    text = (out / CELLS_TABLE).read_text()
    losing = cells.filter(pl.col("measured_speedup") < 1.0)

    assert losing.height > 0
    assert text.count("\n| ") == cells.height
    assert text.count("loses") == losing.height
    for row in losing.iter_rows(named=True):
        assert f"| {row['workload_tag']} |" in text or f"| {row['workload_tag']} " in text


def test_summary_table_has_one_row_per_tag(written: tuple[Path, pl.DataFrame]) -> None:
    out, _ = written
    text = (out / SUMMARY_TABLE).read_text()

    for tag in DEFAULT_SPEC.tag_alpha:
        assert f"| {tag} |" in text
    assert "R²" in text


def test_summary_by_tag_picks_the_best_k_and_counts_losers(
    written: tuple[Path, pl.DataFrame],
) -> None:
    _, cells = written

    summary = summary_by_tag(cells)

    assert set(summary["workload_tag"]) == set(DEFAULT_SPEC.tag_alpha)
    code = summary.filter(pl.col("workload_tag") == "code").row(0, named=True)
    assert code["best_k"] == 8
    assert code["best_speedup"] > 2.0
    assert code["losing_cells"] == 0
    chat = summary.filter(pl.col("workload_tag") == "chat").row(0, named=True)
    assert chat["losing_cells"] >= 1


def test_summary_json_carries_the_gate_and_every_cell(written: tuple[Path, pl.DataFrame]) -> None:
    out, cells = written
    summary = json.loads((out / SUMMARY_JSON).read_text())

    assert summary["p4"]["passed"] is True
    assert summary["p4"]["r2"] > 0.95
    assert summary["p4"]["threshold"] == 0.85
    assert summary["p4"]["n_cells"] == cells.filter(pl.col("k") > 0).height
    assert len(summary["cells"]) == cells.height
    assert summary["losing_cells"] == cells.filter(pl.col("measured_speedup") < 1.0).height
    assert {row["workload_tag"] for row in summary["by_tag"]} == set(DEFAULT_SPEC.tag_alpha)
