"""The analysis sub-app: p3-gate, cells, fit and figures over a traces root."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from acceptrate.analysis.cli import analysis_app
from acceptrate.analysis.report import CELLS_TABLE, SUMMARY_JSON, SUMMARY_TABLE
from tests.synth_sweep import DEFAULT_SPEC, SynthSpec, write_sweep

runner = CliRunner()


@pytest.fixture(scope="module")
def sweep_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("traces") / "sweep"
    write_sweep(root, SynthSpec(tag_alpha=DEFAULT_SPEC.tag_alpha, dirty_every=9))
    return root


def test_p3_gate_reports_counts_and_fails_below_the_threshold(sweep_root: Path) -> None:
    result = runner.invoke(analysis_app, ["p3-gate", str(sweep_root)])

    assert result.exit_code == 1
    assert "clean windows" in result.output
    assert "dirty windows" in result.output
    assert "P3 gate: FAIL" in result.output


def test_p3_gate_passes_with_a_lowered_threshold(sweep_root: Path) -> None:
    result = runner.invoke(analysis_app, ["p3-gate", str(sweep_root), "--min-clean", "100"])

    assert result.exit_code == 0, result.output
    assert "P3 gate: PASS" in result.output


def test_p3_gate_on_an_empty_root_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(analysis_app, ["p3-gate", str(tmp_path)])

    assert result.exit_code == 1
    assert "no runs" in result.output


def test_cells_prints_the_per_cell_table(sweep_root: Path) -> None:
    result = runner.invoke(analysis_app, ["cells", str(sweep_root)])

    assert result.exit_code == 0, result.output
    assert "| workload_tag |" in result.output
    assert "loses" in result.output


def test_fit_writes_tables_and_reports_the_gate(sweep_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "figures"

    result = runner.invoke(analysis_app, ["fit", str(sweep_root), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "R²" in result.output
    assert "P4 gate: PASS" in result.output
    for name in (CELLS_TABLE, SUMMARY_TABLE, SUMMARY_JSON):
        assert (out / name).exists()


def test_fit_without_speculative_cells_fails_the_gate(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    write_sweep(root, SynthSpec(tag_alpha={"code": 0.8}, ks=(0,), prompts_per_tag=1))

    result = runner.invoke(analysis_app, ["fit", str(root), "--out", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "P4 gate: FAIL" in result.output


def test_figures_writes_all_four_as_pdf_and_png(sweep_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "figures"

    result = runner.invoke(analysis_app, ["figures", str(sweep_root), "--out", str(out)])

    assert result.exit_code == 0, result.output
    names = {p.name for p in out.iterdir()}
    for stem in (
        "speedup_heatmap_draft-1b",  # one heatmap per draft, slug of fake/draft-1b
        "predicted_vs_measured",
        "residuals_vs_pressure",
        "alpha_by_position",
    ):
        assert {f"{stem}.pdf", f"{stem}.png"} <= names, stem
