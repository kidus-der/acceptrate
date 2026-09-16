"""Per-cell metrics reproduce the generating alpha, the cost ratio and the closed-form speedup."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from acceptrate.analysis.cells import CELL_KEY, alpha_by_position, cell_metrics
from acceptrate.analysis.load import clean, load_sweep
from acceptrate.model.speedup import speedup
from tests.synth_sweep import DEFAULT_SPEC, write_sweep

ALPHA_TOLERANCE = 0.05
SPEEDUP_REL_TOLERANCE = 0.15
COST_RATIO = DEFAULT_SPEC.draft_ms_per_token / DEFAULT_SPEC.verify_ms
BASELINE_TOK_S = 1000.0 / DEFAULT_SPEC.verify_ms


@pytest.fixture(scope="module")
def cells(tmp_path_factory: pytest.TempPathFactory) -> pl.DataFrame:
    root = tmp_path_factory.mktemp("traces") / "sweep"
    write_sweep(root)
    return cell_metrics(clean(load_sweep(root)))


def test_one_cell_per_draft_k_and_tag(cells: pl.DataFrame) -> None:
    n_tags = len(DEFAULT_SPEC.tag_alpha)

    assert cells.height == len(DEFAULT_SPEC.ks) * n_tags
    assert cells.select(list(CELL_KEY)).n_unique() == cells.height
    assert cells["n_generations"].unique().to_list() == [
        DEFAULT_SPEC.prompts_per_tag * DEFAULT_SPEC.reps
    ]
    assert (cells["n_windows"] > 0).all()


def test_baseline_cells_have_no_alpha_and_speedup_one(cells: pl.DataFrame) -> None:
    plain = cells.filter(pl.col("k") == 0)

    assert plain["alpha"].null_count() == plain.height
    assert plain["c"].null_count() == plain.height
    assert plain["tok_s_median"].to_list() == pytest.approx([BASELINE_TOK_S] * plain.height)
    assert plain["measured_speedup"].to_list() == pytest.approx([1.0] * plain.height)


def test_cell_alpha_recovers_the_generating_per_token_alpha(cells: pl.DataFrame) -> None:
    spec_cells = cells.filter(pl.col("k") > 0)

    for tag, alpha in zip(spec_cells["workload_tag"], spec_cells["alpha"], strict=True):
        assert alpha == pytest.approx(DEFAULT_SPEC.tag_alpha[tag], abs=ALPHA_TOLERANCE), tag


def test_accept_frac_is_below_alpha_because_rejection_stops_the_window(
    cells: pl.DataFrame,
) -> None:
    spec_cells = cells.filter(pl.col("k") > 1)

    assert (spec_cells["accept_frac"] < spec_cells["alpha"]).all()


def test_cost_ratio_is_draft_per_token_over_verify(cells: pl.DataFrame) -> None:
    spec_cells = cells.filter(pl.col("k") > 0)

    assert spec_cells["c"].to_list() == pytest.approx([COST_RATIO] * spec_cells.height)


def test_measured_speedup_matches_the_closed_form_for_the_cell_alpha(cells: pl.DataFrame) -> None:
    spec_cells = cells.filter(pl.col("k") > 0)

    for row in spec_cells.iter_rows(named=True):
        predicted = speedup(row["alpha"], row["k"], row["c"])
        assert row["measured_speedup"] == pytest.approx(predicted, rel=SPEEDUP_REL_TOLERANCE), row


def test_negative_cells_are_kept(cells: pl.DataFrame) -> None:
    losing = cells.filter(pl.col("measured_speedup") < 1.0)

    assert losing.height > 0
    assert "chat" in losing["workload_tag"].to_list()


def test_iqr_is_q3_minus_q1_and_nonnegative(cells: pl.DataFrame) -> None:
    assert (cells["tok_s_iqr"] >= 0).all()
    assert cells["tok_s_iqr"].to_list() == pytest.approx(
        (cells["tok_s_q3"] - cells["tok_s_q1"]).to_list()
    )


def test_cells_without_a_baseline_have_no_measured_speedup(tmp_path: Path) -> None:
    write_sweep(tmp_path / "sweep")
    only_spec = load_sweep(tmp_path / "sweep").filter(pl.col("k") > 0)

    cells = cell_metrics(only_spec)

    assert cells["measured_speedup"].null_count() == cells.height


def test_alpha_by_position_bins_token_pos_per_tag(tmp_path: Path) -> None:
    write_sweep(tmp_path / "sweep")
    df = clean(load_sweep(tmp_path / "sweep"))

    binned = alpha_by_position(df, bin_size=32)

    assert set(binned.columns) >= {"workload_tag", "pos_bin", "alpha", "n_windows"}
    assert binned["pos_bin"].unique().sort().to_list() == [0, 32, 64, 96]
    assert set(binned["workload_tag"]) == set(DEFAULT_SPEC.tag_alpha)
    code = binned.filter(pl.col("workload_tag") == "code")
    for alpha in code["alpha"]:
        assert alpha == pytest.approx(DEFAULT_SPEC.tag_alpha["code"], abs=ALPHA_TOLERANCE)
