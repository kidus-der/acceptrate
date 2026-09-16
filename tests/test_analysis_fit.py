"""The P4 fit: predicted vs measured speedup across cells, R² and residual breakdowns."""

from __future__ import annotations

import math

import polars as pl
import pytest

from acceptrate.analysis.cells import cell_metrics
from acceptrate.analysis.fit import (
    P4_R2_THRESHOLD,
    fit_report,
    predicted_speedup,
    r_squared,
    residuals_by,
    with_predictions,
)
from acceptrate.analysis.load import clean, load_sweep
from acceptrate.model.speedup import speedup
from tests.synth_sweep import DEFAULT_SPEC, write_sweep

SYNTHETIC_R2_FLOOR = 0.99


@pytest.fixture(scope="module")
def cells(tmp_path_factory: pytest.TempPathFactory) -> pl.DataFrame:
    root = tmp_path_factory.mktemp("traces") / "sweep"
    write_sweep(root)
    return cell_metrics(clean(load_sweep(root)))


def test_predicted_speedup_is_the_closed_form() -> None:
    assert predicted_speedup(0.8, 4, 0.2) == speedup(0.8, 4, 0.2)


def test_r_squared_is_one_for_a_perfect_fit_and_zero_for_the_mean() -> None:
    measured = [1.0, 2.0, 3.0, 4.0]

    assert r_squared(measured, measured) == pytest.approx(1.0)
    assert r_squared(measured, [2.5] * 4) == pytest.approx(0.0)


def test_r_squared_is_nan_when_measured_has_no_variance() -> None:
    assert math.isnan(r_squared([2.0, 2.0], [1.0, 3.0]))


def test_with_predictions_adds_predicted_and_residual_for_speculative_cells(
    cells: pl.DataFrame,
) -> None:
    fitted = with_predictions(cells)

    assert fitted.height == cells.filter(pl.col("k") > 0).height
    for row in fitted.iter_rows(named=True):
        assert row["predicted_speedup"] == pytest.approx(speedup(row["alpha"], row["k"], row["c"]))
        assert row["residual"] == pytest.approx(row["measured_speedup"] - row["predicted_speedup"])


def test_synthetic_sweep_fits_the_model_almost_exactly(cells: pl.DataFrame) -> None:
    report = fit_report(cells)

    assert report.n_cells == cells.filter(pl.col("k") > 0).height
    assert report.r2 > SYNTHETIC_R2_FLOOR
    assert report.threshold == P4_R2_THRESHOLD
    assert report.passed
    assert report.residuals["residual"].abs().max() < 0.1


def test_fit_report_on_no_speculative_cells_does_not_pass(cells: pl.DataFrame) -> None:
    report = fit_report(cells.filter(pl.col("k") == 0))

    assert report.n_cells == 0
    assert math.isnan(report.r2)
    assert not report.passed


def test_residuals_by_tag_and_by_k(cells: pl.DataFrame) -> None:
    by_tag = residuals_by(cells, "workload_tag")
    by_k = residuals_by(cells, "k")

    assert set(by_tag["workload_tag"]) == set(DEFAULT_SPEC.tag_alpha)
    assert set(by_k["k"]) == {k for k in DEFAULT_SPEC.ks if k > 0}
    assert set(by_tag.columns) >= {"workload_tag", "n_cells", "residual_median", "residual_mad"}
    assert by_k["n_cells"].sum() == cells.filter(pl.col("k") > 0).height
    assert (by_tag["residual_mad"] >= 0).all()


def test_fit_report_carries_both_naive_and_cost_corrected_r2(tmp_path) -> None:
    from tests.synth_sweep import DEFAULT_SPEC, write_sweep

    write_sweep(tmp_path, DEFAULT_SPEC)
    cells = cell_metrics(clean(load_sweep(tmp_path)))

    report = fit_report(cells)

    assert report.r2_naive > 0.95  # synthetic timings follow the closed form (v == 1)
    assert report.r2 > 0.95
    assert {"predicted_naive", "predicted_speedup", "residual", "v"} <= set(
        report.residuals.columns
    )


def test_corrected_prediction_divides_by_the_measured_verify_factor() -> None:
    from acceptrate.analysis.fit import predicted_speedup_corrected
    from acceptrate.model.speedup import expected_tokens

    assert predicted_speedup_corrected(0.7, 4, 0.16, 1.0) == pytest.approx(
        predicted_speedup(0.7, 4, 0.16)
    )
    assert predicted_speedup_corrected(0.7, 4, 0.16, 2.0) == pytest.approx(
        expected_tokens(0.7, 4) / (2.0 * (4 * 0.16 + 1.0))
    )
