"""load_sweep joins rows with their manifest config; clean/p3_gate count what the gate counts."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from acceptrate.analysis.load import (
    CONFIG_COLUMNS,
    P3_MIN_CLEAN_WINDOWS,
    clean,
    load_sweep,
    p3_gate,
)
from acceptrate.trace import read_run
from acceptrate.trace.schema import SCHEMA, make_row
from tests.synth_sweep import DEFAULT_SPEC, SynthSpec, write_sweep


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[int, Path]]:
    root = tmp_path_factory.mktemp("traces") / "sweep"
    spec = SynthSpec(tag_alpha=DEFAULT_SPEC.tag_alpha, dirty_every=7)
    return root, write_sweep(root, spec)


def test_load_sweep_adds_the_config_columns_to_every_row(
    sweep: tuple[Path, dict[int, Path]],
) -> None:
    root, runs = sweep

    df = load_sweep(root)

    assert set(CONFIG_COLUMNS) <= set(df.columns)
    assert set(SCHEMA.names()) <= set(df.columns)
    assert df.height == sum(read_run(run_dir).height for run_dir in runs.values())
    assert sorted(df["k"].unique().to_list()) == sorted(DEFAULT_SPEC.ks)
    plain = df.filter(pl.col("k") == 0)
    assert plain["arm"].unique().to_list() == ["plain"]
    assert plain["draft"].null_count() == plain.height
    spec_rows = df.filter(pl.col("k") == 4)
    assert spec_rows["draft"].unique().to_list() == [DEFAULT_SPEC.draft]
    assert df["max_tokens"].unique().to_list() == [DEFAULT_SPEC.max_tokens]


def test_load_sweep_of_an_empty_root_is_an_empty_typed_frame(tmp_path: Path) -> None:
    df = load_sweep(tmp_path)

    assert df.height == 0
    assert set(CONFIG_COLUMNS) <= set(df.columns)


def test_clean_drops_every_kind_of_dirty_window() -> None:
    def row(mem: int, page_ins: int, thermal: int) -> tuple:
        return make_row("r", 0, 0, 4, 2, 40.0, 50.0, "code", mem, page_ins, thermal, "p", 0, 90.0)

    df = pl.DataFrame(
        [row(0, 0, 0), row(1, 0, 0), row(0, 5, 0), row(0, 0, 1), row(0, 0, 0)],
        schema=SCHEMA,
        orient="row",
    )

    assert clean(df).height == 2


def test_p3_gate_counts_clean_and_dirty_windows(sweep: tuple[Path, dict[int, Path]]) -> None:
    root, _ = sweep
    df = load_sweep(root)

    gate = p3_gate(df)

    assert gate.clean_windows == clean(df).height
    assert gate.dirty_windows == df.height - gate.clean_windows
    assert gate.dirty_windows > 0
    assert gate.page_in_rows == df.filter(pl.col("page_ins") > 0).height
    assert gate.page_in_rows_in_clean == 0
    assert gate.min_clean_windows == P3_MIN_CLEAN_WINDOWS
    assert not gate.passed  # a small synthetic sweep is far below 50 000 windows


def test_p3_gate_passes_when_the_clean_count_clears_the_threshold(
    sweep: tuple[Path, dict[int, Path]],
) -> None:
    root, _ = sweep
    df = load_sweep(root)

    gate = p3_gate(df, min_clean_windows=100)

    assert gate.clean_windows >= 100
    assert gate.passed
