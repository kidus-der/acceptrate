"""Load a sweep: every run's rows joined with its manifest config, plus the P3 gate.

A sweep root holds one directory per cell (`<run_id>-<stamp>/`), each with a
manifest whose `config` names the arm. The config keys become columns so
downstream code groups by (draft, k, workload_tag) without reading manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from acceptrate.bench.compare import clean_rows
from acceptrate.trace.schema import frame_from_rows
from acceptrate.trace.writer import MANIFEST_FILENAME, read_manifest, read_run

P3_MIN_CLEAN_WINDOWS = 50_000
"""The P3 gate: at least this many clean draft windows in parquet."""

CONFIG_COLUMNS: tuple[str, ...] = ("k", "arm", "draft", "target", "max_tokens")
CONFIG_DTYPES: dict[str, pl.DataType] = {
    "k": pl.Int8(),
    "arm": pl.Utf8(),
    "draft": pl.Utf8(),
    "target": pl.Utf8(),
    "max_tokens": pl.Int32(),
}


@dataclass(frozen=True)
class P3Gate:
    clean_windows: int
    dirty_windows: int
    page_in_rows: int
    """Rows with page_ins > 0 anywhere in the sweep (excluded, never hidden)."""
    page_in_rows_in_clean: int
    """Rows with page_ins > 0 that survived cleaning. Zero by construction; asserted anyway."""
    min_clean_windows: int = P3_MIN_CLEAN_WINDOWS

    @property
    def passed(self) -> bool:
        return self.clean_windows >= self.min_clean_windows and self.page_in_rows_in_clean == 0


def run_dirs(traces_root: Path | str) -> list[Path]:
    """Every run directory (one holding a manifest) directly under the root, in name order."""
    return sorted(p.parent for p in Path(traces_root).glob(f"*/{MANIFEST_FILENAME}"))


def _config_columns(config: dict) -> list[pl.Expr]:
    return [
        pl.lit(config.get(name), dtype=CONFIG_DTYPES[name]).alias(name) for name in CONFIG_COLUMNS
    ]


def load_run(run_dir: Path | str) -> pl.DataFrame:
    """One run's rows with its manifest config attached as columns."""
    config = read_manifest(run_dir).config
    return read_run(run_dir).with_columns(_config_columns(config))


def load_sweep(traces_root: Path | str) -> pl.DataFrame:
    """Every run under the root, concatenated, each row carrying its cell's config."""
    frames = [load_run(run_dir) for run_dir in run_dirs(traces_root)]
    if not frames:
        return load_empty()
    return pl.concat(frames)


def load_empty() -> pl.DataFrame:
    """A sweep frame with no rows but every column, for roots with no runs."""
    return frame_from_rows([]).with_columns(_config_columns({}))


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Windows with no page-ins, no memory pressure and no thermal throttling."""
    return clean_rows(df)


def p3_gate(df: pl.DataFrame, min_clean_windows: int = P3_MIN_CLEAN_WINDOWS) -> P3Gate:
    cleaned = clean(df)
    return P3Gate(
        clean_windows=cleaned.height,
        dirty_windows=df.height - cleaned.height,
        page_in_rows=df.filter(pl.col("page_ins") > 0).height,
        page_in_rows_in_clean=cleaned.filter(pl.col("page_ins") > 0).height,
        min_clean_windows=min_clean_windows,
    )
