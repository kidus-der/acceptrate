"""TraceWriter: rows buffer per generation and land in numbered parquet parts."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from acceptrate.trace.manifest import RunManifest, build_manifest
from acceptrate.trace.schema import SCHEMA, WindowRow, make_row
from acceptrate.trace.writer import (
    MalformedRowError,
    TraceWriter,
    read_manifest,
    read_run,
    read_runs,
)


def _manifest(config: dict) -> RunManifest:
    return build_manifest(
        config,
        now=lambda: "2026-09-16T12:00:00+00:00",
        chip=lambda: "Apple M4",
        memory_gb=lambda: 16.0,
        git_sha=lambda: "abc123",
        package_version=lambda name: "0",
    )


def _rows(run_id: str, count: int, first_window: int = 0) -> list[WindowRow]:
    return [
        make_row(run_id, first_window + i, i * 4, 4, 2, 8.0, 50.0, "code", 0, 0, 0, "p1", 0, 60.0)
        for i in range(count)
    ]


def _parts(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*.parquet"))


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "traces" / "run-a"


def test_opening_the_writer_writes_a_readable_manifest(run_dir: Path) -> None:
    manifest = _manifest({"k": 4})

    with TraceWriter(run_dir, manifest):
        pass

    assert (run_dir / "manifest.json").exists()
    assert read_manifest(run_dir) == manifest


def test_parts_carry_exactly_the_schema(run_dir: Path) -> None:
    with TraceWriter(run_dir, _manifest({"k": 4})) as writer:
        buffer = writer.begin_generation()
        buffer.extend(_rows("r", 3))
        writer.end_generation(buffer)

    parts = _parts(run_dir)
    assert [p.name for p in parts] == ["part-00001.parquet"]
    assert pl.read_parquet(parts[0]).schema == SCHEMA


def test_nothing_is_written_before_the_checkpoint_boundary(run_dir: Path) -> None:
    writer = TraceWriter(run_dir, _manifest({"k": 4}), checkpoint_every=2)

    writer.end_generation(_rows("r", 3))

    assert _parts(run_dir) == []
    writer.close()


def test_checkpointing_splits_parts_at_the_generation_boundary(run_dir: Path) -> None:
    writer = TraceWriter(run_dir, _manifest({"k": 4}), checkpoint_every=2)

    for generation in range(5):
        writer.end_generation(_rows("r", 3, first_window=generation * 3))
    written_before_close = [p.name for p in _parts(run_dir)]
    writer.close()

    assert written_before_close == ["part-00001.parquet", "part-00002.parquet"]
    heights = [pl.read_parquet(p).height for p in _parts(run_dir)]
    assert heights == [6, 6, 3]


def test_close_is_idempotent_and_flushes_only_once(run_dir: Path) -> None:
    writer = TraceWriter(run_dir, _manifest({"k": 4}))
    writer.end_generation(_rows("r", 2))

    writer.close()
    writer.close()

    assert len(_parts(run_dir)) == 1


def test_empty_generation_yields_no_rows_and_no_error(run_dir: Path) -> None:
    with TraceWriter(run_dir, _manifest({"k": 4})) as writer:
        writer.end_generation(writer.begin_generation())

    assert _parts(run_dir) == []
    assert read_run(run_dir).height == 0
    assert read_run(run_dir).schema == SCHEMA


def test_wrong_arity_row_is_rejected_with_a_clear_error(run_dir: Path) -> None:
    writer = TraceWriter(run_dir, _manifest({"k": 4}))
    short_row = ("r", 0, 0)

    with pytest.raises(MalformedRowError, match=r"expected 14 fields, got 3"):
        writer.end_generation([*_rows("r", 1), short_row])  # type: ignore[list-item]
    writer.close()


def test_writing_after_close_is_an_error(run_dir: Path) -> None:
    writer = TraceWriter(run_dir, _manifest({"k": 4}))
    writer.close()

    with pytest.raises(RuntimeError, match="closed"):
        writer.end_generation(_rows("r", 1))


def test_checkpoint_every_must_be_positive(run_dir: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint_every"):
        TraceWriter(run_dir, _manifest({"k": 4}), checkpoint_every=0)


def test_read_run_returns_rows_in_write_order_across_parts(run_dir: Path) -> None:
    with TraceWriter(run_dir, _manifest({"k": 4}), checkpoint_every=1) as writer:
        for generation in range(12):
            writer.end_generation(_rows("r", 2, first_window=generation * 2))

    frame = read_run(run_dir)

    assert len(_parts(run_dir)) == 12
    assert frame.schema == SCHEMA
    assert frame["window_idx"].to_list() == list(range(24))


def test_reopening_a_run_dir_appends_new_parts_after_the_existing_ones(run_dir: Path) -> None:
    manifest = _manifest({"k": 4})
    with TraceWriter(run_dir, manifest) as writer:
        writer.end_generation(_rows("r", 2))

    with TraceWriter(run_dir, manifest) as writer:
        writer.end_generation(_rows("r", 3, first_window=2))

    assert [p.name for p in _parts(run_dir)] == ["part-00001.parquet", "part-00002.parquet"]
    assert read_run(run_dir)["window_idx"].to_list() == [0, 1, 2, 3, 4]


def test_read_runs_concatenates_every_run_under_a_root(tmp_path: Path) -> None:
    root = tmp_path / "traces"
    with TraceWriter(root / "run-a", _manifest({"k": 2})) as writer:
        writer.end_generation(_rows("a", 2))
    with TraceWriter(root / "run-b", _manifest({"k": 8})) as writer:
        writer.end_generation(_rows("b", 3))

    frame = read_runs(root)

    assert frame.schema == SCHEMA
    assert frame["run_id"].to_list() == ["a", "a", "b", "b", "b"]


def test_read_runs_on_an_empty_root_is_an_empty_frame_with_the_schema(tmp_path: Path) -> None:
    frame = read_runs(tmp_path)

    assert frame.height == 0
    assert frame.schema == SCHEMA
