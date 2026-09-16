"""Append-only trace log: tuple rows per generation, numbered parquet parts on disk.

The hot loop appends plain tuples to the buffer `begin_generation` hands out
and never touches Polars (CLAUDE.md trap 4). `end_generation` types one
generation's rows at once; parts are written every `checkpoint_every`
generations and on close, so a crash loses at most one checkpoint's worth.

Layout of a run directory:

    traces/<run_id>/manifest.json
    traces/<run_id>/part-00001.parquet
    traces/<run_id>/part-00002.parquet
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

import polars as pl

from acceptrate.trace.manifest import RunManifest
from acceptrate.trace.schema import FIELD_NAMES, SCHEMA, WindowRow, frame_from_rows

MANIFEST_FILENAME = "manifest.json"
PART_GLOB = "part-*.parquet"
PART_NUMBER_WIDTH = 5
DEFAULT_CHECKPOINT_EVERY = 50


class MalformedRowError(ValueError):
    """A row does not have one value per schema field."""


class TraceSchemaError(ValueError):
    """A parquet part on disk does not carry the trace schema."""


def _part_path(run_dir: Path, number: int) -> Path:
    return run_dir / f"part-{number:0{PART_NUMBER_WIDTH}d}.parquet"


def _part_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob(PART_GLOB))


def _check_arity(rows: Sequence[WindowRow]) -> None:
    expected = len(FIELD_NAMES)
    for index, row in enumerate(rows):
        if len(row) != expected:
            raise MalformedRowError(f"row {index}: expected {expected} fields, got {len(row)}")


class TraceWriter:
    """Buffers typed frames between checkpoints; writes the manifest on open."""

    def __init__(
        self,
        run_dir: Path | str,
        manifest: RunManifest,
        checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    ) -> None:
        if checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
        self._run_dir = Path(run_dir)
        self._checkpoint_every = checkpoint_every
        self._pending: tuple[pl.DataFrame, ...] = ()
        self._generations_since_flush = 0
        self._parts_written = 0
        self._closed = False
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._parts_written = len(_part_paths(self._run_dir))
        write_manifest(self._run_dir, manifest)

    def begin_generation(self) -> list[WindowRow]:
        """A fresh buffer for the hot loop to append tuples to."""
        return []

    def end_generation(self, rows: Sequence[WindowRow]) -> None:
        """Type one generation's rows; checkpoint if the boundary was reached."""
        if self._closed:
            raise RuntimeError("TraceWriter is closed")
        _check_arity(rows)
        frame = frame_from_rows(rows)
        if frame.height > 0:
            self._pending = (*self._pending, frame)
        self._generations_since_flush += 1
        if self._generations_since_flush >= self._checkpoint_every:
            self._flush()

    def _flush(self) -> None:
        self._generations_since_flush = 0
        if not self._pending:
            return
        part_number = self._parts_written + 1
        pl.concat(self._pending).write_parquet(_part_path(self._run_dir, part_number))
        self._parts_written = part_number
        self._pending = ()

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._closed = True

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    (run_dir / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))


def read_manifest(run_dir: Path | str) -> RunManifest:
    return RunManifest.model_validate_json((Path(run_dir) / MANIFEST_FILENAME).read_text())


def read_run(run_dir: Path | str) -> pl.DataFrame:
    """Every part of one run, in write order, with the schema enforced."""
    parts = _part_paths(Path(run_dir))
    if not parts:
        return frame_from_rows([])
    frame = pl.scan_parquet(parts).collect()
    if frame.schema != SCHEMA:
        raise TraceSchemaError(f"{run_dir}: parts carry {frame.schema}, expected {SCHEMA}")
    return frame


def read_runs(traces_root: Path | str) -> pl.DataFrame:
    """Every run under a root (a directory holding a manifest), concatenated in name order."""
    run_dirs = sorted(p.parent for p in Path(traces_root).glob(f"*/{MANIFEST_FILENAME}"))
    frames = [read_run(run_dir) for run_dir in run_dirs]
    if not frames:
        return frame_from_rows([])
    return pl.concat(frames)
