"""weights.py — nominal on-disk size of cached weights, for the memory-fit guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from acceptrate.config import ModelSpec
from acceptrate.weights import cached_spec, cached_weights_gb

GB = 1024**3


def _snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"\0" * (2 * 1024 * 1024))
    (snap / "config.json").write_text("{}")
    return snap


def test_sums_every_file_in_the_snapshot(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)

    gb = cached_weights_gb("org/model", resolve=lambda repo: snap)

    assert gb == pytest.approx((2 * 1024 * 1024 + 2) / GB)


def test_cached_spec_builds_a_model_spec(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)

    spec = cached_spec("org/model", resolve=lambda repo: snap)

    assert isinstance(spec, ModelSpec)
    assert spec.repo == "org/model"
    assert spec.weights_gb > 0


def test_missing_snapshot_raises_a_clear_error(tmp_path: Path) -> None:
    def resolve(repo: str) -> Path:
        raise FileNotFoundError(repo)

    with pytest.raises(FileNotFoundError):
        cached_weights_gb("org/missing", resolve=resolve)
