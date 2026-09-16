"""verify/noise_floor.py — measure how far batched-verify and sequential-decode logits disagree.

On Metal fp16 the two kernel paths differ by ~0.03 typical / ~0.10 worst
case (docs/gates/P2.md). That floor is what makes a greedy divergence at a
near-tie platform numerics rather than a bug. It is measured through the
Backend protocol only, so this file is model-free and CI-safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from acceptrate.verify.noise_floor import (
    NoiseFloor,
    is_near_tie,
    load_noise_floor,
    measure_noise_floor,
    save_noise_floor,
)
from tests.fakes import FakeBackend

FLOOR = NoiseFloor(
    chip="M4", mlx_version="0.32.2", positions=400, p50=0.03, p99=0.075, max=0.1, argmax_flips=0
)


class NoisyFake(FakeBackend):
    """Batched rows carry +0.05 on one logit that the sequential path does not."""

    def verify(self, tokens):
        rows = super().verify(tokens)
        rows[:, 3] += 0.05
        return rows


def test_exact_backend_has_zero_floor() -> None:
    floor = measure_noise_floor(FakeBackend(32), prompts=[[1, 2], [5]], gen_tokens=20, window=5)

    assert isinstance(floor, NoiseFloor)
    assert floor.positions == 40
    assert floor.p50 == floor.p99 == floor.max == 0.0
    assert floor.argmax_flips == 0


def test_noisy_backend_reports_the_injected_delta_as_the_floor() -> None:
    floor = measure_noise_floor(NoisyFake(32), prompts=[[1, 2]], gen_tokens=20, window=5)

    assert floor.max == pytest.approx(0.05)
    assert floor.p50 == pytest.approx(0.05)


def test_windows_cover_every_generated_position_exactly_once() -> None:
    floor = measure_noise_floor(FakeBackend(32), prompts=[[1]], gen_tokens=23, window=5)

    assert floor.positions == 23


def test_is_near_tie_compares_margin_to_floor() -> None:
    assert is_near_tie(0.0, FLOOR)
    assert is_near_tie(0.1, FLOOR)
    assert not is_near_tie(0.11, FLOOR)


def test_roundtrip_through_json(tmp_path: Path) -> None:
    path = tmp_path / "noise_floor.json"

    save_noise_floor(FLOOR, path)

    assert json.loads(path.read_text())["max"] == 0.1
    assert load_noise_floor(path) == FLOOR


def test_load_rejects_a_file_without_a_max(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"chip": "M4"}))

    with pytest.raises(ValueError):
        load_noise_floor(path)


def test_measure_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        measure_noise_floor(FakeBackend(32), prompts=[[1]], gen_tokens=5, window=0)


def test_measure_uses_finite_float_rows() -> None:
    floor = measure_noise_floor(FakeBackend(32), prompts=[[1]], gen_tokens=5, window=5)

    assert np.isfinite(floor.max)
