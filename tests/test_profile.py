"""profile.py — the per-machine profile `acceptrate calibrate` writes and `serve` reads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from acceptrate.profile import (
    MachineProfile,
    default_profile_path,
    load_profile,
    measure_profile,
    save_profile,
    scheduler_priors,
)
from tests.fakes import FakeBackend


@dataclass(frozen=True)
class Snap:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


def test_measure_profile_records_baseline_c_and_alpha_from_short_runs() -> None:
    target = FakeBackend(32, step=7)
    draft = FakeBackend(32, step=7)  # perfect draft -> alpha 1.0

    profile = measure_profile(
        target, draft, prompts=[[3], [5, 6]], ks=(2, 4), max_tokens=24, guard=Snap,
        chip="fake", mlx_version="0", target_repo="t", draft_repo="d",
    )  # fmt: skip

    assert isinstance(profile, MachineProfile)
    assert profile.baseline_tok_s > 0
    assert set(profile.c_by_k) == {2, 4}
    assert all(c >= 0 for c in profile.c_by_k.values())
    assert profile.alpha_prior == pytest.approx(1.0)
    assert profile.chip == "fake" and profile.target == "t" and profile.draft == "d"


def test_alpha_prior_reflects_a_bad_draft() -> None:
    profile = measure_profile(
        FakeBackend(32, step=7), FakeBackend(32, step=3), prompts=[[3]], ks=(4,), max_tokens=24,
        guard=Snap, chip="f", mlx_version="0", target_repo="t", draft_repo="d",
    )  # fmt: skip

    assert profile.alpha_prior == pytest.approx(0.0, abs=0.05)


def test_roundtrip_through_json(tmp_path: Path) -> None:
    profile = MachineProfile(
        "Apple M4", "0.32.2", "t", "d", "2026-09-16T00:00:00Z", 20.5, {1: 0.19, 4: 0.12}, 0.7
    )
    path = tmp_path / "profile.json"

    save_profile(profile, path)

    assert json.loads(path.read_text())["c_by_k"] == {"1": 0.19, "4": 0.12}
    assert load_profile(path) == profile


def test_scheduler_priors_use_the_profile_c_at_the_requested_k_or_the_median() -> None:
    profile = MachineProfile("c", "m", "t", "d", "now", 20.0, {1: 0.2, 4: 0.1}, 0.7)

    alpha, c4 = scheduler_priors(profile, k=4)
    alpha2, c_any = scheduler_priors(profile, k=8)

    assert (alpha, c4) == (0.7, 0.1)
    assert (alpha2, c_any) == (0.7, pytest.approx(0.15))


def test_default_path_is_under_the_home_dot_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert default_profile_path() == tmp_path / ".acceptrate" / "profile.json"


def test_load_rejects_a_profile_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"chip": "x"}))

    with pytest.raises(ValueError):
        load_profile(path)
