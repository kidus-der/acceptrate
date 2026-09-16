"""The synthetic sweep helper writes real runs: one directory per k, readable by the trace API."""

from __future__ import annotations

from pathlib import Path

from acceptrate.trace import read_manifest, read_run
from tests.synth_sweep import DEFAULT_SPEC, SynthSpec, write_sweep


def test_write_sweep_makes_one_run_per_k_with_the_arm_config(tmp_path: Path) -> None:
    runs = write_sweep(tmp_path / "sweep")

    assert set(runs) == set(DEFAULT_SPEC.ks)
    plain = read_manifest(runs[0]).config
    spec = read_manifest(runs[4]).config
    assert (plain["arm"], plain["k"], plain["draft"]) == ("plain", 0, None)
    assert (spec["arm"], spec["k"], spec["draft"]) == ("spec", 4, DEFAULT_SPEC.draft)
    assert plain["prompts"] == spec["prompts"]


def test_every_generation_covers_max_tokens_and_windows_follow_the_model(tmp_path: Path) -> None:
    runs = write_sweep(tmp_path / "sweep")

    rows = read_run(runs[4])
    per_gen = rows.group_by("prompt_id", "rep").agg(tokens=(rows["n_accepted"] + 1).sum())
    assert per_gen["tokens"].min() >= DEFAULT_SPEC.max_tokens
    assert rows["n_accepted"].max() <= 4
    assert rows["draft_ms"].unique().to_list() == [4 * DEFAULT_SPEC.draft_ms_per_token]
    assert rows["window_ms"].unique().to_list() == [
        4 * DEFAULT_SPEC.draft_ms_per_token + DEFAULT_SPEC.verify_ms
    ]
    assert rows.filter(rows["page_ins"] > 0).height == 0


def test_dirty_every_marks_page_ins_on_some_windows(tmp_path: Path) -> None:
    spec = SynthSpec(tag_alpha={"code": 0.8}, ks=(0, 2), dirty_every=5, prompts_per_tag=1)

    runs = write_sweep(tmp_path / "sweep", spec)

    rows = read_run(runs[2])
    dirty = rows.filter(rows["page_ins"] > 0).height
    assert 0 < dirty < rows.height
