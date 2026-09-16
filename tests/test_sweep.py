"""bench/sweep.py — the grid plan: one arm per draft depth, K=0 as the interleaved baseline."""

from __future__ import annotations

import pytest

from acceptrate.bench.sweep import ArmSpec, parse_ks, sweep_arms

TARGET = "org/target-4bit"
DRAFT = "org/draft-4bit"
PROMPT_IDS = ["code-001", "prose-001"]


def test_parse_ks_accepts_ranges_and_lists() -> None:
    assert parse_ks("0-3") == (0, 1, 2, 3)
    assert parse_ks("0,2,4") == (0, 2, 4)
    assert parse_ks("0-2,6") == (0, 1, 2, 6)


def test_parse_ks_rejects_negative_and_empty() -> None:
    with pytest.raises(ValueError):
        parse_ks("")
    with pytest.raises(ValueError):
        parse_ks("-1")


def test_one_arm_per_k_with_plain_for_zero() -> None:
    arms = sweep_arms((0, 1, 4), TARGET, DRAFT, 200, PROMPT_IDS, reps=1, warmup=2)

    assert [a.k for a in arms] == [0, 1, 4]
    assert arms[0].name == "plain"
    assert arms[1].name == "spec-k1"
    assert all(isinstance(a, ArmSpec) for a in arms)


def test_configs_carry_the_full_cell_and_hash_to_distinct_run_ids() -> None:
    arms = sweep_arms((0, 1, 4), TARGET, DRAFT, 200, PROMPT_IDS, reps=1, warmup=2)

    for arm in arms:
        assert arm.config["k"] == arm.k
        assert arm.config["target"] == TARGET
        assert arm.config["max_tokens"] == 200
        assert arm.config["prompts"] == PROMPT_IDS
    assert arms[0].config["arm"] == "plain" and arms[0].config["draft"] is None
    assert arms[1].config["arm"] == "spec" and arms[1].config["draft"] == DRAFT
    assert len({a.run_id for a in arms}) == 3


def test_plain_arm_ignores_the_draft_so_baselines_are_shared_across_drafts() -> None:
    a = sweep_arms((0,), TARGET, "org/draft-a", 200, PROMPT_IDS, 1, 2)[0]
    b = sweep_arms((0,), TARGET, "org/draft-b", 200, PROMPT_IDS, 1, 2)[0]

    assert a.run_id == b.run_id


def test_requires_a_draft_for_positive_k() -> None:
    with pytest.raises(ValueError):
        sweep_arms((0, 2), TARGET, None, 200, PROMPT_IDS, 1, 2)
