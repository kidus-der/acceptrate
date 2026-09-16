"""bench/selection.py — pick a balanced, deterministic prompt subset for a run."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from acceptrate.bench.selection import select_prompts
from acceptrate.trace.schema import WORKLOAD_TAGS


@dataclass(frozen=True)
class P:
    id: str
    tag: str
    text: str
    split: str


def _corpus() -> tuple[P, ...]:
    out = []
    for tag in WORKLOAD_TAGS:
        for i in range(6):
            out.append(P(f"{tag}-{i:03d}", tag, "x", "heldout" if i == 5 else "train"))
    return tuple(out)


def test_round_robins_across_tags_in_schema_order() -> None:
    chosen = select_prompts(_corpus(), n=8)

    assert [p.tag for p in chosen] == [*WORKLOAD_TAGS, "code", "json"]


def test_never_draws_from_the_heldout_split_by_default() -> None:
    chosen = select_prompts(_corpus(), n=30)

    assert all(p.split == "train" for p in chosen)
    assert len(chosen) == 30


def test_single_tag_filter() -> None:
    chosen = select_prompts(_corpus(), n=3, tag="json")

    assert [p.id for p in chosen] == ["json-000", "json-001", "json-002"]


def test_is_deterministic() -> None:
    assert select_prompts(_corpus(), n=10) == select_prompts(_corpus(), n=10)


def test_rejects_more_than_available() -> None:
    with pytest.raises(ValueError):
        select_prompts(_corpus(), n=31)


def test_rejects_unknown_tag() -> None:
    with pytest.raises(ValueError):
        select_prompts(_corpus(), n=1, tag="poetry")
