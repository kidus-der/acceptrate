"""Pick which corpus prompts a run uses: balanced across tags, train split only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from acceptrate.trace.schema import WORKLOAD_TAGS


class SplitPrompt(Protocol):
    id: str
    tag: str
    text: str
    split: str


def select_prompts[P: SplitPrompt](
    corpus: Sequence[P], n: int, tag: str | None = None, split: str = "train"
) -> tuple[P, ...]:
    """The first `n` prompts round-robined over WORKLOAD_TAGS (or one tag), in corpus order."""
    if tag is not None and tag not in WORKLOAD_TAGS:
        raise ValueError(f"unknown workload tag {tag!r}; expected one of {WORKLOAD_TAGS}")
    tags = (tag,) if tag else WORKLOAD_TAGS
    pools = {t: [p for p in corpus if p.tag == t and p.split == split] for t in tags}
    available = sum(len(pool) for pool in pools.values())
    if n > available:
        raise ValueError(
            f"asked for {n} prompts but only {available} match tag={tag} split={split}"
        )
    chosen: list[P] = []
    round_idx = 0
    while len(chosen) < n:
        for t in tags:
            if len(chosen) == n:
                break
            if round_idx < len(pools[t]):
                chosen.append(pools[t][round_idx])
        round_idx += 1
    return tuple(chosen)
