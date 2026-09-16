"""Compare our greedy output against an external reference, token for token.

The reference itself (mlx-lm's own generation) lives in test infrastructure —
tests/reference_mlx_lm.py — because only backend/mlx_backend.py may import
mlx_lm under src/. This module knows nothing about models; it turns two
integer sequences into a report that names the first mismatch exactly.
Losslessness is exact token match, no tolerance (CLAUDE.md measurement rules).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first position where `a` and `b` differ, or None if identical.

    A sequence that is a strict prefix of the other diverges at its own length.
    """
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


@dataclass(frozen=True, slots=True)
class ReferenceReport:
    prompt_id: str
    ours: tuple[int, ...]
    reference: tuple[int, ...]
    first_divergence: int | None
    matched: bool


def compare_sequences(
    prompt_id: str, ours: Sequence[int], reference: Sequence[int]
) -> ReferenceReport:
    index = first_divergence(ours, reference)
    return ReferenceReport(
        prompt_id=prompt_id,
        ours=tuple(ours),
        reference=tuple(reference),
        first_divergence=index,
        matched=index is None,
    )


def _token_at(tokens: Sequence[int], index: int) -> str:
    return str(tokens[index]) if index < len(tokens) else "<end>"


def describe(report: ReferenceReport) -> str:
    """One line for a test failure: which prompt, where, and which two tokens."""
    if report.matched:
        return f"{report.prompt_id}: matched ({len(report.ours)} tokens)"
    index = report.first_divergence
    assert index is not None
    return (
        f"{report.prompt_id}: diverged at index {index} "
        f"ours={_token_at(report.ours, index)} reference={_token_at(report.reference, index)}"
    )
