"""Greedy-equivalence check: speculative output must be token-identical to plain decoding.

Exact match, no tolerance (brief tab 6). Framework-free: it compares token
sequences and trace rows; the composition root runs the models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from acceptrate.trace.schema import FIELD_NAMES, WindowRow

_K = FIELD_NAMES.index("k_proposed")
_N = FIELD_NAMES.index("n_accepted")


@dataclass(frozen=True)
class LosslessReport:
    prompt_id: str
    k: int
    n_tokens: int
    windows: int
    alpha: float
    """Accepted / proposed draft tokens over the generation."""
    first_divergence: int | None
    plain_token: int | None
    spec_token: int | None

    @property
    def matched(self) -> bool:
        return self.first_divergence is None


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first differing token, or None if identical (length included)."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _alpha(rows: Sequence[WindowRow]) -> float:
    proposed = sum(row[_K] for row in rows)
    if proposed == 0:
        return 0.0
    return sum(row[_N] for row in rows) / proposed


def compare_generation(
    prompt_id: str, k: int, plain: Sequence[int], spec: Sequence[int], rows: Sequence[WindowRow]
) -> LosslessReport:
    idx = first_divergence(plain, spec)
    return LosslessReport(
        prompt_id=prompt_id,
        k=k,
        n_tokens=len(plain),
        windows=len(rows),
        alpha=_alpha(rows),
        first_divergence=idx,
        plain_token=None if idx is None or idx >= len(plain) else int(plain[idx]),
        spec_token=None if idx is None or idx >= len(spec) else int(spec[idx]),
    )


def summarize(reports: Sequence[LosslessReport]) -> tuple[int, int]:
    """(matched, total)."""
    return sum(1 for r in reports if r.matched), len(reports)
