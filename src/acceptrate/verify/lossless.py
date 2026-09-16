"""Greedy-equivalence check: speculative output must be token-identical to plain decoding.

Exact match is the definition. The gate's *evaluation* recognises one
platform fact (docs/gates/P2.md): where the target's top-2 logits are within
the measured cross-kernel noise floor, batched verification and sequential
decoding are free to disagree, and such a divergence is a near-tie, not a
failure. Every near-tie is reported; any other divergence fails the gate.

Framework-free: it works on token sequences, trace rows and the Backend
protocol; the composition root runs the models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from acceptrate.backend.protocol import Backend
from acceptrate.trace.schema import FIELD_NAMES, WindowRow
from acceptrate.verify.noise_floor import NoiseFloor, is_near_tie

_K = FIELD_NAMES.index("k_proposed")
_N = FIELD_NAMES.index("n_accepted")


class Verdict(Enum):
    IDENTICAL = "identical"
    NEAR_TIE = "near_tie"
    DIVERGENT = "divergent"


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
    margin: float | None = None
    """Sequential top-2 logit margin at the divergence, if one was measured."""

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
    prompt_id: str,
    k: int,
    plain: Sequence[int],
    spec: Sequence[int],
    rows: Sequence[WindowRow],
    margin: float | None = None,
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
        margin=margin,
    )


def sequential_margin_at(
    target: Backend, prompt: Sequence[int], plain: Sequence[int], index: int
) -> float:
    """Replay plain decoding to `index` and return logit(top1) - logit(top2) there."""
    logits = target.prefill(prompt)
    for token in plain[:index]:
        logits = target.decode_step(token)
    top2 = np.partition(logits, -2)[-2:]
    return float(top2[1] - top2[0])


def classify(report: LosslessReport, floor: NoiseFloor) -> Verdict:
    if report.matched:
        return Verdict.IDENTICAL
    if report.margin is not None and is_near_tie(report.margin, floor):
        return Verdict.NEAR_TIE
    return Verdict.DIVERGENT


def gate_verdicts(
    reports: Sequence[LosslessReport], floor: NoiseFloor
) -> tuple[bool, dict[Verdict, int]]:
    """(passed, counts). Passes iff no report is DIVERGENT."""
    counts = {v: 0 for v in Verdict}
    for report in reports:
        counts[classify(report, floor)] += 1
    return counts[Verdict.DIVERGENT] == 0, counts


def summarize(reports: Sequence[LosslessReport]) -> tuple[int, int]:
    """(matched, total) — exact matches only."""
    return sum(1 for r in reports if r.matched), len(reports)
