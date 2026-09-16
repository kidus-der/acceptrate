"""P5 evaluation plan and verdict: adaptive K against every fixed K on the held-out mixed workload.

Gate (brief tab 6): adaptive beats the best single fixed K by >= 5% median
throughput. The held-out split was fixed before any measurement existed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from acceptrate.bench.runner import ArmSummary
from acceptrate.trace.manifest import run_id_for

GAIN_REQUIRED = 0.05
ADAPTIVE_ARM = "adaptive"


@dataclass(frozen=True)
class P5Arm:
    name: str
    k: int | None
    """None for the adaptive arm."""
    run_id: str
    config: dict[str, Any]


@dataclass(frozen=True)
class P5Verdict:
    adaptive_tok_s: float
    best_fixed_name: str
    best_fixed_tok_s: float
    gain: float

    @property
    def passed(self) -> bool:
        return self.gain >= GAIN_REQUIRED


def p5_arms(
    fixed_ks: Sequence[int],
    target: str,
    draft: str,
    max_tokens: int,
    prompt_ids: Sequence[str],
    reps: int,
    warmup: int,
) -> tuple[P5Arm, ...]:
    base = {
        "target": target,
        "draft": draft,
        "max_tokens": max_tokens,
        "prompts": list(prompt_ids),
        "reps": reps,
        "warmup": warmup,
        "split": "heldout",
    }
    adaptive_cfg = {**base, "arm": ADAPTIVE_ARM, "k": None}
    arms = [P5Arm(ADAPTIVE_ARM, None, run_id_for(adaptive_cfg), adaptive_cfg)]
    for k in fixed_ks:
        cfg = {**base, "arm": "spec", "k": k}
        arms.append(P5Arm(f"spec-k{k}", k, run_id_for(cfg), cfg))
    return tuple(arms)


def score_p5(summaries: Mapping[str, ArmSummary]) -> P5Verdict:
    if ADAPTIVE_ARM not in summaries:
        raise ValueError("no adaptive arm in the summaries")
    fixed = {name: s for name, s in summaries.items() if name != ADAPTIVE_ARM}
    if not fixed:
        raise ValueError("no fixed-K arms to compare against")
    best_name = max(fixed, key=lambda name: fixed[name].tok_s.median)
    adaptive = summaries[ADAPTIVE_ARM].tok_s.median
    best = fixed[best_name].tok_s.median
    gain = adaptive / best - 1.0 if best > 0 else float("inf")
    return P5Verdict(adaptive, best_name, best, gain)
