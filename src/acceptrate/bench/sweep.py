"""The sweep grid plan: draft x K x workload tags, K = 0 interleaved as the baseline.

Pure planning — the composition root (cli.py) turns each ArmSpec into a
runner Arm with a generate() callable. bench/ never sees the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from acceptrate.trace.manifest import run_id_for


@dataclass(frozen=True)
class ArmSpec:
    name: str
    k: int
    run_id: str
    config: dict[str, Any]


def parse_ks(spec: str) -> tuple[int, ...]:
    """'0-3' -> (0,1,2,3); '0,2,4' -> (0,2,4); mixes allowed."""
    if not spec.strip():
        raise ValueError("empty K specification")
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part.lstrip("-"):
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < lo:
                raise ValueError(f"bad K range {part!r}")
            out.extend(range(lo, hi + 1))
        else:
            k = int(part)
            if k < 0:
                raise ValueError(f"K must be non-negative, got {k}")
            out.append(k)
    return tuple(out)


def sweep_arms(
    ks: Sequence[int],
    target: str,
    draft: str | None,
    max_tokens: int,
    prompt_ids: Sequence[str],
    reps: int,
    warmup: int,
) -> tuple[ArmSpec, ...]:
    """One arm per K. The K = 0 config omits the draft so every draft shares one baseline cell."""
    if draft is None and any(k > 0 for k in ks):
        raise ValueError("a draft model is required for K > 0")
    arms: list[ArmSpec] = []
    for k in ks:
        config = {
            "arm": "plain" if k == 0 else "spec",
            "k": k,
            "target": target,
            "draft": None if k == 0 else draft,
            "max_tokens": max_tokens,
            "prompts": list(prompt_ids),
            "reps": reps,
            "warmup": warmup,
        }
        name = "plain" if k == 0 else f"spec-k{k}"
        arms.append(ArmSpec(name=name, k=k, run_id=run_id_for(config), config=config))
    return tuple(arms)
