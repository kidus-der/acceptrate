"""The per-machine profile: `acceptrate calibrate` writes it, `serve --adaptive` reads it.

What transfers between machines is the decision logic; what does not is the
cost ratio c and a sensible opening alpha (brief tab 4). A short calibration
measures both here, once per machine, and stores them as priors. The
runtime still measures alpha and c live and re-solves for K every window.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from acceptrate.backend.protocol import Backend
from acceptrate.runtime.engine import GenerationContext, GuardReader, generate_plain
from acceptrate.runtime.speculative import generate_speculative
from acceptrate.trace.schema import FIELD_NAMES

_K, _N, _DRAFT_MS, _VERIFY_MS, _WINDOW_MS = (
    FIELD_NAMES.index(n) for n in ("k_proposed", "n_accepted", "draft_ms", "verify_ms", "window_ms")
)
PROFILE_DIRNAME = ".acceptrate"
PROFILE_FILENAME = "profile.json"
DEFAULT_KS = (1, 2, 4, 8)
MS_PER_S = 1000.0


@dataclass(frozen=True)
class MachineProfile:
    chip: str
    mlx_version: str
    target: str
    draft: str
    created_at: str
    baseline_tok_s: float
    c_by_k: dict[int, float]
    alpha_prior: float


def default_profile_path() -> Path:
    return Path(os.environ.get("HOME", "~")).expanduser() / PROFILE_DIRNAME / PROFILE_FILENAME


def _tok_s(rows) -> float:
    tokens = sum(r[_N] + 1 for r in rows)
    ms = sum(r[_WINDOW_MS] for r in rows)
    return tokens / ms * MS_PER_S if ms > 0 else 0.0


def measure_profile(
    target: Backend,
    draft: Backend,
    prompts: Sequence[Sequence[int]],
    ks: Sequence[int],
    max_tokens: int,
    guard: GuardReader,
    chip: str,
    mlx_version: str,
    target_repo: str,
    draft_repo: str,
    eos: frozenset[int] = frozenset(),
) -> MachineProfile:
    """Short plain and speculative runs: baseline tok/s, per-K cost ratio, per-token alpha."""
    ctx = GenerationContext("calibrate", "chat", "calibrate", 0)
    baseline = [
        _tok_s(generate_plain(target, p, max_tokens, eos, ctx, guard).rows) for p in prompts
    ]
    c_by_k: dict[int, float] = {}
    accepted = examined = 0
    for k in ks:
        ratios: list[float] = []
        for p in prompts:
            result = generate_speculative(target, draft, p, max_tokens, eos, ctx, guard, k)
            for r in result.rows:
                if r[_VERIFY_MS] > 0:
                    ratios.append((r[_DRAFT_MS] / r[_K]) / r[_VERIFY_MS])
                accepted += r[_N]
                examined += r[_N] + 1 if r[_N] < r[_K] else r[_K]
        c_by_k[int(k)] = float(median(ratios)) if ratios else 0.0
    return MachineProfile(
        chip=chip,
        mlx_version=mlx_version,
        target=target_repo,
        draft=draft_repo,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        baseline_tok_s=float(median(baseline)) if baseline else 0.0,
        c_by_k=c_by_k,
        alpha_prior=accepted / examined if examined else 0.0,
    )


def scheduler_priors(profile: MachineProfile, k: int) -> tuple[float, float]:
    """(alpha_prior, c_prior): c at the requested K if profiled, else the median over K."""
    c = profile.c_by_k.get(k)
    if c is None:
        c = float(median(profile.c_by_k.values())) if profile.c_by_k else 0.16
    return profile.alpha_prior, c


def save_profile(profile: MachineProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(profile)
    data["c_by_k"] = {str(k): v for k, v in profile.c_by_k.items()}
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_profile(path: Path) -> MachineProfile:
    data = json.loads(path.read_text())
    missing = [f for f in MachineProfile.__dataclass_fields__ if f not in data]
    if missing:
        raise ValueError(f"{path} is not a machine profile: missing {missing}")
    data["c_by_k"] = {int(k): float(v) for k, v in data["c_by_k"].items()}
    return MachineProfile(**{k: data[k] for k in MachineProfile.__dataclass_fields__})
