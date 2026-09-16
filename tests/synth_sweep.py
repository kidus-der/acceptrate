"""A fake sweep on disk, written with the real TraceWriter, for analysis tests.

Every window is drawn from the analytical model itself: each drafted token is
accepted with the tag's per-token alpha until the first rejection, draft cost
is k * d, verify cost is v. Because timing is exactly the model, measured
speedup must reproduce the closed form and the fit's R² must come out ~1.
No model is loaded here, ever.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acceptrate.trace import TraceWriter, build_manifest, run_id_for
from acceptrate.trace.schema import WindowRow, make_row

TAG_ALPHA: dict[str, float] = {
    "code": 0.9,
    "json": 0.85,
    "reason": 0.7,
    "summarize": 0.6,
    "prose": 0.5,
    "chat": 0.4,
}
"""Per-token acceptance rates chosen so that some (tag, K) cells lose (speedup < 1)."""


@dataclass(frozen=True)
class SynthSpec:
    tag_alpha: dict[str, float]
    ks: tuple[int, ...] = (0, 2, 4, 8)
    draft: str = "fake/draft-1b"
    target: str = "fake/target-8b"
    max_tokens: int = 128
    prompts_per_tag: int = 3
    reps: int = 2
    draft_ms_per_token: float = 10.0
    verify_ms: float = 50.0
    overhead_ms: float = 0.0
    dirty_every: int = 0
    """Every n-th window of a run gets page_ins > 0; 0 means every window is clean."""
    seed: int = 0


DEFAULT_SPEC = SynthSpec(tag_alpha=TAG_ALPHA)

DIRTY_PAGE_INS = 3


def cell_config(spec: SynthSpec, k: int) -> dict:
    prompts = [f"{tag}-{i:03d}" for tag in spec.tag_alpha for i in range(spec.prompts_per_tag)]
    return {
        "arm": "plain" if k == 0 else "spec",
        "k": k,
        "target": spec.target,
        "draft": None if k == 0 else spec.draft,
        "max_tokens": spec.max_tokens,
        "prompts": prompts,
        "reps": spec.reps,
        "warmup": 2,
    }


def _n_accepted(rng: np.random.Generator, alpha: float, k: int) -> int:
    """Consecutive acceptances before the first rejection, capped at k."""
    accepted = 0
    while accepted < k and rng.random() < alpha:
        accepted += 1
    return accepted


def _generation_rows(
    rng: np.random.Generator,
    spec: SynthSpec,
    run_id: str,
    k: int,
    tag: str,
    prompt_id: str,
    rep: int,
    dirty: bool,
) -> list[WindowRow]:
    rows: list[WindowRow] = []
    token_pos = 0
    draft_ms = k * spec.draft_ms_per_token
    while token_pos < spec.max_tokens:
        accepted = _n_accepted(rng, spec.tag_alpha[tag], k)
        page_ins = DIRTY_PAGE_INS if dirty and len(rows) % spec.dirty_every == 0 else 0
        window_ms = draft_ms + spec.verify_ms + spec.overhead_ms
        row = make_row(
            run_id, len(rows), token_pos, k, accepted, draft_ms, spec.verify_ms,
            tag, 0, page_ins, 0, prompt_id, rep, window_ms,
        )  # fmt: skip
        rows.append(row)
        token_pos += accepted + 1
    return rows


def write_cell(root: Path, spec: SynthSpec, k: int, stamp: str = "20260916-120000") -> Path:
    """One run directory for one (draft, k) arm over every tag; returns its path."""
    config = cell_config(spec, k)
    run_id = run_id_for(config)
    manifest = build_manifest(
        config,
        now=lambda: "2026-09-16T12:00:00+00:00",
        chip=lambda: "Apple M4",
        memory_gb=lambda: 16.0,
        git_sha=lambda: "synthetic",
        package_version=lambda name: "0",
    )
    run_dir = root / f"{run_id}-{stamp}"
    rng = np.random.default_rng(spec.seed + k)
    dirty = spec.dirty_every > 0
    with TraceWriter(run_dir, manifest) as writer:
        for rep in range(spec.reps):
            for prompt_id in config["prompts"]:
                tag = prompt_id.rsplit("-", 1)[0]
                rows = _generation_rows(rng, spec, run_id, k, tag, prompt_id, rep, dirty)
                writer.end_generation(rows)
    return run_dir


def write_sweep(root: Path, spec: SynthSpec = DEFAULT_SPEC) -> dict[int, Path]:
    """Every k of the spec as its own run under `root`; returns k -> run directory."""
    root.mkdir(parents=True, exist_ok=True)
    return {k: write_cell(root, spec, k) for k in spec.ks}
