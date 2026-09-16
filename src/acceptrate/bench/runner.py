"""Executes a run: interleaves arms against thermal drift, discards warmup.

bench/ never imports the runtime. An Arm carries a plain callable that the
composition root (cli.py) builds; the runner only sees prompts in, trace
rows out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from acceptrate.bench.stats import MedianIQR, median_iqr, throughput_tok_s
from acceptrate.trace.schema import FIELD_NAMES, WindowRow

_MEM = FIELD_NAMES.index("mem_pressure")
_PAGE_INS = FIELD_NAMES.index("page_ins")
_THERMAL = FIELD_NAMES.index("thermal_level")


class PromptLike(Protocol):
    id: str
    tag: str
    text: str


class GenerationOutcome(NamedTuple):
    n_tokens: int
    rows: tuple[WindowRow, ...]


Generator = Callable[[PromptLike, int, str], GenerationOutcome]
"""(prompt, rep, run_id) -> outcome. Built by cli.py; bench never sees the engine."""

Sink = Callable[[str, Sequence[WindowRow]], None]
Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class Arm:
    name: str
    run_id: str
    generate: Generator


@dataclass(frozen=True)
class Job:
    arm: Arm
    prompt: PromptLike
    rep: int
    warmup: bool


@dataclass(frozen=True)
class RunPlan:
    prompts: Sequence[PromptLike]
    reps: int = 1
    warmup: int = 2
    """Generations per arm run and discarded before anything is recorded."""


@dataclass(frozen=True)
class ArmSummary:
    generations: int
    windows: int
    dirty_windows: int
    tok_s: MedianIQR


@dataclass(frozen=True)
class RunSummary:
    arms: dict[str, ArmSummary]


def is_dirty(row: WindowRow) -> bool:
    return row[_PAGE_INS] > 0 or row[_MEM] > 0 or row[_THERMAL] > 0


def interleaved_schedule(
    arms: Sequence[Arm], prompts: Sequence[PromptLike], reps: int, warmup: int
) -> tuple[Job, ...]:
    """Warmup jobs first, then for each rep and prompt every arm — with the
    leading arm rotating per prompt so no arm always runs on a cooler chip."""
    jobs: list[Job] = []
    for i in range(warmup):
        prompt = prompts[i % len(prompts)]
        jobs.extend(Job(arm, prompt, -1, True) for arm in arms)
    for rep in range(reps):
        for p_idx, prompt in enumerate(prompts):
            order = list(arms[p_idx % len(arms) :]) + list(arms[: p_idx % len(arms)])
            jobs.extend(Job(arm, prompt, rep, False) for arm in order)
    return tuple(jobs)


def run_plan(plan: RunPlan, arms: Sequence[Arm], sink: Sink, on_progress: Progress) -> RunSummary:
    jobs = interleaved_schedule(arms, plan.prompts, plan.reps, plan.warmup)
    per_arm: dict[str, list[tuple[WindowRow, ...]]] = {arm.name: [] for arm in arms}
    for done, job in enumerate(jobs, start=1):
        outcome = job.arm.generate(job.prompt, job.rep, job.arm.run_id)
        if not job.warmup:
            sink(job.arm.run_id, outcome.rows)
            per_arm[job.arm.name].append(outcome.rows)
        on_progress(done, len(jobs))
    return RunSummary(arms={name: _summarise(gens) for name, gens in per_arm.items()})


def _summarise(generations: Sequence[tuple[WindowRow, ...]]) -> ArmSummary:
    rows = [row for gen in generations for row in gen]
    rates = [throughput_tok_s(gen) for gen in generations if gen]
    tok_s = median_iqr(rates) if rates else MedianIQR(0.0, 0.0, 0.0, 0)
    return ArmSummary(
        generations=len(generations),
        windows=len(rows),
        dirty_windows=sum(1 for row in rows if is_dirty(row)),
        tok_s=tok_s,
    )
