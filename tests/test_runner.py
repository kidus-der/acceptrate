"""bench/runner.py — interleaved A/B schedule, warmup discard, sink per generation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from acceptrate.bench.runner import (
    Arm,
    GenerationOutcome,
    Job,
    RunPlan,
    interleaved_schedule,
    run_plan,
)
from acceptrate.trace.schema import make_row


@dataclass(frozen=True)
class P:
    id: str
    tag: str
    text: str


PROMPTS = (P("code-001", "code", "a"), P("prose-001", "prose", "b"), P("json-001", "json", "c"))


def _gen(window_ms: float):
    def generate(prompt: P, rep: int, run_id: str) -> GenerationOutcome:
        rows = tuple(
            make_row(
                run_id,
                i,
                i + 1,
                0,
                0,
                0.0,
                window_ms,
                prompt.tag,
                0,
                0,
                0,
                prompt.id,
                rep,
                window_ms,
            )
            for i in range(4)
        )
        return GenerationOutcome(n_tokens=5, rows=rows)

    return generate


def test_schedule_covers_every_arm_prompt_rep_exactly_once() -> None:
    arms = (Arm("base", "run-base", _gen(50.0)), Arm("spec", "run-spec", _gen(30.0)))

    jobs = interleaved_schedule(arms, PROMPTS, reps=2, warmup=0)

    keys = [(j.arm.name, j.prompt.id, j.rep) for j in jobs]
    assert len(keys) == len(set(keys)) == 12
    assert all(not j.warmup for j in jobs)


def test_schedule_interleaves_arms_and_rotates_which_goes_first() -> None:
    arms = (Arm("base", "run-base", _gen(50.0)), Arm("spec", "run-spec", _gen(30.0)))

    jobs = interleaved_schedule(arms, PROMPTS, reps=1, warmup=0)

    names = [j.arm.name for j in jobs]
    assert all(a != b for a, b in pairwise(names))
    assert names[0] != names[2]  # the leading arm alternates prompt to prompt


def test_schedule_puts_warmup_jobs_first_and_flags_them() -> None:
    arms = (Arm("base", "run-base", _gen(50.0)),)

    jobs = interleaved_schedule(arms, PROMPTS, reps=1, warmup=2)

    assert [j.warmup for j in jobs] == [True, True, False, False, False]
    assert all(isinstance(j, Job) for j in jobs)


def test_run_plan_sinks_only_non_warmup_rows_and_reports_progress() -> None:
    arms = (Arm("base", "run-base", _gen(50.0)),)
    plan = RunPlan(prompts=PROMPTS, reps=1, warmup=1)
    sunk: list[tuple[str, int]] = []
    progress: list[tuple[int, int]] = []

    summary = run_plan(
        plan, arms, sink=lambda run_id, rows: sunk.append((run_id, len(rows))),
        on_progress=lambda done, total: progress.append((done, total)),
    )  # fmt: skip

    assert sunk == [("run-base", 4)] * 3
    assert progress[-1] == (4, 4)
    assert summary.arms["base"].generations == 3


def test_summary_reports_median_iqr_throughput_per_arm() -> None:
    arms = (Arm("base", "run-base", _gen(50.0)), Arm("spec", "run-spec", _gen(25.0)))
    plan = RunPlan(prompts=PROMPTS, reps=2, warmup=0)

    summary = run_plan(plan, arms, sink=lambda *_: None, on_progress=lambda *_: None)

    assert summary.arms["base"].tok_s.median == 20.0
    assert summary.arms["spec"].tok_s.median == 40.0
    assert summary.arms["base"].tok_s.n == 6


def test_summary_counts_dirty_windows() -> None:
    def dirty(prompt: P, rep: int, run_id: str) -> GenerationOutcome:
        rows = (
            make_row(run_id, 0, 1, 0, 0, 0.0, 50.0, prompt.tag, 0, 0, 0, prompt.id, rep, 50.0),
            make_row(run_id, 1, 2, 0, 0, 0.0, 50.0, prompt.tag, 0, 7, 0, prompt.id, rep, 50.0),
            make_row(run_id, 2, 3, 0, 0, 0.0, 50.0, prompt.tag, 2, 0, 0, prompt.id, rep, 50.0),
        )
        return GenerationOutcome(n_tokens=4, rows=rows)

    arms = (Arm("base", "run-base", dirty),)
    plan = RunPlan(prompts=PROMPTS[:1], reps=1, warmup=0)

    summary = run_plan(plan, arms, sink=lambda *_: None, on_progress=lambda *_: None)

    assert summary.arms["base"].windows == 3
    assert summary.arms["base"].dirty_windows == 2
