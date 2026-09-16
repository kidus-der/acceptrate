"""The acceptrate command line.

Only entry points live here — cli.py is the composition root that wires
bench/ to the runtime through plain callables. `bench` prints one progress
line and never starts a live view (CLAUDE.md trap 5).
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from acceptrate.config import DEFAULT_PAIR, ModelPairConfig
from acceptrate.memory import assess_fit, current_pressure_level, total_bytes
from acceptrate.verify.noise_floor import DEFAULT_CALIBRATION_PATH

app = typer.Typer(no_args_is_help=True, add_completion=False)
models_app = typer.Typer(no_args_is_help=True)
bench_app = typer.Typer(invoke_without_command=True)
app.add_typer(models_app, name="models", help="Pull and inspect model weights.")
app.add_typer(bench_app, name="bench", help="The research harness. One progress line, no live view")
verify_app = typer.Typer(no_args_is_help=True)
app.add_typer(verify_app, name="verify", help="Losslessness gates.")

SMOKE_PROMPT = "Write a short Python function that reverses a string, then explain it."
SMOKE_TOKENS = 128
DEFAULT_TRACES = Path("traces")


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@models_app.command("pull")
def models_pull() -> None:
    """Download the default target and draft weights into the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    for spec in (DEFAULT_PAIR.draft, DEFAULT_PAIR.target):
        if spec is None:
            continue
        typer.echo(f"pulling {spec.repo} ...")
        path = snapshot_download(spec.repo)
        typer.echo(f"  -> {path}")


def _check_fit(pair: ModelPairConfig) -> None:
    fit = assess_fit(pair, total_bytes(), current_pressure_level())
    if not fit.ok:
        _fail(f"refusing to load: {fit.reason}")


def _smoke_one(label: str, repo: str) -> float:
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.bench.smoke import stream_greedy

    backend = MLXBackend.load(repo)
    prompt = backend.tokenizer.encode(SMOKE_PROMPT)
    started = time.perf_counter()
    result = stream_greedy(backend, prompt, SMOKE_TOKENS, backend.tokenizer.eos_token_ids)
    elapsed = time.perf_counter() - started
    if not result.stopped_on_eos and len(result.tokens) != SMOKE_TOKENS:
        _fail(f"{label}: streamed {len(result.tokens)} tokens, expected {SMOKE_TOKENS}")
    typer.echo(
        f"{label:6s} {repo}: {len(result.tokens)} tokens in {elapsed:.1f}s "
        f"({len(result.tokens) / elapsed:.1f} tok/s, plain greedy)"
    )
    return elapsed


SmokeFlag = Annotated[bool, typer.Option("--smoke", help="Stream 128 tokens from both models.")]


@bench_app.callback()
def bench(ctx: typer.Context, smoke: SmokeFlag = False) -> None:
    """The research harness. Prints one progress line; never opens a live view."""
    if ctx.invoked_subcommand is not None:
        return
    if not smoke:
        _fail("use --smoke, or a subcommand: baseline, compare")
    _check_fit(DEFAULT_PAIR)
    _smoke_one("draft", DEFAULT_PAIR.draft.repo)  # type: ignore[union-attr]
    _smoke_one("target", DEFAULT_PAIR.target.repo)
    typer.echo("smoke: ok")


def _progress_line(done: int, total: int) -> None:
    sys.stdout.write(f"\r  generation {done}/{total}")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


@bench_app.command("baseline")
def bench_baseline(
    prompts: Annotated[int, typer.Option(help="Corpus prompts, round-robin over tags.")] = 12,
    reps: Annotated[int, typer.Option(help="Repeats per prompt.")] = 1,
    warmup: Annotated[int, typer.Option(help="Generations discarded before recording.")] = 2,
    max_tokens: Annotated[int, typer.Option(help="Tokens per generation.")] = 128,
    tag: Annotated[str | None, typer.Option(help="Restrict to one workload tag.")] = None,
    out: Annotated[Path, typer.Option(help="Traces root directory.")] = DEFAULT_TRACES,
) -> None:
    """Plain K=0 decoding with the target model: the trustworthy baseline numbers."""
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.bench.guards import SystemGuard
    from acceptrate.bench.runner import Arm, GenerationOutcome, RunPlan, run_plan
    from acceptrate.bench.selection import select_prompts
    from acceptrate.bench.workloads import as_chat_messages, load_corpus
    from acceptrate.runtime.engine import GenerationContext, generate_plain
    from acceptrate.trace import TraceWriter, build_manifest, run_id_for

    _check_fit(ModelPairConfig(target=DEFAULT_PAIR.target, draft=None))
    chosen = select_prompts(load_corpus(), n=prompts, tag=tag)
    config = {
        "arm": "plain",
        "k": 0,
        "target": DEFAULT_PAIR.target.repo,
        "max_tokens": max_tokens,
        "prompts": [p.id for p in chosen],
        "reps": reps,
        "warmup": warmup,
    }
    run_id = run_id_for(config)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = out / f"{run_id}-{stamp}"
    target = MLXBackend.load(DEFAULT_PAIR.target.repo)
    tokenizer = target.tokenizer
    eos = tokenizer.eos_token_ids

    with SystemGuard() as guard, TraceWriter(run_dir, build_manifest(config)) as writer:

        def generate(prompt, rep: int, rid: str) -> GenerationOutcome:
            tokens = tokenizer.encode_chat(as_chat_messages(prompt))
            ctx = GenerationContext(rid, prompt.tag, prompt.id, rep)
            result = generate_plain(target, tokens, max_tokens, eos, ctx, lambda: guard.latest)
            return GenerationOutcome(len(result.tokens), result.rows)

        typer.echo(f"run {run_id} -> {run_dir}")
        typer.echo(f"  {len(chosen)} prompts x {reps} reps, warmup {warmup}, {max_tokens} tokens")
        summary = run_plan(
            RunPlan(prompts=chosen, reps=reps, warmup=warmup),
            (Arm("plain", run_id, generate),),
            sink=lambda _rid, rows: writer.end_generation(rows),
            on_progress=_progress_line,
        )

    arm = summary.arms["plain"]
    stats = arm.tok_s
    typer.echo(
        f"plain: median {stats.median:.2f} tok/s  IQR [{stats.q1:.2f}, {stats.q3:.2f}]  n={stats.n}"
    )
    typer.echo(
        f"  windows={arm.windows}  dirty={arm.dirty_windows}  guard_errors={guard.sample_errors}"
    )


@bench_app.command("compare")
def bench_compare(run_a: Path, run_b: Path) -> None:
    """P1 gate: two runs of the same config must agree on median tok/s within 2%."""
    from acceptrate.bench.compare import GATE_TOLERANCE, compare_runs
    from acceptrate.trace import read_manifest, read_run

    if read_manifest(run_a).run_id != read_manifest(run_b).run_id:
        _fail("runs have different configs (run_id mismatch); the gate compares like with like")
    result = compare_runs(read_run(run_a), read_run(run_b))
    for label, stats, dirty in (("A", result.a, result.dirty_a), ("B", result.b, result.dirty_b)):
        typer.echo(f"{label}: median {stats.median:.2f} tok/s (n={stats.n}, dirty rows {dirty})")
    gate_pct = GATE_TOLERANCE * 100
    typer.echo(f"relative difference {result.rel_diff * 100:.2f}% (gate {gate_pct:.0f}%)")
    if not result.passed:
        _fail("P1 gate: FAIL")
    typer.echo("P1 gate: PASS")


@verify_app.command("calibrate")
def verify_calibrate(
    prompts: Annotated[int, typer.Option(help="Corpus prompts to score.")] = 10,
    gen_tokens: Annotated[int, typer.Option(help="Tokens generated per prompt.")] = 40,
    window: Annotated[int, typer.Option(help="Batched window size (K+1).")] = 5,
    out: Annotated[Path, typer.Option(help="Calibration file.")] = DEFAULT_CALIBRATION_PATH,
) -> None:
    """Measure the batched-vs-sequential logit noise floor of the target on this machine."""
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.bench.selection import select_prompts
    from acceptrate.bench.workloads import as_chat_messages, load_corpus
    from acceptrate.trace.manifest import chip_name, package_version
    from acceptrate.verify.noise_floor import measure_noise_floor, save_noise_floor

    _check_fit(ModelPairConfig(target=DEFAULT_PAIR.target, draft=None))
    target = MLXBackend.load(DEFAULT_PAIR.target.repo)
    chosen = select_prompts(load_corpus(), n=prompts)
    token_lists = [target.tokenizer.encode_chat(as_chat_messages(p)) for p in chosen]
    floor = measure_noise_floor(
        target,
        token_lists,
        gen_tokens,
        window,
        chip=chip_name(),
        mlx_version=package_version("mlx"),
    )
    save_noise_floor(floor, out)
    typer.echo(
        f"noise floor over {floor.positions} positions: p50 {floor.p50:.4f}  p99 {floor.p99:.4f}  "
        f"max {floor.max:.4f}  argmax flips {floor.argmax_flips}  -> {out}"
    )


@verify_app.command("lossless")
def verify_lossless(
    prompts: Annotated[int, typer.Option(help="Corpus prompts, round-robin over tags.")] = 20,
    k: Annotated[int, typer.Option(help="Draft depth.")] = 4,
    max_tokens: Annotated[int, typer.Option(help="Tokens per generation.")] = 128,
    calibration: Annotated[Path, typer.Option(help="Noise-floor file.")] = DEFAULT_CALIBRATION_PATH,
) -> None:
    """P2 gate: at temperature 0, speculative output must be token-identical to plain decoding.

    A divergence passes only as a near-tie: its sequential top-2 logit margin
    must be within the calibrated cross-kernel noise floor. Every near-tie is printed.
    """
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.bench.selection import select_prompts
    from acceptrate.bench.workloads import as_chat_messages, load_corpus
    from acceptrate.runtime.engine import GenerationContext, generate_plain
    from acceptrate.runtime.speculative import generate_speculative
    from acceptrate.verify.lossless import (
        Verdict,
        classify,
        compare_generation,
        gate_verdicts,
        sequential_margin_at,
    )
    from acceptrate.verify.noise_floor import load_noise_floor

    if not calibration.exists():
        _fail(f"no calibration at {calibration}; run 'acceptrate verify calibrate' first")
    floor = load_noise_floor(calibration)
    _check_fit(DEFAULT_PAIR)
    chosen = select_prompts(load_corpus(), n=prompts)
    target = MLXBackend.load(DEFAULT_PAIR.target.repo)
    draft = MLXBackend.load(DEFAULT_PAIR.draft.repo)  # type: ignore[union-attr]
    eos = target.tokenizer.eos_token_ids
    typer.echo(
        f"noise floor {floor.max:.4f} ({floor.chip}, mlx {floor.mlx_version}, n={floor.positions})"
    )
    reports = []
    labels = {Verdict.IDENTICAL: "ok  ", Verdict.NEAR_TIE: "TIE ", Verdict.DIVERGENT: "DIFF"}
    for prompt in chosen:
        tokens = target.tokenizer.encode_chat(as_chat_messages(prompt))
        ctx = GenerationContext("verify", prompt.tag, prompt.id, 0)
        plain = generate_plain(target, tokens, max_tokens, eos, ctx, _NULL_GUARD)
        spec = generate_speculative(target, draft, tokens, max_tokens, eos, ctx, _NULL_GUARD, k)
        report = compare_generation(prompt.id, k, plain.tokens, spec.tokens, spec.rows)
        if not report.matched:
            margin = sequential_margin_at(target, tokens, plain.tokens, report.first_divergence)
            report = compare_generation(
                prompt.id, k, plain.tokens, spec.tokens, spec.rows, margin=margin
            )
        reports.append(report)
        detail = ""
        if not report.matched:
            detail = (
                f"  at {report.first_divergence}: plain {report.plain_token} vs "
                f"spec {report.spec_token}, margin {report.margin:.4f}"
            )
        typer.echo(
            f"{labels[classify(report, floor)]} {prompt.id:14s} tokens={report.n_tokens:3d} "
            f"windows={report.windows:3d} alpha={report.alpha:.2f}{detail}"
        )
    passed, counts = gate_verdicts(reports, floor)
    typer.echo(
        f"greedy equivalence K={k}: {counts[Verdict.IDENTICAL]} identical, "
        f"{counts[Verdict.NEAR_TIE]} near-tie (within noise floor), "
        f"{counts[Verdict.DIVERGENT]} divergent of {len(reports)}"
    )
    if not passed:
        _fail("P2 gate: FAIL")
    typer.echo("P2 gate: PASS")


class _NullSnapshot:
    mem_pressure = 0
    page_ins = 0
    thermal_level = 0


def _NULL_GUARD() -> _NullSnapshot:  # noqa: N802 — a guard reader, used like a constant
    return _NullSnapshot()


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
