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


LOOKUP_DRAFT = "lookup"
"""Pass as --draft to use prompt-lookup drafting (no draft model, c ~ 0)."""


def _draft_spec(draft: str):
    from acceptrate.weights import cached_spec

    return None if draft == LOOKUP_DRAFT else cached_spec(draft)


def _load_draft(draft: str, target):
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.runtime.lookup import LookupBackend

    if draft == LOOKUP_DRAFT:
        return LookupBackend(vocab_size=target.vocab_size)
    return MLXBackend.load(draft)


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


DEFAULT_SWEEP_OUT = Path("traces/sweep")


@bench_app.command("sweep")
def bench_sweep(
    draft: Annotated[str, typer.Option(help="Draft model repo.")] = DEFAULT_PAIR.draft.repo,  # type: ignore[union-attr]
    ks: Annotated[str, typer.Option(help="Draft depths, e.g. '0-8' or '0,2,4'.")] = "0-8",
    prompts_per_tag: Annotated[int, typer.Option(help="Train-split prompts per tag.")] = 6,
    max_tokens: Annotated[int, typer.Option(help="Tokens per generation.")] = 200,
    reps: Annotated[int, typer.Option(help="Repeats per prompt.")] = 1,
    warmup: Annotated[int, typer.Option(help="Generations discarded per arm.")] = 2,
    out: Annotated[Path, typer.Option(help="Traces root for sweep cells.")] = DEFAULT_SWEEP_OUT,
) -> None:
    """P3: one draft x every K x six tags, arms interleaved per prompt, one parquet cell per arm."""
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.bench.guards import SystemGuard
    from acceptrate.bench.runner import Arm, GenerationOutcome, RunPlan, run_plan
    from acceptrate.bench.selection import select_prompts
    from acceptrate.bench.sweep import parse_ks, sweep_arms
    from acceptrate.bench.workloads import as_chat_messages, load_corpus
    from acceptrate.runtime.engine import GenerationContext, generate_plain
    from acceptrate.runtime.speculative import generate_speculative
    from acceptrate.trace import TraceWriter, build_manifest
    from acceptrate.trace.schema import WORKLOAD_TAGS

    depths = parse_ks(ks)
    needs_draft = any(k > 0 for k in depths)
    pair = ModelPairConfig(
        target=DEFAULT_PAIR.target, draft=_draft_spec(draft) if needs_draft else None
    )
    _check_fit(pair)
    chosen = select_prompts(load_corpus(), n=prompts_per_tag * len(WORKLOAD_TAGS))
    specs = sweep_arms(
        depths,
        pair.target.repo,
        draft if needs_draft else None,
        max_tokens,
        [p.id for p in chosen],
        reps,
        warmup,
    )
    target = MLXBackend.load(pair.target.repo)
    draft_backend = _load_draft(draft, target) if needs_draft else None
    tokenizer = target.tokenizer
    eos = tokenizer.eos_token_ids
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    writers = {
        spec.run_id: TraceWriter(out / f"{spec.run_id}-{stamp}", build_manifest(spec.config))
        for spec in specs
    }

    def make_generate(k: int):
        def generate(prompt, rep: int, rid: str) -> GenerationOutcome:
            tokens = tokenizer.encode_chat(as_chat_messages(prompt))
            ctx = GenerationContext(rid, prompt.tag, prompt.id, rep)
            if k == 0:
                result = generate_plain(target, tokens, max_tokens, eos, ctx, lambda: guard.latest)
            else:
                result = generate_speculative(
                    target, draft_backend, tokens, max_tokens, eos, ctx, lambda: guard.latest, k
                )
            return GenerationOutcome(len(result.tokens), result.rows)

        return generate

    arms = tuple(Arm(spec.name, spec.run_id, make_generate(spec.k)) for spec in specs)
    typer.echo(f"sweep draft={draft} ks={depths} prompts={len(chosen)} reps={reps} -> {out}")
    with SystemGuard() as guard:
        try:
            summary = run_plan(
                RunPlan(prompts=chosen, reps=reps, warmup=warmup),
                arms,
                sink=lambda rid, rows: writers[rid].end_generation(rows),
                on_progress=_progress_line,
            )
        finally:
            for writer in writers.values():
                writer.close()
    for spec in specs:
        arm = summary.arms[spec.name]
        typer.echo(
            f"{spec.name:8s} run {spec.run_id}: median {arm.tok_s.median:.2f} tok/s "
            f"IQR [{arm.tok_s.q1:.2f}, {arm.tok_s.q3:.2f}] n={arm.tok_s.n} "
            f"windows={arm.windows} dirty={arm.dirty_windows}"
        )
    typer.echo(f"guard_errors={guard.sample_errors}")


DEFAULT_P5_OUT = Path("traces/p5")


@bench_app.command("adaptive")
def bench_adaptive(
    draft: Annotated[str, typer.Option(help="Draft model repo.")] = DEFAULT_PAIR.draft.repo,  # type: ignore[union-attr]
    ks: Annotated[str, typer.Option(help="Fixed depths to compare against.")] = "1-8",
    prompts: Annotated[int, typer.Option(help="Held-out mixed prompts.")] = 36,
    seed: Annotated[int, typer.Option(help="Seed for the mixed workload.")] = 0,
    max_tokens: Annotated[int, typer.Option(help="Tokens per generation.")] = 200,
    reps: Annotated[int, typer.Option(help="Repeats per prompt.")] = 1,
    warmup: Annotated[int, typer.Option(help="Generations discarded per arm.")] = 2,
    k_max: Annotated[int, typer.Option(help="Adaptive scheduler's ceiling.")] = 8,
    out: Annotated[Path, typer.Option(help="Traces root for P5 cells.")] = DEFAULT_P5_OUT,
) -> None:
    """P5 gate: adaptive K beats the best single fixed K by >= 5% on the held-out mixed workload."""
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.bench.guards import SystemGuard
    from acceptrate.bench.p5 import p5_arms, score_p5
    from acceptrate.bench.runner import Arm, GenerationOutcome, RunPlan, run_plan
    from acceptrate.bench.sweep import parse_ks
    from acceptrate.bench.workloads import as_chat_messages, mixed_workload
    from acceptrate.runtime.adaptive import AdaptiveScheduler, SchedulerConfig, generate_adaptive
    from acceptrate.runtime.engine import GenerationContext
    from acceptrate.runtime.speculative import generate_speculative
    from acceptrate.trace import TraceWriter, build_manifest

    fixed = tuple(k for k in parse_ks(ks) if k >= 1)
    pair = ModelPairConfig(target=DEFAULT_PAIR.target, draft=_draft_spec(draft))
    _check_fit(pair)
    chosen = mixed_workload(seed=seed, n=prompts, split="heldout")
    specs = p5_arms(
        fixed, pair.target.repo, draft, max_tokens, [p.id for p in chosen], reps, warmup
    )
    target = MLXBackend.load(pair.target.repo)
    draft_backend = _load_draft(draft, target)
    tokenizer = target.tokenizer
    eos = tokenizer.eos_token_ids
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    writers = {
        spec.run_id: TraceWriter(out / f"{spec.run_id}-{stamp}", build_manifest(spec.config))
        for spec in specs
    }
    sched_cfg = SchedulerConfig(
        k_min=1, k_max=k_max, prior_alpha=0.6, prior_c=0.16, half_life_windows=4, warmup_windows=0
    )

    def make_generate(k: int | None):
        def generate(prompt, rep: int, rid: str) -> GenerationOutcome:
            tokens = tokenizer.encode_chat(as_chat_messages(prompt))
            ctx = GenerationContext(rid, prompt.tag, prompt.id, rep)
            if k is None:
                result = generate_adaptive(
                    target, draft_backend, tokens, max_tokens, eos, ctx, lambda: guard.latest,
                    AdaptiveScheduler(sched_cfg),
                )  # fmt: skip
            else:
                result = generate_speculative(
                    target, draft_backend, tokens, max_tokens, eos, ctx, lambda: guard.latest, k
                )
            return GenerationOutcome(len(result.tokens), result.rows)

        return generate

    arms = tuple(Arm(spec.name, spec.run_id, make_generate(spec.k)) for spec in specs)
    typer.echo(f"p5 draft={draft} fixed={fixed} heldout prompts={len(chosen)} seed={seed} -> {out}")
    with SystemGuard() as guard:
        try:
            summary = run_plan(
                RunPlan(prompts=chosen, reps=reps, warmup=warmup),
                arms,
                sink=lambda rid, rows: writers[rid].end_generation(rows),
                on_progress=_progress_line,
            )
        finally:
            for writer in writers.values():
                writer.close()
    for spec in specs:
        arm = summary.arms[spec.name]
        typer.echo(
            f"{spec.name:8s} median {arm.tok_s.median:.2f} tok/s IQR [{arm.tok_s.q1:.2f}, "
            f"{arm.tok_s.q3:.2f}] n={arm.tok_s.n} windows={arm.windows} dirty={arm.dirty_windows}"
        )
    verdict = score_p5(summary.arms)
    typer.echo(
        f"adaptive {verdict.adaptive_tok_s:.2f} vs best fixed {verdict.best_fixed_name} "
        f"{verdict.best_fixed_tok_s:.2f}: gain {verdict.gain * 100:+.2f}% (need >= +5%)"
    )
    if not verdict.passed:
        _fail("P5 gate: FAIL")
    typer.echo("P5 gate: PASS")


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


@verify_app.command("distributional")
def verify_distributional(
    prompt_id: Annotated[str, typer.Option(help="Corpus prompt to sample from.")] = "prose-001",
    k: Annotated[int, typer.Option(help="Draft depth.")] = 4,
    temperature: Annotated[float, typer.Option(help="Sampling temperature > 0.")] = 1.0,
    n_samples: Annotated[int, typer.Option(help="Samples per path.")] = 400,
    seed: Annotated[int, typer.Option(help="RNG seed.")] = 0,
) -> None:
    """Above temperature 0: the speculative next-token law must match plain sampling (TV bound)."""
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.bench.workloads import as_chat_messages, load_corpus
    from acceptrate.verify.distributional import passes, run_distributional_check

    if temperature <= 0:
        _fail("temperature must be > 0; use 'verify lossless' for greedy")
    _check_fit(DEFAULT_PAIR)
    prompt = next((p for p in load_corpus() if p.id == prompt_id), None)
    if prompt is None:
        _fail(f"unknown prompt id {prompt_id}")
    target = MLXBackend.load(DEFAULT_PAIR.target.repo)
    draft = MLXBackend.load(DEFAULT_PAIR.draft.repo)  # type: ignore[union-attr]
    tokens = target.tokenizer.encode_chat(as_chat_messages(prompt))
    report, bound = run_distributional_check(
        target, draft, tokens, k, temperature, n_samples, seed, target.vocab_size
    )
    typer.echo(
        f"{prompt_id} K={k} T={temperature}: TV {report.tv:.4f} vs bound {bound:.4f} "
        f"(n={report.n_plain}/{report.n_spec})"
    )
    if not passes(report, bound):
        _fail("distributional check: FAIL")
    typer.echo("distributional check: PASS")


predictor_app = typer.Typer(no_args_is_help=True)
app.add_typer(predictor_app, name="predictor", help="The cold-start alpha predictor (P6).")
DEFAULT_PREDICTOR_PATH = Path("models/predictor.joblib")


@predictor_app.command("train")
def predictor_train(
    traces: Annotated[Path, typer.Option(help="Traces root with train-split cells.")] = Path(
        "traces/sweep"
    ),
    out: Annotated[Path, typer.Option(help="Where to save the model.")] = DEFAULT_PREDICTOR_PATH,
) -> None:
    """Train the cold-start predictor on per-prompt alpha from train-split traces."""
    from acceptrate.bench.workloads import load_corpus
    from acceptrate.model.predictor import train_predictor
    from acceptrate.model.predictor_data import build_dataset
    from acceptrate.trace import read_runs

    ds = build_dataset(read_runs(traces), load_corpus(), split="train")
    if not ds.prompt_ids:
        _fail(f"no train-split prompts with windows under {traces}")
    predictor = train_predictor(ds.features, ds.alphas)
    predictor.save(out)
    typer.echo(
        f"trained on {len(ds.prompt_ids)} prompts (alpha mean {ds.alphas.mean():.3f}) -> {out}"
    )


@predictor_app.command("eval")
def predictor_eval(
    train_traces: Annotated[Path, typer.Option(help="Train-split cells.")] = Path("traces/sweep"),
    heldout_traces: Annotated[Path, typer.Option(help="Held-out cells.")] = Path("traces/p5"),
    model: Annotated[Path, typer.Option(help="Saved predictor.")] = DEFAULT_PREDICTOR_PATH,
) -> None:
    """P6 gate: the predictor must beat a constant prior (train mean) on held-out prompts."""
    from acceptrate.bench.workloads import load_corpus
    from acceptrate.model.predictor import Predictor, constant_prior_mae, evaluate_mae
    from acceptrate.model.predictor_data import build_dataset
    from acceptrate.trace import read_runs

    corpus = load_corpus()
    train = build_dataset(read_runs(train_traces), corpus, split="train")
    held = build_dataset(read_runs(heldout_traces), corpus, split="heldout")
    if not held.prompt_ids:
        _fail(f"no held-out prompts with windows under {heldout_traces}")
    predictor = Predictor.load(model)
    mae = evaluate_mae(predictor, held.features, held.alphas)
    prior = constant_prior_mae(train.alphas, held.alphas)
    typer.echo(
        f"held-out prompts={len(held.prompt_ids)}  predictor MAE {mae:.4f}  "
        f"constant-prior MAE {prior:.4f}  improvement {(1 - mae / prior) * 100:+.1f}%"
    )
    if mae >= prior:
        _fail("P6 predictor gate: FAIL")
    typer.echo("P6 predictor gate: PASS")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
