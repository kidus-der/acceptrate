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


@verify_app.command("lossless")
def verify_lossless(
    prompts: Annotated[int, typer.Option(help="Corpus prompts, round-robin over tags.")] = 20,
    k: Annotated[int, typer.Option(help="Draft depth.")] = 4,
    max_tokens: Annotated[int, typer.Option(help="Tokens per generation.")] = 128,
) -> None:
    """P2 gate: at temperature 0, speculative output must be token-identical to plain decoding."""
    from acceptrate.backend.mlx_backend import MLXBackend
    from acceptrate.bench.selection import select_prompts
    from acceptrate.bench.workloads import as_chat_messages, load_corpus
    from acceptrate.runtime.engine import GenerationContext, generate_plain
    from acceptrate.runtime.speculative import generate_speculative
    from acceptrate.verify.lossless import compare_generation, summarize

    _check_fit(DEFAULT_PAIR)
    chosen = select_prompts(load_corpus(), n=prompts)
    target = MLXBackend.load(DEFAULT_PAIR.target.repo)
    draft = MLXBackend.load(DEFAULT_PAIR.draft.repo)  # type: ignore[union-attr]
    eos = target.tokenizer.eos_token_ids
    reports = []
    for prompt in chosen:
        tokens = target.tokenizer.encode_chat(as_chat_messages(prompt))
        ctx = GenerationContext("verify", prompt.tag, prompt.id, 0)
        plain = generate_plain(target, tokens, max_tokens, eos, ctx, _NULL_GUARD)
        spec = generate_speculative(target, draft, tokens, max_tokens, eos, ctx, _NULL_GUARD, k)
        report = compare_generation(prompt.id, k, plain.tokens, spec.tokens, spec.rows)
        reports.append(report)
        verdict = "ok  " if report.matched else "DIFF"
        detail = ""
        if not report.matched:
            margin = _top2_margin_at(target, tokens, plain.tokens, report.first_divergence)
            detail = (
                f"  at {report.first_divergence}: plain {report.plain_token} vs "
                f"spec {report.spec_token}, sequential top-2 margin {margin:.4f}"
            )
        typer.echo(
            f"{verdict} {prompt.id:14s} tokens={report.n_tokens:3d} windows={report.windows:3d} "
            f"alpha={report.alpha:.2f}{detail}"
        )
    matched, total = summarize(reports)
    typer.echo(f"greedy equivalence K={k}: {matched}/{total} token-identical")
    if matched != total:
        _fail("P2 gate: FAIL")
    typer.echo("P2 gate: PASS")


def _top2_margin_at(target, prompt_tokens, plain_tokens, index: int) -> float:
    """Replay plain decoding to `index` and return logit(top1) - logit(top2) there.

    A divergence at a margin below the measured cross-kernel noise (~0.1) is
    a near-tie the batched and sequential Metal kernels resolve differently,
    not a bug in the speculative logic.
    """
    import numpy as np

    logits = target.prefill(prompt_tokens)
    for token in plain_tokens[:index]:
        logits = target.decode_step(token)
    top2 = np.partition(logits, -2)[-2:]
    return float(top2[1] - top2[0])


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
