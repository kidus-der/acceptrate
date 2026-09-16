"""The acceptrate command line.

Only entry points live here — cli.py is the composition root that wires
bench/ to the runtime through plain callables. `bench` prints one progress
line and never starts a live view (CLAUDE.md trap 5).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from acceptrate.config import DEFAULT_PAIR, ModelPairConfig
from acceptrate.memory import assess_fit, available_bytes

app = typer.Typer(no_args_is_help=True, add_completion=False)
models_app = typer.Typer(no_args_is_help=True)
bench_app = typer.Typer(invoke_without_command=True)
app.add_typer(models_app, name="models", help="Pull and inspect model weights.")
app.add_typer(bench_app, name="bench", help="The research harness. One progress line, no live view")

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
    fit = assess_fit(pair, available_bytes())
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
        _fail("use --smoke, or a subcommand: compare")
    _check_fit(DEFAULT_PAIR)
    _smoke_one("draft", DEFAULT_PAIR.draft.repo)  # type: ignore[union-attr]
    _smoke_one("target", DEFAULT_PAIR.target.repo)
    typer.echo("smoke: ok")


def _progress_line(done: int, total: int) -> None:
    sys.stdout.write(f"\r  generation {done}/{total}")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


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


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
