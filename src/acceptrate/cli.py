"""The acceptrate command line.

Only entry points live here. `bench` prints one progress line and never
starts a live view (CLAUDE.md trap 5).
"""

from __future__ import annotations

import sys
import time
from typing import Annotated

import typer

from acceptrate.config import DEFAULT_PAIR, ModelPairConfig
from acceptrate.memory import assess_fit, available_bytes

app = typer.Typer(no_args_is_help=True, add_completion=False)
models_app = typer.Typer(no_args_is_help=True)
app.add_typer(models_app, name="models", help="Pull and inspect model weights.")

SMOKE_PROMPT = "Write a short Python function that reverses a string, then explain it."
SMOKE_TOKENS = 128


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


@app.command()
def bench(smoke: SmokeFlag = False) -> None:
    """The research harness. Prints one progress line; never opens a live view."""
    if not smoke:
        _fail("only --smoke is implemented in P0")
    _check_fit(DEFAULT_PAIR)
    _smoke_one("draft", DEFAULT_PAIR.draft.repo)  # type: ignore[union-attr]
    _smoke_one("target", DEFAULT_PAIR.target.repo)
    typer.echo("smoke: ok")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
