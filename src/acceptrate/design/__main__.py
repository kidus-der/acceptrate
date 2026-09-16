"""`python -m acceptrate.design --check | --write [--root PATH]`.

`--check` exits 1 and lists every stale output; CI should run it so the
generated CSS and Go can never drift from design/tokens.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from acceptrate.design.codegen import stale_outputs, write_outputs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    check: Annotated[bool, typer.Option("--check", help="Exit 1 if any output is stale.")] = False,
    write: Annotated[bool, typer.Option("--write", help="Regenerate every stale output.")] = False,
    root: Annotated[Path, typer.Option(help="Repo root holding design/tokens.json.")] = Path(),
) -> None:
    """Codegen design/tokens.json to CSS custom properties and Lip Gloss constants."""
    if check == write:
        typer.secho("pass exactly one of --check or --write", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if write:
        for rel in write_outputs(root):
            typer.echo(f"wrote {rel}")
        return
    stale = stale_outputs(root)
    for rel in stale:
        typer.echo(f"stale: {rel}")
    if stale:
        typer.secho("run: uv run python -m acceptrate.design --write", err=True)
        raise typer.Exit(code=1)
    typer.echo("design tokens: outputs are fresh")


if __name__ == "__main__":
    app()
