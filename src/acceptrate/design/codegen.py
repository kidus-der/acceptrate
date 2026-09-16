"""Generated outputs, and the stale/fresh check that CI runs against them.

Both outputs are rendered from one tokens file and compared byte for byte
with what is on disk, so a hand edit or a forgotten `--write` fails fast.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from acceptrate.design.css import render_css
from acceptrate.design.go import render_go
from acceptrate.design.tokens import DEFAULT_TOKENS_PATH, Tokens, load_tokens


@dataclass(frozen=True)
class Output:
    path: Path
    """Relative to the repo root."""

    render: Callable[[Tokens], str]


OUTPUTS: tuple[Output, ...] = (
    Output(Path("dashboard") / "src" / "tokens.css", render_css),
    Output(Path("tui") / "theme" / "theme_gen.go", render_go),
)
OUTPUT_PATHS: tuple[Path, ...] = tuple(output.path for output in OUTPUTS)


def expected_outputs(root: Path) -> tuple[tuple[Path, str], ...]:
    """(relative path, rendered text) for every output, from the tokens under `root`."""
    tokens = load_tokens(root / DEFAULT_TOKENS_PATH)
    return tuple((output.path, output.render(tokens)) for output in OUTPUTS)


def stale_outputs(root: Path) -> tuple[Path, ...]:
    """Relative paths whose on-disk content is missing or differs from the render."""
    return tuple(rel for rel, text in expected_outputs(root) if _on_disk(root / rel) != text)


def write_outputs(root: Path) -> tuple[Path, ...]:
    """Write every output that differs; return the relative paths that changed."""
    changed: list[Path] = []
    for rel, text in expected_outputs(root):
        target = root / rel
        if _on_disk(target) == text:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        changed.append(rel)
    return tuple(changed)


def _on_disk(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None
