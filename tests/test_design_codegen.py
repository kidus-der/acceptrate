"""`python -m acceptrate.design --check` fails on stale outputs; `--write` refreshes them."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from acceptrate.design.__main__ import app
from acceptrate.design.codegen import OUTPUT_PATHS, stale_outputs, write_outputs

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENS_REL = Path("design") / "tokens.json"
CSS_REL = Path("dashboard") / "src" / "tokens.css"
GO_REL = Path("tui") / "theme" / "theme_gen.go"

runner = CliRunner()


def _copy_tree(tmp_path: Path) -> Path:
    """A minimal repo copy: the tokens file plus the committed generated outputs."""
    root = tmp_path / "repo"
    for rel in (TOKENS_REL, *OUTPUT_PATHS):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, target)
    return root


def test_output_paths_are_the_dashboard_css_and_the_tui_theme() -> None:
    assert OUTPUT_PATHS == (CSS_REL, GO_REL)


def test_committed_outputs_are_fresh() -> None:
    assert stale_outputs(REPO_ROOT) == ()


def test_tampered_output_is_reported_stale(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    (root / CSS_REL).write_text("/* hand edited */\n")

    assert stale_outputs(root) == (CSS_REL,)


def test_missing_output_is_reported_stale(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    (root / GO_REL).unlink()

    assert stale_outputs(root) == (GO_REL,)


def test_changed_tokens_make_every_output_stale(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    tokens = root / TOKENS_REL
    tokens.write_text(tokens.read_text().replace("#52B9C8", "#52B9C9"))

    assert stale_outputs(root) == OUTPUT_PATHS


def test_write_refreshes_stale_outputs_and_reports_what_changed(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    (root / CSS_REL).write_text("stale\n")

    written = write_outputs(root)

    assert written == (CSS_REL,)
    assert stale_outputs(root) == ()


def test_write_creates_missing_directories(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    (root / TOKENS_REL).parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / TOKENS_REL, root / TOKENS_REL)

    written = write_outputs(root)

    assert written == OUTPUT_PATHS
    assert (root / GO_REL).read_text() == (REPO_ROOT / GO_REL).read_text()


def test_cli_check_passes_on_fresh_tree() -> None:
    result = runner.invoke(app, ["--check", "--root", str(REPO_ROOT)])

    assert result.exit_code == 0, result.output


def test_cli_check_fails_and_names_the_stale_file(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    (root / GO_REL).write_text("package theme\n")

    result = runner.invoke(app, ["--check", "--root", str(root)])

    assert result.exit_code == 1
    assert str(GO_REL) in result.output


def test_cli_write_then_check_is_green(tmp_path: Path) -> None:
    root = _copy_tree(tmp_path)
    (root / CSS_REL).write_text("stale\n")

    write_result = runner.invoke(app, ["--write", "--root", str(root)])
    check_result = runner.invoke(app, ["--check", "--root", str(root)])

    assert write_result.exit_code == 0, write_result.output
    assert str(CSS_REL) in write_result.output
    assert check_result.exit_code == 0, check_result.output


def test_cli_requires_exactly_one_mode() -> None:
    neither = runner.invoke(app, ["--root", str(REPO_ROOT)])
    both = runner.invoke(app, ["--check", "--write", "--root", str(REPO_ROOT)])

    assert neither.exit_code != 0
    assert both.exit_code != 0


def test_module_is_runnable_with_python_dash_m() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "acceptrate.design", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
