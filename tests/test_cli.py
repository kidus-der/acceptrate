"""CLI surface: the bench group keeps --smoke (P0 gate) and gains compare (P1 gate)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from acceptrate.cli import app
from acceptrate.trace import TraceWriter, build_manifest, run_id_for
from acceptrate.trace.schema import make_row

runner = CliRunner()


def _write_run(root: Path, name: str, config: dict, window_ms: float) -> Path:
    run_dir = root / name
    manifest = build_manifest(
        config, now=lambda: "t", chip=lambda: "c", memory_gb=lambda: 16.0, git_sha=lambda: "s"
    )
    with TraceWriter(run_dir, manifest) as writer:
        for g in range(5):
            rows = [
                make_row(
                    manifest.run_id,
                    w,
                    w + 1,
                    0,
                    0,
                    0.0,
                    window_ms,
                    "code",
                    0,
                    0,
                    0,
                    f"p{g}",
                    0,
                    window_ms,
                )
                for w in range(4)
            ]
            writer.end_generation(rows)
    return run_dir


def test_bench_without_flag_or_subcommand_fails_with_guidance() -> None:
    result = runner.invoke(app, ["bench"])

    assert result.exit_code == 1
    assert "--smoke" in result.output


def test_compare_passes_for_two_agreeing_runs(tmp_path: Path) -> None:
    cfg = {"arm": "plain", "k": 0}
    a = _write_run(tmp_path, "a", cfg, 50.0)
    b = _write_run(tmp_path, "b", cfg, 50.5)

    result = runner.invoke(app, ["bench", "compare", str(a), str(b)])

    assert result.exit_code == 0, result.output
    assert "P1 gate: PASS" in result.output


def test_compare_fails_for_runs_outside_two_percent(tmp_path: Path) -> None:
    cfg = {"arm": "plain", "k": 0}
    a = _write_run(tmp_path, "a", cfg, 50.0)
    b = _write_run(tmp_path, "b", cfg, 53.0)

    result = runner.invoke(app, ["bench", "compare", str(a), str(b)])

    assert result.exit_code == 1
    assert "P1 gate: FAIL" in result.output


def test_compare_refuses_runs_with_different_configs(tmp_path: Path) -> None:
    a = _write_run(tmp_path, "a", {"k": 0}, 50.0)
    b = _write_run(tmp_path, "b", {"k": 4}, 50.0)
    assert run_id_for({"k": 0}) != run_id_for({"k": 4})

    result = runner.invoke(app, ["bench", "compare", str(a), str(b)])

    assert result.exit_code == 1
    assert "run_id mismatch" in result.output
