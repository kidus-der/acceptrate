"""The two architectural boundaries, enforced by AST scan.

HORIZONTAL: only backend/mlx_backend.py may import mlx or mlx_lm.
VERTICAL:   bench/ never imports runtime/ or model/; runtime/ and model/ never import bench/.

These pass trivially on an empty skeleton. They exist so that every later
commit is checked, in CI, forever. Do not weaken them.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "acceptrate"
APPLE_SPECIFIC_FILE = SRC / "backend" / "mlx_backend.py"
APPLE_MODULES = frozenset({"mlx", "mlx_lm"})
BENCH_SIDE = ("acceptrate.bench",)
RUNTIME_SIDE = ("acceptrate.runtime", "acceptrate.model")


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module)
    return roots


def _dotted_module(path: Path, root: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    return ".".join(rel.parts)


def find_violations(root: Path, apple_file: Path) -> list[str]:
    """Return one human-readable line per boundary violation under `root`."""
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        module = _dotted_module(path, root)
        imports = _imported_roots(path)
        for name in imports:
            top = name.split(".")[0]
            if top in APPLE_MODULES and path.resolve() != apple_file.resolve():
                violations.append(f"{module} imports {name} (only mlx_backend.py may)")
            if module.startswith(BENCH_SIDE) and name.startswith(RUNTIME_SIDE):
                violations.append(f"{module} imports {name} (bench must not see the runtime)")
            if module.startswith(RUNTIME_SIDE) and name.startswith(BENCH_SIDE):
                violations.append(f"{module} imports {name} (runtime must not see bench)")
    return violations


def _write(root: Path, rel: str, body: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)


def test_scanner_catches_mlx_import_above_the_seam(tmp_path: Path) -> None:
    root = tmp_path / "acceptrate"
    _write(root, "runtime/engine.py", "import mlx.core as mx\n")
    _write(root, "backend/mlx_backend.py", "import mlx.core as mx\n")

    found = find_violations(root, root / "backend" / "mlx_backend.py")

    assert found == ["acceptrate.runtime.engine imports mlx.core (only mlx_backend.py may)"]


def test_scanner_catches_bench_importing_runtime(tmp_path: Path) -> None:
    root = tmp_path / "acceptrate"
    _write(root, "bench/runner.py", "from acceptrate.runtime.adaptive import Scheduler\n")

    found = find_violations(root, root / "backend" / "mlx_backend.py")

    assert len(found) == 1
    assert "bench must not see the runtime" in found[0]


def test_scanner_catches_runtime_importing_bench(tmp_path: Path) -> None:
    root = tmp_path / "acceptrate"
    _write(root, "model/estimator.py", "import acceptrate.bench.guards\n")

    found = find_violations(root, root / "backend" / "mlx_backend.py")

    assert len(found) == 1
    assert "runtime must not see bench" in found[0]


def test_scanner_accepts_a_clean_tree(tmp_path: Path) -> None:
    root = tmp_path / "acceptrate"
    _write(root, "runtime/engine.py", "from acceptrate.backend.protocol import Backend\n")
    _write(root, "bench/runner.py", "from acceptrate.trace.schema import Window\n")
    _write(root, "backend/mlx_backend.py", "import mlx_lm\n")

    assert find_violations(root, root / "backend" / "mlx_backend.py") == []


def test_real_tree_has_no_boundary_violations() -> None:
    assert find_violations(SRC, APPLE_SPECIFIC_FILE) == []
