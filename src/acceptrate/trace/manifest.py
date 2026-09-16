"""The run manifest: which config and environment produced a trace directory.

`run_id` is a hash of the canonical config so the same sweep cell always lands
in the same directory and two runs of it are comparable by construction. The
environment fields exist because a version bump or a different chip silently
invalidates the dataset (CLAUDE.md, Stack); every probe is injectable so tests
never shell out.

This module must not import mlx or mlx_lm (tests/test_boundaries.py); versions
come from package metadata instead.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import psutil
from pydantic import BaseModel, ConfigDict

RUN_ID_HEX_CHARS = 12
UNKNOWN = "unknown"
PROBE_TIMEOUT_S = 5.0
GB = 1024**3

CHIP_PROBE = ("sysctl", "-n", "machdep.cpu.brand_string")
GIT_SHA_PROBE = ("git", "rev-parse", "HEAD")


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    config: dict[str, Any]
    created_at: str  # ISO-8601, UTC
    mlx_version: str
    mlx_lm_version: str
    python_version: str
    chip: str
    memory_gb: float
    git_sha: str = UNKNOWN


def run_id_for(config: dict[str, Any]) -> str:
    """Stable short hash of the canonical JSON (sorted keys) of a config.

    Raises TypeError if the config is not JSON-serialisable: a config that
    cannot be written to the manifest cannot identify a run either.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:RUN_ID_HEX_CHARS]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def command_output(args: Sequence[str]) -> str:
    """Stripped stdout of a probe command, or UNKNOWN if it cannot run."""
    try:
        completed = subprocess.run(
            list(args), capture_output=True, text=True, check=True, timeout=PROBE_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return completed.stdout.strip() or UNKNOWN


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return UNKNOWN


def chip_name() -> str:
    return command_output(CHIP_PROBE)


def git_sha() -> str:
    return command_output(GIT_SHA_PROBE)


def total_memory_gb() -> float:
    return psutil.virtual_memory().total / GB


def build_manifest(
    config: dict[str, Any],
    *,
    now: Callable[[], str] = utc_now_iso,
    chip: Callable[[], str] = chip_name,
    memory_gb: Callable[[], float] = total_memory_gb,
    git_sha: Callable[[], str] = git_sha,
    package_version: Callable[[str], str] = package_version,
) -> RunManifest:
    """Assemble a manifest for `config`, probing the environment via the given callables."""
    return RunManifest(
        run_id=run_id_for(config),
        config=config,
        created_at=now(),
        mlx_version=package_version("mlx"),
        mlx_lm_version=package_version("mlx-lm"),
        python_version=sys.version.split()[0],
        chip=chip(),
        memory_gb=memory_gb(),
        git_sha=git_sha(),
    )
