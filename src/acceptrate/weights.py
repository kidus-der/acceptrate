"""Nominal weight sizes from the local Hugging Face cache.

Used only by the startup memory-fit guard for repos that are not the
default pair. Never downloads: a repo that is not cached raises.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from acceptrate.config import ModelSpec

GB = 1024**3


def _resolve_cached(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, local_files_only=True))


def cached_weights_gb(repo: str, resolve: Callable[[str], Path] = _resolve_cached) -> float:
    snapshot = resolve(repo)
    total = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
    return total / GB


def cached_spec(repo: str, resolve: Callable[[str], Path] = _resolve_cached) -> ModelSpec:
    return ModelSpec(repo=repo, weights_gb=max(cached_weights_gb(repo, resolve), 1e-6))
