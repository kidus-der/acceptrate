"""The run manifest: which config and environment produced a trace directory."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from acceptrate.trace.manifest import (
    RUN_ID_HEX_CHARS,
    UNKNOWN,
    RunManifest,
    build_manifest,
    command_output,
    package_version,
    run_id_for,
)

CONFIG = {"k": 4, "workload": "code", "pair": {"target": "t", "draft": "d"}}


def _fake_manifest(config: dict = CONFIG) -> RunManifest:
    return build_manifest(
        config,
        now=lambda: "2026-09-16T12:00:00+00:00",
        chip=lambda: "Apple M4",
        memory_gb=lambda: 16.0,
        git_sha=lambda: "abc123",
        package_version=lambda name: f"{name}-version",
    )


def test_run_id_is_twelve_hex_chars() -> None:
    run_id = run_id_for(CONFIG)

    assert len(run_id) == RUN_ID_HEX_CHARS
    int(run_id, 16)


def test_run_id_is_stable_and_ignores_key_order() -> None:
    reordered = {"workload": "code", "pair": {"draft": "d", "target": "t"}, "k": 4}

    assert run_id_for(CONFIG) == run_id_for(CONFIG)
    assert run_id_for(CONFIG) == run_id_for(reordered)


def test_run_id_changes_when_config_changes() -> None:
    changed = {**CONFIG, "k": 5}

    assert run_id_for(changed) != run_id_for(CONFIG)


def test_run_id_rejects_config_that_is_not_json_serialisable() -> None:
    with pytest.raises(TypeError):
        run_id_for({"when": object()})


def test_build_manifest_uses_injected_environment_and_hashes_the_config() -> None:
    manifest = _fake_manifest()

    assert manifest.run_id == run_id_for(CONFIG)
    assert manifest.config == CONFIG
    assert manifest.created_at == "2026-09-16T12:00:00+00:00"
    assert manifest.chip == "Apple M4"
    assert manifest.memory_gb == 16.0
    assert manifest.git_sha == "abc123"
    assert manifest.mlx_version == "mlx-version"
    assert manifest.mlx_lm_version == "mlx-lm-version"


def test_build_manifest_records_the_running_python_version() -> None:
    import sys

    manifest = _fake_manifest()

    assert manifest.python_version == sys.version.split()[0]


def test_build_manifest_defaults_to_an_iso8601_utc_timestamp() -> None:
    manifest = build_manifest(
        CONFIG,
        chip=lambda: "x",
        memory_gb=lambda: 1.0,
        git_sha=lambda: "x",
        package_version=lambda _: "x",
    )

    parsed = datetime.fromisoformat(manifest.created_at)
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_manifest_roundtrips_through_json() -> None:
    manifest = _fake_manifest()

    restored = RunManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest


def test_manifest_is_frozen() -> None:
    manifest = _fake_manifest()

    with pytest.raises(ValidationError):
        manifest.run_id = "other"  # type: ignore[misc]


def test_command_output_is_unknown_when_the_command_cannot_run() -> None:
    assert command_output(["/definitely/not/a/binary"]) == UNKNOWN


def test_command_output_strips_whitespace() -> None:
    assert command_output(["echo", "  Apple M4  "]) == "Apple M4"


def test_package_version_is_unknown_for_a_missing_package() -> None:
    assert package_version("no-such-package-acceptrate") == UNKNOWN
