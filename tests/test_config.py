"""Model-pair config is validated at the boundary (Pydantic v2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from acceptrate.config import DEFAULT_PAIR, ModelPairConfig, ModelSpec


def test_default_pair_is_the_llama3_8b_1b_pair_from_the_brief() -> None:
    assert DEFAULT_PAIR.target.repo == "mlx-community/Llama-3.1-8B-Instruct-4bit"
    assert DEFAULT_PAIR.draft.repo == "mlx-community/Llama-3.2-1B-Instruct-4bit"


def test_default_pair_has_a_memory_estimate_that_fits_the_brief() -> None:
    # tab 3: target 4.5 GB, draft 0.7 GB, KV @8k 1.1 GB
    assert 4.0 <= DEFAULT_PAIR.target.weights_gb <= 5.0
    assert 0.5 <= DEFAULT_PAIR.draft.weights_gb <= 1.0
    assert 5.0 <= DEFAULT_PAIR.estimated_gb <= 7.0


def test_repo_must_look_like_an_hf_repo_id() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(repo="not a repo", weights_gb=1.0)


def test_weights_gb_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(repo="org/model", weights_gb=0)


def test_config_is_frozen() -> None:
    with pytest.raises(ValidationError):
        DEFAULT_PAIR.target.repo = "org/other"  # type: ignore[misc]


def test_pair_without_draft_is_allowed_for_baseline_runs() -> None:
    pair = ModelPairConfig(target=DEFAULT_PAIR.target, draft=None)

    assert pair.draft is None
    assert pair.estimated_gb < DEFAULT_PAIR.estimated_gb
