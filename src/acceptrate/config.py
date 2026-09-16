"""Model configuration, validated at the boundary.

The default pair is the one the brief names: an 8B 4-bit target with a 1B
4-bit draft that shares its tokenizer. Weight sizes are nominal and used only
for the startup memory-fit guard; the runtime measures everything else live.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

HF_REPO_PATTERN = r"^[\w.-]+/[\w.-]+$"
KV_CACHE_GB_AT_8K = 1.1
"""Nominal KV-cache footprint for the 8B target at 8k context (brief tab 3)."""


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo: str = Field(pattern=HF_REPO_PATTERN, description="Hugging Face repo id")
    weights_gb: float = Field(gt=0, description="Nominal on-disk weight size in GB")


class ModelPairConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: ModelSpec
    draft: ModelSpec | None = None
    kv_cache_gb: float = Field(default=KV_CACHE_GB_AT_8K, ge=0)

    @property
    def estimated_gb(self) -> float:
        draft_gb = self.draft.weights_gb if self.draft else 0.0
        return self.target.weights_gb + draft_gb + self.kv_cache_gb


DEFAULT_PAIR = ModelPairConfig(
    target=ModelSpec(repo="mlx-community/Llama-3.1-8B-Instruct-4bit", weights_gb=4.5),
    draft=ModelSpec(repo="mlx-community/Llama-3.2-1B-Instruct-4bit", weights_gb=0.7),
)
