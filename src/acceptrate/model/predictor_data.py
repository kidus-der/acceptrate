"""Pair per-prompt acceptance (from traces) with prompt-text features (from the corpus)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import polars as pl

from acceptrate.model.predictor import alpha_per_prompt, prompt_features


class CorpusPrompt(Protocol):
    id: str
    tag: str
    text: str
    split: str


@dataclass(frozen=True)
class PredictorDataset:
    prompt_ids: tuple[str, ...]
    features: npt.NDArray[np.float64]
    alphas: npt.NDArray[np.float64]


def build_dataset(df: pl.DataFrame, corpus: Sequence[CorpusPrompt], split: str) -> PredictorDataset:
    alphas = alpha_per_prompt(df)
    chosen = [p for p in corpus if p.split == split and p.id in alphas]
    if not chosen:
        return PredictorDataset((), np.zeros((0, 0)), np.zeros(0))
    features = np.stack([prompt_features(p.text) for p in chosen])
    return PredictorDataset(
        prompt_ids=tuple(p.id for p in chosen),
        features=features,
        alphas=np.array([alphas[p.id] for p in chosen], dtype=np.float64),
    )
