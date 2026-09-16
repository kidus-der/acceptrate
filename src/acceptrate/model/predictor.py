"""Layer 3: the cold-start alpha prior.

The EWMA is blind for the first few windows. A small tabular regressor over
prompt features (structure, code fences, JSON-ness, length, character
entropy) predicts the opening alpha so early tokens are not a guess. It is a
HistGradientBoostingRegressor on a few thousand rows: scikit-learn, never
PyTorch (CLAUDE.md). Trained per machine class from sweep traces.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURE_NAMES: tuple[str, ...] = (
    "n_chars",
    "n_lines",
    "n_words",
    "code_fences",
    "brace_ratio",
    "bracket_ratio",
    "digit_ratio",
    "punct_ratio",
    "upper_ratio",
    "char_entropy",
    "mean_word_len",
    "code_keywords",
    "question_marks",
)
MIN_TRAINING_ROWS = 20
_CODE_WORDS = re.compile(
    r"\b(def|class|import|return|function|const|let|var|SELECT|FROM|WHERE|fn|struct|impl)\b"
)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def prompt_features(text: str) -> npt.NDArray[np.float64]:
    n = max(len(text), 1)
    words = text.split()
    return np.array(
        [
            len(text),
            text.count("\n") + 1,
            len(words),
            1.0 if "```" in text else 0.0,
            (text.count("{") + text.count("}")) / n,
            (text.count("[") + text.count("]") + text.count("(") + text.count(")")) / n,
            sum(ch.isdigit() for ch in text) / n,
            sum(ch in ".,;:!?" for ch in text) / n,
            sum(ch.isupper() for ch in text) / n,
            _entropy(text),
            (sum(len(w) for w in words) / len(words)) if words else 0.0,
            len(_CODE_WORDS.findall(text)),
            text.count("?"),
        ],
        dtype=np.float64,
    )


def alpha_per_prompt(df: pl.DataFrame) -> dict[str, float]:
    """Pooled accepted / proposed per prompt over every K >= 1 window."""
    pooled = (
        df.filter(pl.col("k_proposed") > 0)
        .group_by("prompt_id")
        .agg(acc=pl.col("n_accepted").sum(), prop=pl.col("k_proposed").sum())
        .filter(pl.col("prop") > 0)
    )
    return {row[0]: float(row[1]) / float(row[2]) for row in pooled.iter_rows()}


@dataclass(frozen=True)
class Predictor:
    model: HistGradientBoostingRegressor

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.clip(self.model.predict(np.atleast_2d(features)), 0.0, 1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> Predictor:
        return cls(model=joblib.load(path))


def train_predictor(
    features: npt.NDArray[np.float64], alphas: npt.NDArray[np.float64], seed: int = 0
) -> Predictor:
    if len(alphas) < MIN_TRAINING_ROWS:
        raise ValueError(f"need at least {MIN_TRAINING_ROWS} rows to train, got {len(alphas)}")
    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5, random_state=seed
    )
    model.fit(features, alphas)
    return Predictor(model=model)


def evaluate_mae(
    predictor: Predictor, features: npt.NDArray[np.float64], alphas: npt.NDArray[np.float64]
) -> float:
    return float(np.mean(np.abs(predictor.predict(features) - alphas)))


def constant_prior_mae(
    train_alphas: npt.NDArray[np.float64], test_alphas: npt.NDArray[np.float64]
) -> float:
    """The baseline the gate is measured against: always predict the training mean."""
    return float(np.mean(np.abs(test_alphas - float(np.mean(train_alphas)))))
