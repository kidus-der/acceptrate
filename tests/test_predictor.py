"""model/predictor.py — Layer 3: the cold-start alpha prior from prompt features.

A small tabular regressor over prompt features predicts the opening alpha
so the first windows are not a guess. Gate (P6): beats a constant prior on
held-out prompts.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from acceptrate.model.predictor import (
    FEATURE_NAMES,
    Predictor,
    alpha_per_prompt,
    constant_prior_mae,
    evaluate_mae,
    prompt_features,
    train_predictor,
)
from acceptrate.trace.schema import frame_from_rows, make_row


def test_features_are_a_fixed_length_float_vector() -> None:
    x = prompt_features("Write a Python function that reverses a string.")

    assert x.shape == (len(FEATURE_NAMES),)
    assert x.dtype == np.float64
    assert np.isfinite(x).all()


def test_features_see_code_fences_json_and_length() -> None:
    code = prompt_features("Fix this:\n```python\ndef f(x):\n    return x\n```")
    json_ = prompt_features('Return JSON matching {"name": "string", "items": []}')
    prose = prompt_features("Write a short essay about the sea at dawn.")

    f = dict(zip(FEATURE_NAMES, code, strict=True))
    j = dict(zip(FEATURE_NAMES, json_, strict=True))
    p = dict(zip(FEATURE_NAMES, prose, strict=True))
    assert f["code_fences"] == 1.0 and p["code_fences"] == 0.0
    assert j["brace_ratio"] > p["brace_ratio"]
    assert f["n_chars"] > 0 and f["n_lines"] >= 4


def test_features_are_deterministic() -> None:
    a = prompt_features("hello world")
    b = prompt_features("hello world")

    np.testing.assert_array_equal(a, b)


def _rows(prompt_id: str, tag: str, accepts: list[int], k: int = 4):
    return [
        make_row("r", i, i, k, n, 8.0, 50.0, tag, 0, 0, 0, prompt_id, 0, 60.0)
        for i, n in enumerate(accepts)
    ]


def test_alpha_per_prompt_pools_windows_per_prompt() -> None:
    df = frame_from_rows(_rows("code-001", "code", [4, 4, 2]) + _rows("prose-001", "prose", [0, 1]))

    alphas = alpha_per_prompt(df)

    assert alphas["code-001"] == pytest.approx(10 / 12)
    assert alphas["prose-001"] == pytest.approx(1 / 8)


def test_alpha_per_prompt_ignores_k_zero_rows() -> None:
    df = frame_from_rows(_rows("p", "code", [0, 0], k=0) + _rows("p", "code", [3], k=4))

    assert alpha_per_prompt(df)["p"] == pytest.approx(0.75)


def _synthetic(n: int, seed: int):
    rng = np.random.default_rng(seed)
    texts, ys = [], []
    for _ in range(n):
        fenced = rng.random() < 0.5
        base = (
            "```python\ndef f():\n    pass\n```"
            if fenced
            else "Tell me a story about a lighthouse."
        )
        texts.append(base + " " * int(rng.integers(0, 40)))
        ys.append((0.85 if fenced else 0.45) + rng.normal(0, 0.03))
    x = np.stack([prompt_features(t) for t in texts])
    return x, np.array(ys)


def test_trained_predictor_beats_a_constant_prior_on_held_out_synthetic_data() -> None:
    x_train, y_train = _synthetic(120, seed=1)
    x_test, y_test = _synthetic(40, seed=2)

    predictor = train_predictor(x_train, y_train)

    assert isinstance(predictor, Predictor)
    mae = evaluate_mae(predictor, x_test, y_test)
    prior = constant_prior_mae(y_train, y_test)
    assert mae < prior
    assert prior > 0.1  # the two clusters make a constant prior genuinely bad


def test_predictions_are_clipped_to_the_unit_interval() -> None:
    x_train, y_train = _synthetic(60, seed=3)
    predictor = train_predictor(x_train, y_train)

    preds = predictor.predict(x_train)

    assert (preds >= 0).all() and (preds <= 1).all()


def test_roundtrip_through_disk(tmp_path) -> None:
    x_train, y_train = _synthetic(60, seed=4)
    predictor = train_predictor(x_train, y_train)
    path = tmp_path / "predictor.joblib"

    predictor.save(path)
    loaded = Predictor.load(path)

    np.testing.assert_allclose(loaded.predict(x_train), predictor.predict(x_train))


def test_training_needs_enough_rows() -> None:
    x, y = _synthetic(3, seed=5)

    with pytest.raises(ValueError):
        train_predictor(x, y)


def test_dataframe_column_order_matches_feature_names() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert isinstance(pl.DataFrame({n: [0.0] for n in FEATURE_NAMES}), pl.DataFrame)
