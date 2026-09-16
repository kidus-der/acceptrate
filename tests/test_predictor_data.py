"""model/predictor_data.py — pair per-prompt alpha from traces with corpus text features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acceptrate.model.predictor import FEATURE_NAMES
from acceptrate.model.predictor_data import build_dataset
from acceptrate.trace.schema import frame_from_rows, make_row


@dataclass(frozen=True)
class P:
    id: str
    tag: str
    text: str
    split: str


CORPUS = (
    P("code-001", "code", "```python\ndef f(): pass\n```", "train"),
    P("prose-001", "prose", "Write about the sea.", "train"),
    P("chat-004", "chat", "hey, what's up?", "heldout"),
)


def _rows(pid: str, accepts: list[int]):
    return [
        make_row("r", i, i, 4, n, 8.0, 50.0, "x", 0, 0, 0, pid, 0, 60.0)
        for i, n in enumerate(accepts)
    ]


def test_build_dataset_pairs_alpha_with_features_for_the_requested_split() -> None:
    df = frame_from_rows(
        _rows("code-001", [4, 4]) + _rows("prose-001", [1, 2]) + _rows("chat-004", [0, 1])
    )

    ds = build_dataset(df, CORPUS, split="train")

    assert ds.prompt_ids == ("code-001", "prose-001")
    assert ds.features.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_allclose(ds.alphas, [1.0, 3 / 8])


def test_build_dataset_skips_prompts_without_windows() -> None:
    df = frame_from_rows(_rows("code-001", [4]))

    ds = build_dataset(df, CORPUS, split="train")

    assert ds.prompt_ids == ("code-001",)


def test_heldout_split_selects_only_heldout_prompts() -> None:
    df = frame_from_rows(_rows("code-001", [4]) + _rows("chat-004", [2]))

    ds = build_dataset(df, CORPUS, split="heldout")

    assert ds.prompt_ids == ("chat-004",)
