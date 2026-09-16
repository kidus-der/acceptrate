"""tests/fakes_prob.py — the probabilistic fake honours the Backend protocol and is non-trivial."""

from __future__ import annotations

import numpy as np

from acceptrate.backend.protocol import Backend
from acceptrate.model.sampling import softmax_with_temperature
from tests.fakes_prob import ProbFake


def test_prob_fake_satisfies_the_backend_protocol() -> None:
    assert isinstance(ProbFake(), Backend)


def test_logits_are_deterministic_in_seed_and_not_one_hot() -> None:
    a, b = ProbFake(32, seed=7), ProbFake(32, seed=7)

    logits = a.prefill([3])

    assert np.array_equal(logits, b.prefill([3]))
    assert logits.dtype == np.float32
    probs = softmax_with_temperature(logits, 1.0)
    assert 0.05 < probs.max() < 0.95  # a real distribution, not a delta


def test_different_seeds_give_different_tables() -> None:
    assert not np.array_equal(ProbFake(32, seed=1).table, ProbFake(32, seed=2).table)


def test_verify_and_trim_keep_the_cache_honest() -> None:
    fake = ProbFake(32, seed=0)
    fake.prefill([1, 2])

    rows = fake.verify([5, 6, 7])
    fake.trim(2)

    assert rows.shape == (3, 32)
    assert np.array_equal(rows[1], fake.table[6])
    assert fake.tokens == (1, 2, 5)
    assert fake.position == 3


def test_perturbed_draft_stays_close_to_its_target() -> None:
    target = ProbFake(32, seed=0)

    draft = ProbFake.perturbed(target, seed=1, noise=0.5)

    delta = np.abs(draft.table - target.table)
    assert 0.0 < float(delta.mean()) < 1.0
    assert draft.vocab_size == target.vocab_size
