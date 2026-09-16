"""runtime/lookup.py — prompt-lookup drafting as a Backend (c ~ 0, no draft model).

Predicts the next token by matching the most recent n-gram against earlier
text (prompt + committed output). Repetitive text — code, JSON, quoted
passages — drafts almost for free; novel prose drafts nothing useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acceptrate.backend.protocol import Backend
from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.runtime.lookup import LookupBackend
from acceptrate.runtime.speculative import generate_speculative
from tests.fakes import FakeBackend

CTX = GenerationContext(run_id="r", workload_tag="code", prompt_id="code-001", rep=0)


@dataclass(frozen=True)
class Snap:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


def test_satisfies_the_backend_protocol() -> None:
    assert isinstance(LookupBackend(vocab_size=32), Backend)


def test_predicts_the_token_that_followed_the_last_matching_ngram() -> None:
    lb = LookupBackend(vocab_size=32, max_ngram=3)
    # sequence: 1 2 3 9 | 1 2 3 -> the 3-gram (1,2,3) was followed by 9
    lb.prefill([1, 2, 3, 9, 1, 2])

    logits = lb.decode_step(3)

    assert int(np.argmax(logits)) == 9
    assert lb.position == 7


def test_prefers_the_longest_match_then_the_most_recent() -> None:
    lb = LookupBackend(vocab_size=32, max_ngram=3)
    # (5,6) -> 7 early, later (4,5,6) -> 8: the 3-gram match wins over the 2-gram
    lb.prefill([5, 6, 7, 0, 4, 5, 6, 8, 0, 4, 5])

    assert int(np.argmax(lb.decode_step(6))) == 8


def test_no_match_gives_a_deterministic_fallback_that_cannot_be_accepted_by_accident() -> None:
    lb = LookupBackend(vocab_size=32, max_ngram=3, fallback=0)
    lb.prefill([1, 2, 3])

    logits = lb.decode_step(30)

    assert int(np.argmax(logits)) == 0


def test_verify_appends_and_returns_one_row_per_token_and_trim_rolls_back() -> None:
    lb = LookupBackend(vocab_size=32)
    lb.prefill([1, 2, 3, 4, 1, 2])

    rows = lb.verify([3, 4])

    assert rows.shape == (2, 32)
    assert int(np.argmax(rows[0])) == 4  # after 1,2,3 -> 4
    assert lb.position == 8
    lb.trim(2)
    assert lb.position == 6
    assert lb.tokens == (1, 2, 3, 4, 1, 2)


def test_logits_are_float32_rows_over_the_vocab() -> None:
    lb = LookupBackend(vocab_size=32)
    lb.prefill([1])

    logits = lb.decode_step(2)

    assert logits.dtype == np.float32 and logits.shape == (32,)


def test_speculative_with_lookup_draft_is_lossless_and_accepts_on_repetition() -> None:
    target = FakeBackend(32, step=7)  # the fake's rule is periodic: lookup should catch on
    plain = generate_plain(FakeBackend(32, step=7), [3], 60, frozenset(), CTX, Snap)

    result = generate_speculative(target, LookupBackend(32), [3], 60, frozenset(), CTX, Snap, k=4)

    assert result.tokens == plain.tokens
    accepted = sum(r[4] for r in result.rows)
    assert accepted > len(result.rows)  # more than one accepted token per window on average


def test_rejects_bad_config() -> None:
    import pytest

    with pytest.raises(ValueError):
        LookupBackend(vocab_size=32, max_ngram=0)
    with pytest.raises(ValueError):
        LookupBackend(vocab_size=0)
