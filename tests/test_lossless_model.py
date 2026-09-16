"""Greedy equivalence on real weights. Needs models; never runs in CI.

The first test is the sharpest: with the 1B model as BOTH target and draft,
every draft token equals the sequential argmax, so any divergence can only
come from batched-verify vs sequential-decode fp16 differences flipping a
near-tie argmax. That is the failure mode P2 exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from acceptrate.bench.workloads import as_chat_messages, by_tag, load_corpus
from acceptrate.config import DEFAULT_PAIR
from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.runtime.speculative import generate_speculative
from acceptrate.verify.lossless import compare_generation

pytestmark = pytest.mark.model


@dataclass(frozen=True)
class NullGuard:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


def _prompts(n_per_tag: int):
    corpus = load_corpus()
    return [p for tag in ("code", "prose", "reason") for p in by_tag(tag, corpus)[:n_per_tag]]


def _check(target, draft, prompt, k: int, max_tokens: int):
    tokens = target.tokenizer.encode_chat(as_chat_messages(prompt))
    eos = target.tokenizer.eos_token_ids
    ctx = GenerationContext("t", prompt.tag, prompt.id, 0)
    plain = generate_plain(target, tokens, max_tokens, eos, ctx, NullGuard)
    spec = generate_speculative(target, draft, tokens, max_tokens, eos, ctx, NullGuard, k=k)
    return compare_generation(prompt.id, k, plain.tokens, spec.tokens, spec.rows)


@pytest.fixture(scope="module")
def draft():
    from acceptrate.backend.mlx_backend import MLXBackend

    return MLXBackend.load(DEFAULT_PAIR.draft.repo)  # type: ignore[union-attr]


def test_1b_as_its_own_draft_is_token_identical_and_accepts_nearly_everything(draft) -> None:
    from acceptrate.backend.mlx_backend import MLXBackend

    # A second instance: target and draft must own separate KV caches.
    target = MLXBackend.load(DEFAULT_PAIR.draft.repo)  # type: ignore[union-attr]

    reports = [_check(target, draft, p, k=4, max_tokens=64) for p in _prompts(1)]

    for r in reports:
        assert r.matched, (r.prompt_id, r.first_divergence, r.plain_token, r.spec_token)
        assert r.alpha > 0.9, (r.prompt_id, r.alpha)


def test_8b_target_with_1b_draft_is_token_identical_on_one_prompt(draft) -> None:
    from acceptrate.backend.mlx_backend import MLXBackend

    target = MLXBackend.load(DEFAULT_PAIR.target.repo)
    prompt = by_tag("code")[0]

    report = _check(target, draft, prompt, k=4, max_tokens=64)

    assert report.matched, (report.first_divergence, report.plain_token, report.spec_token)
    assert 0.0 < report.alpha <= 1.0
