"""P2 cross-check: our MLXBackend + generate_plain against mlx-lm's own greedy path.

mlx-lm is an EXTERNAL reference (CLAUDE.md, "why the engine owns the loop").
If our greedy tokens ever differ from its greedy tokens on the same prompt, the
divergence index and both tokens are in the assertion message — investigate,
never loosen. Needs weights and an Apple GPU; never runs in CI.

Machine etiquette: the 1B draft model is used for the six-tag sweep and the 8B
target only for the final three prompts, so the whole file loads each model
exactly once (module-scoped fixtures share the weights with the reference).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from acceptrate.bench.workloads import Prompt, as_chat_messages, by_tag, load_corpus
from acceptrate.config import DEFAULT_PAIR
from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.trace.schema import WORKLOAD_TAGS
from acceptrate.verify.reference import compare_sequences, describe
from tests.reference_mlx_lm import (
    backend_for,
    reference_greedy_tokens,
    reference_speculative,
)

pytestmark = pytest.mark.model

MAX_TOKENS = 64
NUM_DRAFT_TOKENS = 4
TARGET_REPO = DEFAULT_PAIR.target.repo
DRAFT_REPO = DEFAULT_PAIR.draft.repo  # type: ignore[union-attr]
TARGET_TAGS = ("code", "prose", "chat")
"""The 8B target is checked on three tags only; the 1B covers all six."""


@dataclass(frozen=True)
class NullSnapshot:
    mem_pressure: int = 0
    page_ins: int = 0
    thermal_level: int = 0


@pytest.fixture(scope="module")
def corpus() -> tuple[Prompt, ...]:
    return load_corpus()


@pytest.fixture(scope="module")
def draft_backend():
    return backend_for(DRAFT_REPO)


@pytest.fixture(scope="module")
def target_backend():
    return backend_for(TARGET_REPO)


def _first_prompt(tag: str, corpus: tuple[Prompt, ...]) -> Prompt:
    return by_tag(tag, corpus)[0]


def _ours(backend, prompt: Prompt) -> tuple[list[int], list[int]]:
    """(prompt tokens, our greedy continuation) for one corpus prompt."""
    tokens = backend.tokenizer.encode_chat(as_chat_messages(prompt))
    ctx = GenerationContext(run_id="ref", workload_tag=prompt.tag, prompt_id=prompt.id, rep=0)
    result = generate_plain(
        backend, tokens, MAX_TOKENS, backend.tokenizer.eos_token_ids, ctx, NullSnapshot
    )
    return tokens, list(result.tokens)


@pytest.mark.parametrize("tag", WORKLOAD_TAGS)
def test_draft_model_greedy_tokens_match_mlx_lm(draft_backend, corpus, tag: str) -> None:
    prompt = _first_prompt(tag, corpus)
    tokens, ours = _ours(draft_backend, prompt)

    reference = reference_greedy_tokens(DRAFT_REPO, tokens, MAX_TOKENS)

    report = compare_sequences(prompt.id, ours, reference)
    assert report.matched, describe(report)
    assert len(ours) == MAX_TOKENS or ours[-1] in draft_backend.tokenizer.eos_token_ids


@pytest.mark.parametrize("tag", TARGET_TAGS)
def test_target_model_greedy_tokens_match_mlx_lm(target_backend, corpus, tag: str) -> None:
    prompt = _first_prompt(tag, corpus)
    tokens, ours = _ours(target_backend, prompt)

    reference = reference_greedy_tokens(TARGET_REPO, tokens, MAX_TOKENS)

    report = compare_sequences(prompt.id, ours, reference)
    assert report.matched, describe(report)
    assert len(ours) == MAX_TOKENS or ours[-1] in target_backend.tokenizer.eos_token_ids


@pytest.mark.parametrize("tag", TARGET_TAGS)
def test_mlx_lm_speculative_path_is_token_identical_to_its_greedy_path(
    target_backend, corpus, tag: str
) -> None:
    """External data point: mlx-lm's own draft-model path is lossless at T=0 too."""
    prompt = _first_prompt(tag, corpus)
    tokens = target_backend.tokenizer.encode_chat(as_chat_messages(prompt))
    reference = reference_greedy_tokens(TARGET_REPO, tokens, MAX_TOKENS)

    run = reference_speculative(TARGET_REPO, DRAFT_REPO, tokens, MAX_TOKENS, NUM_DRAFT_TOKENS)

    report = compare_sequences(prompt.id, run.tokens, reference)
    assert report.matched, describe(report)
    assert len(run.from_draft) == len(run.tokens)
    assert 0.0 <= run.draft_fraction <= 1.0
    assert run.generation_tps > 0.0
