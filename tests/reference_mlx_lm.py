"""mlx-lm's own generation paths, used as an EXTERNAL reference only.

This file lives under tests/ on purpose. The horizontal boundary says only
src/acceptrate/backend/mlx_backend.py may import mlx or mlx_lm, and
tests/test_boundaries.py scans src/ to enforce it. The reference is test
infrastructure, not product code, so it sits outside the scanned tree.

Stop semantics match runtime.engine.generate_plain exactly: mlx-lm's
stream_generate yields the eos token in its final response (finish_reason
"stop") and caps at max_tokens (finish_reason "length"), so collecting
`.token` over every response gives the same list our engine returns.

Run `uv run python -m tests.reference_mlx_lm` to print the speculative
reference table recorded in docs/reference/mlx_lm_speculative.md.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx_lm
from mlx_lm.sample_utils import make_sampler

from acceptrate.backend.mlx_backend import MLXBackend, MLXTokenizer

GREEDY = make_sampler(temp=0.0)
"""Explicit argmax sampler; the default when no sampler is passed, made visible."""

TABLE_HEADER = (
    "| prompt | tokens | finish | from draft | draft frac | spec tok/s | plain tok/s | vs plain |",
    "|---|---|---|---|---|---|---|---|",
)


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    vocab_size: int


@functools.cache
def load_reference(repo: str) -> LoadedModel:
    """Load once per process so our backend and the reference share one copy of the weights."""
    model, tokenizer, config = mlx_lm.load(repo, return_config=True)
    return LoadedModel(model=model, tokenizer=tokenizer, vocab_size=int(config["vocab_size"]))


def backend_for(repo: str) -> MLXBackend:
    """Our MLXBackend over the same model object the reference generates with."""
    loaded = load_reference(repo)
    return MLXBackend(loaded.model, MLXTokenizer.from_wrapper(loaded.tokenizer), loaded.vocab_size)


@dataclass(frozen=True)
class ReferenceRun:
    """One mlx-lm generation, as mlx-lm itself reports it (single run, no guards)."""

    tokens: tuple[int, ...]
    from_draft: tuple[bool, ...]
    prompt_tps: float
    generation_tps: float
    finish_reason: str | None

    @property
    def draft_fraction(self) -> float:
        return sum(self.from_draft) / len(self.from_draft) if self.from_draft else 0.0


def _run(repo: str, prompt_tokens: Sequence[int], max_tokens: int, **kwargs: Any) -> ReferenceRun:
    loaded = load_reference(repo)
    responses = list(
        mlx_lm.stream_generate(
            loaded.model,
            loaded.tokenizer,
            list(prompt_tokens),
            max_tokens=max_tokens,
            sampler=GREEDY,
            **kwargs,
        )
    )
    last = responses[-1]
    return ReferenceRun(
        tokens=tuple(int(r.token) for r in responses),
        from_draft=tuple(bool(r.from_draft) for r in responses),
        prompt_tps=float(last.prompt_tps),
        generation_tps=float(last.generation_tps),
        finish_reason=last.finish_reason,
    )


def reference_greedy(repo: str, prompt_tokens: Sequence[int], max_tokens: int) -> ReferenceRun:
    """mlx-lm's own plain greedy path."""
    return _run(repo, prompt_tokens, max_tokens)


def reference_greedy_tokens(repo: str, prompt_tokens: Sequence[int], max_tokens: int) -> list[int]:
    """mlx-lm's own greedy continuation of `prompt_tokens`, eos included if produced."""
    return list(reference_greedy(repo, prompt_tokens, max_tokens).tokens)


def reference_speculative(
    target_repo: str,
    draft_repo: str,
    prompt_tokens: Sequence[int],
    max_tokens: int,
    num_draft_tokens: int,
) -> ReferenceRun:
    """mlx-lm's built-in draft-model speculative path at temperature 0."""
    draft = load_reference(draft_repo)
    return _run(
        target_repo,
        prompt_tokens,
        max_tokens,
        draft_model=draft.model,
        num_draft_tokens=num_draft_tokens,
    )


def _main() -> None:
    """Print the reference table for docs/reference/mlx_lm_speculative.md (CLI entry point)."""
    from acceptrate.bench.workloads import as_chat_messages, by_tag, load_corpus
    from acceptrate.config import DEFAULT_PAIR
    from acceptrate.verify.reference import compare_sequences, describe

    max_tokens, k, tags = 64, 4, ("code", "prose", "chat")
    target_repo = DEFAULT_PAIR.target.repo
    draft_repo = DEFAULT_PAIR.draft.repo  # type: ignore[union-attr]
    tokenizer = backend_for(target_repo).tokenizer
    corpus = load_corpus()
    print("\n".join(TABLE_HEADER))
    for tag in tags:
        prompt = by_tag(tag, corpus)[0]
        tokens = tokenizer.encode_chat(as_chat_messages(prompt))
        plain = reference_greedy(target_repo, tokens, max_tokens)
        spec = reference_speculative(target_repo, draft_repo, tokens, max_tokens, k)
        report = compare_sequences(prompt.id, spec.tokens, plain.tokens)
        match = "identical" if report.matched else describe(report)
        print(
            f"| {prompt.id} | {len(spec.tokens)} | {spec.finish_reason} | "
            f"{sum(spec.from_draft)} | {spec.draft_fraction:.3f} | "
            f"{spec.generation_tps:.2f} | {plain.generation_tps:.2f} | {match} |"
        )


if __name__ == "__main__":
    _main()
