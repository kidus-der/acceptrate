"""The prompt corpus: tagged, validated at the boundary, split fixed before any measurement.

The held-out split is a pure function of the prompt id, so it was decided before
a single trace row existed and can never be tuned on. These tests pin that down.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from acceptrate.bench.workloads import (
    MAX_TEXT_CHARS,
    MIN_PROMPTS_PER_TAG,
    MIN_TEXT_CHARS,
    CorpusError,
    Prompt,
    as_chat_messages,
    by_tag,
    load_corpus,
    mixed_workload,
    split,
    split_for,
)
from acceptrate.trace.schema import WORKLOAD_TAGS

ID_RE = re.compile(r"^(code|json|prose|chat|summarize|reason)-\d{3}$")


@pytest.fixture(scope="module")
def corpus() -> tuple[Prompt, ...]:
    return load_corpus()


# --- corpus shape -----------------------------------------------------------


def test_every_tag_has_at_least_the_minimum_number_of_prompts(corpus: tuple[Prompt, ...]) -> None:
    counts = Counter(p.tag for p in corpus)

    assert MIN_PROMPTS_PER_TAG >= 24
    for tag in WORKLOAD_TAGS:
        assert counts[tag] >= MIN_PROMPTS_PER_TAG, f"{tag}: {counts[tag]} prompts"


def test_no_prompt_carries_a_tag_outside_the_schema(corpus: tuple[Prompt, ...]) -> None:
    assert {p.tag for p in corpus} == set(WORKLOAD_TAGS)


def test_ids_are_unique_and_well_formed(corpus: tuple[Prompt, ...]) -> None:
    ids = [p.id for p in corpus]

    assert len(ids) == len(set(ids))
    for prompt in corpus:
        assert ID_RE.match(prompt.id), prompt.id
        assert prompt.id.startswith(f"{prompt.tag}-"), (prompt.id, prompt.tag)


def test_text_length_is_within_bounds(corpus: tuple[Prompt, ...]) -> None:
    for prompt in corpus:
        assert MIN_TEXT_CHARS <= len(prompt.text) <= MAX_TEXT_CHARS, (prompt.id, len(prompt.text))


def test_prompts_are_immutable(corpus: tuple[Prompt, ...]) -> None:
    with pytest.raises(AttributeError):
        corpus[0].text = "changed"  # type: ignore[misc]


# --- the split --------------------------------------------------------------


def test_heldout_fraction_per_tag_is_between_15_and_35_percent(
    corpus: tuple[Prompt, ...],
) -> None:
    for tag in WORKLOAD_TAGS:
        prompts = by_tag(tag, corpus)
        heldout = sum(1 for p in prompts if p.split == "heldout")
        fraction = heldout / len(prompts)
        assert 0.15 <= fraction <= 0.35, f"{tag}: {heldout}/{len(prompts)} held out"


def test_split_is_a_pure_function_of_the_id(corpus: tuple[Prompt, ...]) -> None:
    for prompt in corpus:
        assert prompt.split == split_for(prompt.id)
        assert split_for(prompt.id) == split_for(prompt.id)


def test_split_is_deterministic_across_loads() -> None:
    first = {p.id: p.split for p in load_corpus()}
    second = {p.id: p.split for p in load_corpus()}

    assert first == second


def test_split_for_uses_the_first_sha256_byte_below_64() -> None:
    # sha256("code-004")[0] < 64, sha256("code-001")[0] >= 64 — fixed forever.
    assert split_for("code-004") == "heldout"
    assert split_for("code-001") == "train"


def test_split_selects_only_that_split(corpus: tuple[Prompt, ...]) -> None:
    train = split("train", corpus)
    heldout = split("heldout", corpus)

    assert all(p.split == "train" for p in train)
    assert all(p.split == "heldout" for p in heldout)
    assert len(train) + len(heldout) == len(corpus)


def test_split_rejects_unknown_name(corpus: tuple[Prompt, ...]) -> None:
    with pytest.raises(ValueError, match="split"):
        split("validation", corpus)  # type: ignore[arg-type]


def test_by_tag_selects_only_that_tag_and_rejects_unknown(corpus: tuple[Prompt, ...]) -> None:
    assert all(p.tag == "json" for p in by_tag("json", corpus))
    with pytest.raises(ValueError, match="tag"):
        by_tag("poetry", corpus)


# --- mixed workload ---------------------------------------------------------


def test_mixed_workload_is_deterministic_for_a_seed(corpus: tuple[Prompt, ...]) -> None:
    first = mixed_workload(seed=7, n=30, split="heldout", corpus=corpus)
    second = mixed_workload(seed=7, n=30, split="heldout", corpus=corpus)

    assert [p.id for p in first] == [p.id for p in second]


def test_mixed_workload_differs_across_seeds(corpus: tuple[Prompt, ...]) -> None:
    a = mixed_workload(seed=1, n=30, split="heldout", corpus=corpus)
    b = mixed_workload(seed=2, n=30, split="heldout", corpus=corpus)

    assert [p.id for p in a] != [p.id for p in b]


def test_mixed_workload_is_balanced_across_tags(corpus: tuple[Prompt, ...]) -> None:
    picked = mixed_workload(seed=3, n=25, split="train", corpus=corpus)
    counts = Counter(p.tag for p in picked)

    assert len(picked) == 25
    assert set(counts) == set(WORKLOAD_TAGS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_mixed_workload_draws_only_from_the_requested_split(corpus: tuple[Prompt, ...]) -> None:
    picked = mixed_workload(seed=5, n=60, split="heldout", corpus=corpus)

    assert all(p.split == "heldout" for p in picked)


def test_mixed_workload_rejects_negative_n(corpus: tuple[Prompt, ...]) -> None:
    with pytest.raises(ValueError, match="n"):
        mixed_workload(seed=0, n=-1, split="train", corpus=corpus)


# --- chat messages ----------------------------------------------------------


def test_as_chat_messages_wraps_text_as_a_single_user_turn() -> None:
    prompt = Prompt(id="chat-001", tag="chat", text="hello there", split="train")

    assert as_chat_messages(prompt) == [{"role": "user", "content": "hello there"}]


def test_prompt_rejects_unknown_tag_or_split() -> None:
    with pytest.raises(ValueError, match="tag"):
        Prompt(id="poetry-001", tag="poetry", text="x", split="train")
    with pytest.raises(ValueError, match="split"):
        Prompt(id="chat-001", tag="chat", text="x", split="validation")  # type: ignore[arg-type]


# --- boundary validation ----------------------------------------------------

GOOD_TEXT = "x" * MIN_TEXT_CHARS


def _reader_for(tmp_path: Path, lines: list[str], tag: str = "chat"):
    """Serve `lines` as `tag`'s file and an empty file for every other tag."""
    path = tmp_path / f"{tag}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return lambda requested: path.read_text() if requested == tag else ""


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["not json"], "line 1"),
        ([json.dumps({"id": "chat-1", "text": GOOD_TEXT})], "chat-1"),
        ([json.dumps({"id": "code-001", "text": GOOD_TEXT})], "code-001"),
        ([json.dumps({"id": "chat-001", "text": "short"})], "chat-001"),
        ([json.dumps({"id": "chat-001", "text": "y" * (MAX_TEXT_CHARS + 1)})], "chat-001"),
        ([json.dumps({"id": "chat-001"})], "text"),
        ([json.dumps({"id": "chat-001", "text": GOOD_TEXT, "extra": 1})], "extra"),
        (
            [
                json.dumps({"id": "chat-001", "text": GOOD_TEXT}),
                json.dumps({"id": "chat-001", "text": GOOD_TEXT}),
            ],
            "duplicate",
        ),
    ],
    ids=["bad-json", "bad-id", "tag-mismatch", "too-short", "too-long", "no-text", "extra", "dup"],
)
def test_malformed_record_raises_a_clear_error(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    read_tag = _reader_for(tmp_path, lines)

    with pytest.raises(CorpusError, match=message):
        load_corpus(read_tag=read_tag)


def test_injected_reader_is_used_for_every_tag(tmp_path: Path) -> None:
    def read_tag(tag: str) -> str:
        return json.dumps({"id": f"{tag}-001", "text": GOOD_TEXT}) + "\n"

    corpus = load_corpus(read_tag=read_tag)

    assert [p.id for p in corpus] == [f"{tag}-001" for tag in WORKLOAD_TAGS]
