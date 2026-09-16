"""The prompt corpus, tagged by task type (one of WORKLOAD_TAGS).

Every sweep prompt lives here as package data: `corpus/<tag>.jsonl`, one
`{"id": "<tag>-NNN", "text": "..."}` record per line. All text is original.

THE SPLIT IS FIXED BEFORE ANY MEASUREMENT EXISTS. A prompt is held out iff the
first byte of sha256(id) is below HELDOUT_BYTE_THRESHOLD (~25%). It depends on
nothing but the id, so no result, model, or config can move a prompt between
splits. P5's adaptive runtime is judged only on `split == "heldout"`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from acceptrate.trace.schema import WORKLOAD_TAGS

Split = Literal["train", "heldout"]
SPLITS: tuple[Split, ...] = ("train", "heldout")

MIN_PROMPTS_PER_TAG = 24
MIN_TEXT_CHARS = 60
MAX_TEXT_CHARS = 3000
"""Upper bound is set by `summarize`, whose prompts carry a 150-400 word passage."""

HELDOUT_BYTE_THRESHOLD = 64
"""First sha256 byte below this is held out: 64/256 = 25% in expectation."""

ID_PATTERN = rf"^({'|'.join(WORKLOAD_TAGS)})-\d{{3}}$"
CORPUS_PACKAGE = "acceptrate.bench.corpus"

ReadTag = Callable[[str], str]
"""Returns the raw JSONL text for one tag. Injectable so tests can feed bad records."""


class CorpusError(ValueError):
    """A corpus file failed validation at the boundary. The message names the record."""


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str
    tag: str
    text: str
    split: Split

    def __post_init__(self) -> None:
        if self.tag not in WORKLOAD_TAGS:
            raise ValueError(f"unknown tag {self.tag!r}; expected one of {WORKLOAD_TAGS}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}; expected one of {SPLITS}")


class _Record(BaseModel):
    """Shape of one JSONL line. Pydantic does the field checks; the loader does cross-record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    text: str = Field(min_length=MIN_TEXT_CHARS, max_length=MAX_TEXT_CHARS)


def split_for(prompt_id: str) -> Split:
    """The split is a pure function of the id. See the module docstring."""
    first_byte = hashlib.sha256(prompt_id.encode("utf-8")).digest()[0]
    return "heldout" if first_byte < HELDOUT_BYTE_THRESHOLD else "train"


def _read_packaged(tag: str) -> str:
    return resources.files(CORPUS_PACKAGE).joinpath(f"{tag}.jsonl").read_text(encoding="utf-8")


def _parse_line(tag: str, line_no: int, line: str) -> _Record:
    where = f"{tag}.jsonl line {line_no}"
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{where}: not valid JSON ({exc.msg})") from exc
    try:
        record = _Record.model_validate(raw)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'record'}: {e['msg']}" for e in exc.errors()
        )
        raise CorpusError(f"{where}: invalid record {raw!r}: {detail}") from exc
    if not record.id.startswith(f"{tag}-"):
        raise CorpusError(f"{where}: id {record.id!r} does not belong to tag {tag!r}")
    return record


def _load_tag(tag: str, read_tag: ReadTag) -> tuple[Prompt, ...]:
    lines = read_tag(tag).splitlines()
    records = [_parse_line(tag, i, line) for i, line in enumerate(lines, start=1) if line.strip()]
    return tuple(Prompt(id=r.id, tag=tag, text=r.text, split=split_for(r.id)) for r in records)


def _check_unique_ids(prompts: Sequence[Prompt]) -> None:
    seen: set[str] = set()
    for prompt in prompts:
        if prompt.id in seen:
            raise CorpusError(f"duplicate prompt id {prompt.id!r}")
        seen.add(prompt.id)


def load_corpus(read_tag: ReadTag = _read_packaged) -> tuple[Prompt, ...]:
    """Load every tag's file, validating each record at the boundary. Order: tag, then file."""
    prompts = tuple(
        itertools.chain.from_iterable(_load_tag(tag, read_tag) for tag in WORKLOAD_TAGS)
    )
    _check_unique_ids(prompts)
    return prompts


def _resolve(corpus: Sequence[Prompt] | None) -> Sequence[Prompt]:
    """Callers may pass an already-loaded corpus to avoid re-reading package data."""
    return load_corpus() if corpus is None else corpus


def _check_split(name: str) -> None:
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")


def by_tag(tag: str, corpus: Sequence[Prompt] | None = None) -> tuple[Prompt, ...]:
    if tag not in WORKLOAD_TAGS:
        raise ValueError(f"unknown tag {tag!r}; expected one of {WORKLOAD_TAGS}")
    return tuple(p for p in _resolve(corpus) if p.tag == tag)


def split(name: Split, corpus: Sequence[Prompt] | None = None) -> tuple[Prompt, ...]:
    _check_split(name)
    return tuple(p for p in _resolve(corpus) if p.split == name)


def _shuffled_pool(prompts: Sequence[Prompt], rng: random.Random) -> list[Prompt]:
    ordered = sorted(prompts, key=lambda p: p.id)  # independent of file order
    rng.shuffle(ordered)
    return ordered


def mixed_workload(
    seed: int, n: int, split: Split, corpus: Sequence[Prompt] | None = None
) -> tuple[Prompt, ...]:
    """`n` prompts from `split`, round-robin over per-tag seeded shuffles.

    Deterministic for a seed; tag counts differ by at most one. Pools cycle when
    `n` exceeds what the split holds, so held-out evaluation can run longer than
    the held-out set (repeats are distinguishable by `rep` in the trace).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    _check_split(split)
    prompts = _resolve(corpus)
    rng = random.Random(seed)
    pools = {
        tag: _shuffled_pool([p for p in prompts if p.tag == tag and p.split == split], rng)
        for tag in WORKLOAD_TAGS
    }
    empty = [tag for tag, pool in pools.items() if not pool]
    if empty:
        raise ValueError(f"split {split!r} has no prompts for tags {empty}")
    cycles = {tag: itertools.cycle(pool) for tag, pool in pools.items()}
    return tuple(next(cycles[tag]) for tag in itertools.islice(itertools.cycle(WORKLOAD_TAGS), n))


def as_chat_messages(prompt: Prompt) -> list[dict[str, str]]:
    """One user turn. The backend's tokenizer applies the chat template later."""
    return [{"role": "user", "content": prompt.text}]
