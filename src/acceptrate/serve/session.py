"""The Session: one generation at a time, streamed as events, with live stats.

Why exactly one at a time: there is one GPU, and the trace rows /stats is
built from are timed per window. A second interleaved generation would share
the device and turn draft_ms / verify_ms into queueing noise — the stats
would lie. So `generate` takes a non-blocking lock and raises `SessionBusyError`
instead of queueing; the HTTP layer turns that into a 429.

K comes from `choose_k`, consulted once per generation. P5 plugs the adaptive
scheduler in here; a per-window re-pick will need a streaming variant that
takes the callable, which is a change to runtime/streaming.py, not to this
class's contract.
"""

from __future__ import annotations

import threading
import weakref
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from acceptrate.backend.protocol import Backend, ChatMessage, Tokenizer
from acceptrate.runtime.adaptive import Scheduler
from acceptrate.runtime.engine import GenerationContext, GuardReader
from acceptrate.runtime.streaming import (
    StreamEvent,
    generate_adaptive_streaming,
    generate_plain_streaming,
    generate_speculative_streaming,
)
from acceptrate.serve.detok import Detok, flush, step
from acceptrate.serve.stats import Stats, Totals, WindowStat, compute_stats, window_stat_from_row
from acceptrate.trace.schema import WindowRow

ROLLING_WINDOW = 256
"""Draft windows kept for alpha_ewma / tok_s_recent."""

RUN_ID = "serve"
WORKLOAD_TAG = "chat"

FINISH_STOP = "stop"
FINISH_LENGTH = "length"


class SessionBusyError(RuntimeError):
    """A generation is already running; the caller should retry later, not queue."""


@dataclass(frozen=True)
class Event:
    text: str
    """Detokenised delta safe to show now (may be empty mid multibyte character)."""
    tokens: tuple[int, ...]
    """Tokens committed by this window, EOS included if it was hit."""
    finish_reason: str | None = None
    """Set on the final event only: "stop" (EOS) or "length" (max_tokens)."""


class _Generation:
    """Iterator over a generation that releases the session on close() or GC.

    An unstarted generator's finally never runs, so the release cannot live
    only inside `_run`; this wrapper calls it explicitly on close() and via
    weakref.finalize when the object is dropped. `release` is one-shot.
    """

    __slots__ = ("__weakref__", "_inner", "_release")

    def __init__(self, inner: Iterator[Event], release: Callable[[], None]) -> None:
        self._inner = inner
        self._release = release
        weakref.finalize(self, release)

    def __iter__(self) -> Iterator[Event]:
        return self

    def __next__(self) -> Event:
        return next(self._inner)

    def close(self) -> None:
        self._inner.close()
        self._release()


class _NullSnapshot:
    mem_pressure = 0
    page_ins = 0
    thermal_level = 0


_NULL_SNAPSHOT = _NullSnapshot()


def null_guard() -> _NullSnapshot:
    """Guard reader for serving without the bench guard thread."""
    return _NULL_SNAPSHOT


class Session:
    def __init__(
        self,
        target: Backend,
        draft: Backend | None,
        tokenizer: Tokenizer,
        choose_k: Callable[[], int],
        *,
        model: str,
        draft_name: str | None = None,
        guard: GuardReader = null_guard,
        window_size: int = ROLLING_WINDOW,
        make_scheduler: Callable[[], Scheduler] | None = None,
        v_by_k: Mapping[int, float] | None = None,
    ) -> None:
        self._target = target
        self._draft = draft
        self._tokenizer = tokenizer
        self._choose_k = choose_k
        self._make_scheduler = make_scheduler
        self._v_by_k = dict(v_by_k) if v_by_k else None
        self._guard = guard
        self.model = model
        self.draft_name = draft_name
        self._running = threading.Lock()
        self._state = threading.Condition()
        self._window: deque[WindowStat] = deque(maxlen=window_size)
        self._totals = Totals(0, 0, 0)
        self._version = 0
        self._k_current = 0
        self._requests = 0

    @property
    def busy(self) -> bool:
        return self._running.locked()

    @property
    def version(self) -> int:
        """Bumps on every window and on generation start/end; drives /stats/stream."""
        with self._state:
            return self._version

    @property
    def k_current(self) -> int:
        with self._state:
            return self._k_current

    def generate(self, messages: Sequence[ChatMessage], max_tokens: int) -> Iterator[Event]:
        """Start a generation now; raises SessionBusyError if one is running.

        The returned iterator holds the session until it is exhausted or closed.
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not self._running.acquire(blocking=False):
            raise SessionBusyError("a generation is already running; retry when it finishes")
        try:
            k = self._choose_k()
            prompt = self._tokenizer.encode_chat(messages)
            self._begin(k)
        except BaseException:
            self._running.release()
            raise
        release = self._release_once()
        return _Generation(self._run(prompt, max_tokens, k, release), release)

    def _release_once(self) -> Callable[[], None]:
        """One-shot end-of-generation: safe from the generator's finally and from GC."""
        once = threading.Lock()

        def release() -> None:
            if not once.acquire(blocking=False):
                return
            self._end()
            self._running.release()

        return release

    def _run(
        self, prompt: list[int], max_tokens: int, k: int, release: Callable[[], None]
    ) -> Iterator[Event]:
        eos = self._tokenizer.eos_token_ids
        decode = self._tokenizer.decode
        try:
            detok = Detok()
            last_token: int | None = None
            for tokens, row in self._engine_stream(prompt, max_tokens, k):
                if row is not None:
                    self._record(row, len(tokens))
                last_token = tokens[-1]
                visible = tokens[:-1] if last_token in eos else tokens
                detok, text = step(detok, visible, decode)
                yield Event(text, tokens)
            finish = FINISH_STOP if last_token in eos else FINISH_LENGTH
            yield Event(flush(detok, decode), (), finish)
        finally:
            release()

    def _engine_stream(self, prompt: list[int], max_tokens: int, k: int) -> Iterator[StreamEvent]:
        eos = self._tokenizer.eos_token_ids
        ctx = GenerationContext(RUN_ID, WORKLOAD_TAG, str(self._requests), 0)
        if k <= 0 or self._draft is None:
            return generate_plain_streaming(self._target, prompt, max_tokens, eos, ctx, self._guard)
        if self._make_scheduler is not None:
            return generate_adaptive_streaming(
                self._target, self._draft, prompt, max_tokens, eos, ctx, self._guard,
                self._make_scheduler(),
            )  # fmt: skip
        return generate_speculative_streaming(
            self._target, self._draft, prompt, max_tokens, eos, ctx, self._guard, k
        )

    def prompt_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """Prompt length after the chat template, for the OpenAI `usage` block."""
        return len(self._tokenizer.encode_chat(messages))

    def stats(self) -> Stats:
        with self._state:
            window = tuple(self._window)
            totals = self._totals
            k = self._k_current
        return compute_stats(
            window,
            totals,
            model=self.model,
            draft=self.draft_name,
            busy=self.busy,
            k_current=k,
            v_by_k=self._v_by_k,
        )

    def wait_for_change(self, version: int, timeout: float) -> int:
        """Block until `version` is stale or `timeout` elapses; return the current version."""
        with self._state:
            self._state.wait_for(lambda: self._version != version, timeout)
            return self._version

    def _begin(self, k: int) -> None:
        with self._state:
            self._k_current = k
            self._requests += 1
            self._bump()

    def _end(self) -> None:
        with self._state:
            self._bump()

    def _record(self, row: WindowRow, n_committed: int) -> None:
        stat = window_stat_from_row(row, n_committed)
        with self._state:
            self._k_current = stat.k_proposed
            self._window.append(stat)
            self._totals = Totals(
                self._totals.windows + 1,
                self._totals.accepted + stat.n_accepted,
                self._totals.proposed + stat.k_proposed,
            )
            self._bump()

    def _bump(self) -> None:
        self._version += 1
        self._state.notify_all()
