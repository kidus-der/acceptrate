"""serve/session.py — one generation at a time, streamed as events, with live stats.

The Session is the single object `acceptrate serve` wraps: it owns the
backends, picks K through `choose_k` (the P5 seam), streams committed tokens
and detokenised text, and keeps the rolling window /stats reads from.
"""

from __future__ import annotations

import gc
import threading

import pytest

from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.serve.session import Event, Session, SessionBusyError
from tests.fakes import FakeBackend, FakeTokenizer
from tests.serve_fakes import EosTokenizer, HoldableBackend

V = 32
MESSAGES = ({"role": "user", "content": "hi"},)
CTX = GenerationContext(run_id="x", workload_tag="chat", prompt_id="x", rep=0)


class _Snap:
    mem_pressure = 0
    page_ins = 0
    thermal_level = 0


def _plain_tokens(tokenizer, max_tokens):
    prompt = tokenizer.encode_chat(MESSAGES)
    return list(
        generate_plain(
            FakeBackend(V), prompt, max_tokens, tokenizer.eos_token_ids, CTX, _Snap
        ).tokens
    )


def _session(k: int, draft: FakeBackend | None = None, tokenizer=None, **kwargs) -> Session:
    return Session(
        FakeBackend(V),
        draft,
        tokenizer or FakeTokenizer(),
        choose_k=lambda: k,
        model="target-fake",
        draft_name=None if draft is None else "draft-fake",
        **kwargs,
    )


def _drain(events) -> tuple[list[int], str, Event]:
    tokens: list[int] = []
    text = ""
    last: Event | None = None
    for event in events:
        tokens.extend(event.tokens)
        text += event.text
        last = event
    assert last is not None
    return tokens, text, last


def test_plain_generation_streams_the_engine_tokens_and_their_text() -> None:
    tok = FakeTokenizer()

    tokens, text, last = _drain(_session(k=0).generate(MESSAGES, max_tokens=12))

    assert tokens == _plain_tokens(tok, 12)
    assert text == tok.decode(tokens)
    assert last.finish_reason == "length"


def test_speculative_generation_is_lossless_and_records_windows() -> None:
    session = _session(k=3, draft=FakeBackend(V, step=3))

    tokens, text, _ = _drain(session.generate(MESSAGES, max_tokens=16))
    stats = session.stats()

    assert tokens == _plain_tokens(FakeTokenizer(), 16)
    assert text == FakeTokenizer().decode(tokens)
    assert stats.windows_total == 15  # every window commits exactly one token with a bad draft
    assert stats.proposed_total == 45
    assert stats.accepted_total == 0
    assert stats.k_current == 3
    assert stats.model == "target-fake"
    assert stats.draft == "draft-fake"
    assert stats.busy is False


def test_k_zero_with_a_draft_falls_back_to_plain_decoding() -> None:
    draft = FakeBackend(V)
    session = _session(k=0, draft=draft)

    _drain(session.generate(MESSAGES, max_tokens=8))

    assert draft.calls == []
    assert session.stats().proposed_total == 0


def test_eos_ends_the_generation_with_stop_and_is_not_decoded_into_text() -> None:
    plain = _plain_tokens(FakeTokenizer(), 30)
    eos_token = plain[4]
    first_hit = plain.index(eos_token)  # the fake's rule cycles, so it may appear earlier
    tok = EosTokenizer(frozenset({eos_token}))

    tokens, text, last = _drain(
        _session(k=2, draft=FakeBackend(V), tokenizer=tok).generate(MESSAGES, 30)
    )

    assert tokens == plain[: first_hit + 1]
    assert text == tok.decode(plain[:first_hit])
    assert last.finish_reason == "stop"


def test_choose_k_is_consulted_once_per_generation() -> None:
    calls: list[int] = []

    def choose_k() -> int:
        calls.append(1)
        return 2

    session = Session(FakeBackend(V), FakeBackend(V), FakeTokenizer(), choose_k, model="m")
    _drain(session.generate(MESSAGES, 10))
    _drain(session.generate(MESSAGES, 10))

    assert len(calls) == 2


def test_a_second_generation_is_refused_while_one_is_running() -> None:
    backend = HoldableBackend(V)
    session = Session(backend, None, FakeTokenizer(), choose_k=lambda: 0, model="m")
    backend.gate.clear()
    first = session.generate(MESSAGES, 10)
    worker = threading.Thread(target=lambda: _drain(first), daemon=True)
    worker.start()
    assert backend.entered.wait(2.0)

    assert session.busy is True
    with pytest.raises(SessionBusyError):
        session.generate(MESSAGES, 10)

    backend.gate.set()
    worker.join(2.0)
    assert not worker.is_alive()
    assert session.busy is False
    _drain(session.generate(MESSAGES, 10))  # the lock was released


def test_closing_an_unfinished_generation_releases_the_session() -> None:
    session = _session(k=0)
    events = session.generate(MESSAGES, 10)
    next(events)
    assert session.busy is True

    events.close()

    assert session.busy is False


def test_closing_a_never_started_generation_releases_the_session() -> None:
    session = _session(k=0)
    events = session.generate(MESSAGES, 10)
    assert session.busy is True

    events.close()  # an unstarted generator's finally never runs; release must not rely on it

    assert session.busy is False
    _drain(session.generate(MESSAGES, 5))


def test_dropping_a_never_started_generation_releases_the_session() -> None:
    session = _session(k=0)
    events = session.generate(MESSAGES, 10)
    assert session.busy is True

    del events
    gc.collect()

    assert session.busy is False


def test_version_bumps_per_window_and_wait_for_change_wakes_or_times_out() -> None:
    session = _session(k=0)
    before = session.version

    _drain(session.generate(MESSAGES, 6))

    assert session.version >= before + 5  # at least one bump per decode window
    assert session.wait_for_change(before, timeout=0.01) == session.version
    assert session.wait_for_change(session.version, timeout=0.01) == session.version


def test_rolling_window_is_bounded_while_totals_keep_counting() -> None:
    session = _session(k=0, window_size=4)

    _drain(session.generate(MESSAGES, 20))
    stats = session.stats()

    assert stats.windows_total == 19
    assert len(stats.last_windows) == 4
    assert stats.tok_s_recent is not None and stats.tok_s_recent > 0
    assert stats.alpha_ewma is None  # plain windows carry no acceptance signal


def test_rejects_non_positive_max_tokens_before_touching_the_backend() -> None:
    session = _session(k=0)

    with pytest.raises(ValueError):
        session.generate(MESSAGES, max_tokens=0)
    assert session.busy is False
