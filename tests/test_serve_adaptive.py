"""Session with a scheduler factory: K is re-picked per window and k_current tracks it."""

from __future__ import annotations

from acceptrate.serve.session import Session
from tests.fakes import FakeBackend, FakeTokenizer


class PartialFake(FakeBackend):
    def next_token(self, token: int) -> int:
        wrong = 1 if token % 3 == 0 else 0
        return (token * 7 + wrong) % self.vocab_size


class Cycling:
    def __init__(self, pattern: tuple[int, ...]) -> None:
        self.pattern, self.i = pattern, 0

    def next_k(self) -> int:
        k = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return k

    def observe(self, *_) -> None:
        return None


def _session(pattern: tuple[int, ...]) -> Session:
    return Session(
        FakeBackend(32, step=7),
        PartialFake(32),
        FakeTokenizer(),
        lambda: 4,
        model="fake",
        draft_name="fake-draft",
        make_scheduler=lambda: Cycling(pattern),
    )


def test_adaptive_session_streams_and_reports_the_per_window_k() -> None:
    session = _session((2, 4, 6))

    events = list(session.generate([{"role": "user", "content": "hello there"}], max_tokens=60))

    stats = session.stats()
    assert events[-1].finish_reason == "length"
    ks = {w.k_proposed for w in stats.last_windows}
    assert ks == {2, 4, 6}
    assert stats.k_current == stats.last_windows[-1].k_proposed


def test_adaptive_session_matches_fixed_session_text_when_scheduler_is_constant() -> None:
    fixed = Session(FakeBackend(32, step=7), PartialFake(32), FakeTokenizer(), lambda: 4, model="f")
    adaptive = _session((4,))
    msgs = [{"role": "user", "content": "abc"}]

    a = "".join(e.text for e in fixed.generate(msgs, 40))
    b = "".join(e.text for e in adaptive.generate(msgs, 40))

    assert a == b
