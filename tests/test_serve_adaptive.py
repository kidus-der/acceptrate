"""Session with a scheduler factory: K is re-picked per window and k_current tracks it."""

from __future__ import annotations

from acceptrate.runtime.adaptive import AdaptiveScheduler, SchedulerConfig
from acceptrate.serve.session import Session
from tests.fakes import FakeBackend, FakeTokenizer


class PartialFake(FakeBackend):
    def next_token(self, token: int) -> int:
        wrong = 1 if token % 3 == 0 else 0
        return (token * 7 + wrong) % self.vocab_size


def _session(prior: float = 0.6) -> Session:
    cfg = SchedulerConfig(
        k_min=1, k_max=6, prior_alpha=prior, prior_c=0.16, half_life_windows=2, warmup_windows=0
    )
    return Session(
        FakeBackend(32, step=7),
        PartialFake(32),
        FakeTokenizer(),
        lambda: 4,
        model="fake",
        draft_name="fake-draft",
        make_scheduler=lambda: AdaptiveScheduler(cfg),
    )


def test_adaptive_session_streams_and_varies_k() -> None:
    session = _session()

    events = list(session.generate([{"role": "user", "content": "hello there"}], max_tokens=60))

    stats = session.stats()
    assert events[-1].finish_reason == "length"
    ks = {w.k_proposed for w in stats.last_windows}
    assert len(ks) > 1
    assert stats.k_current in ks
    assert stats.k_current >= 1


def test_adaptive_session_matches_fixed_session_text_when_scheduler_is_constant() -> None:
    class Constant:
        def next_k(self):
            return 4

        def observe(self, *_):
            return None

    fixed = Session(FakeBackend(32, step=7), PartialFake(32), FakeTokenizer(), lambda: 4, model="f")
    adaptive = Session(
        FakeBackend(32, step=7), PartialFake(32), FakeTokenizer(), lambda: 4, model="f",
        make_scheduler=lambda: Constant(),
    )  # fmt: skip
    msgs = [{"role": "user", "content": "abc"}]

    a = "".join(e.text for e in fixed.generate(msgs, 40))
    b = "".join(e.text for e in adaptive.generate(msgs, 40))

    assert a == b
