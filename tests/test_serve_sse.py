"""serve/sse.py — frame format, and the stats feed pushing per window during a live generation."""

from __future__ import annotations

import asyncio
import json
import threading

from acceptrate.serve.session import Session
from acceptrate.serve.sse import sse, stats_events
from tests.fakes import FakeBackend, FakeTokenizer
from tests.serve_fakes import HoldableBackend

MESSAGES = ({"role": "user", "content": "hi"},)


def test_sse_frame_format() -> None:
    assert sse('{"a":1}') == 'data: {"a":1}\n\n'
    assert sse("x", "stats") == "event: stats\ndata: x\n\n"


def _split(frames: list[str]) -> list[tuple[str, dict]]:
    out = []
    for frame in frames:
        name, data = frame.rstrip("\n").split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


async def _consume(session: Session, limit: int) -> list[str]:
    return [frame async for frame in stats_events(session, heartbeat_s=0.05, limit=limit)]


def test_stats_feed_pushes_stats_events_while_a_generation_runs() -> None:
    backend = HoldableBackend()
    session = Session(backend, FakeBackend(step=3), FakeTokenizer(), lambda: 2, model="m")
    backend.gate.clear()
    events = session.generate(MESSAGES, max_tokens=12)
    worker = threading.Thread(target=lambda: list(events), daemon=True)
    worker.start()
    assert backend.entered.wait(2.0)

    first = _split(asyncio.run(_consume(session, limit=1)))
    backend.gate.set()
    worker.join(2.0)
    assert not worker.is_alive()
    later = _split(asyncio.run(_consume(session, limit=2)))

    assert first[0][0] == "stats" and first[0][1]["busy"] is True
    assert later[0][0] == "stats" and later[0][1]["busy"] is False
    assert later[0][1]["windows_total"] == 11
    assert later[1][0] == "heartbeat"


def test_stats_feed_wakes_on_a_window_instead_of_waiting_for_the_heartbeat() -> None:
    backend = HoldableBackend()
    session = Session(backend, None, FakeTokenizer(), lambda: 0, model="m")
    backend.gate.clear()
    events = session.generate(MESSAGES, max_tokens=6)
    worker = threading.Thread(target=lambda: list(events), daemon=True)
    worker.start()
    assert backend.entered.wait(2.0)

    async def run() -> list[str]:
        loop = asyncio.get_running_loop()
        loop.call_later(0.01, backend.gate.set)
        return [frame async for frame in stats_events(session, heartbeat_s=5.0, limit=2)]

    frames = _split(asyncio.run(run()))
    worker.join(2.0)
    assert not worker.is_alive()

    assert [name for name, _ in frames] == ["stats", "stats"]
    assert frames[1][1]["windows_total"] >= 1
