"""serve/bridge.py — a blocking iterator pumped into an async consumer from a worker thread."""

from __future__ import annotations

import asyncio
import threading

import pytest

from acceptrate.serve.bridge import iterate_in_thread

WAIT_S = 2.0


async def _collect(iterator, stop_after: int | None = None) -> list:
    out = []
    async for item in iterate_in_thread(iterator):
        out.append(item)
        if stop_after is not None and len(out) >= stop_after:
            break
    return out


def test_yields_every_item_in_order() -> None:
    assert asyncio.run(_collect(iter(range(5)))) == [0, 1, 2, 3, 4]


def test_an_exception_in_the_iterator_reaches_the_consumer() -> None:
    def broken():
        yield 1
        raise RuntimeError("engine blew up")

    with pytest.raises(RuntimeError, match="engine blew up"):
        asyncio.run(_collect(broken()))


def test_stopping_early_closes_the_generator_so_its_finally_runs() -> None:
    closed = threading.Event()

    def endless():
        try:
            n = 0
            while True:
                yield n
                n += 1
        finally:
            closed.set()

    got = asyncio.run(_collect(endless(), stop_after=3))

    assert got == [0, 1, 2]
    assert closed.wait(WAIT_S)
