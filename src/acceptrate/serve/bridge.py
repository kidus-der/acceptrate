"""Bridge a synchronous engine iterator into an async response.

The engine is synchronous and must run off the event loop, so a worker
thread pumps items into an asyncio.Queue via `call_soon_threadsafe`. When
the consumer stops early (client disconnected), the pump sees the stop flag
after the current item, closes the iterator (which releases the Session
lock through its `finally`) and exits. That release lands one window late;
a request arriving in that gap gets a 429 and retries.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

log = logging.getLogger(__name__)


class _Done:
    pass


@dataclass(frozen=True)
class _Failed:
    exc: BaseException


_DONE = _Done()

_background: set[asyncio.Task[None]] = set()
"""Keeps pump tasks alive until they finish, even if the consumer is gone."""


def _close_quietly[T](iterator: Iterator[T]) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


async def iterate_in_thread[T](iterator: Iterator[T]) -> AsyncIterator[T]:
    """Yield items of a blocking iterator from a worker thread, one at a time."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[T | _Done | _Failed] = asyncio.Queue()
    stop = threading.Event()

    def push(item: T | _Done | _Failed) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # loop closed: the server is shutting down
            log.debug("dropped an item after the event loop closed")

    def pump() -> None:
        try:
            for item in iterator:
                push(item)
                if stop.is_set():
                    break
        except BaseException as exc:  # forwarded to the consumer, never swallowed
            push(_Failed(exc))
        else:
            push(_DONE)
        finally:
            _close_quietly(iterator)

    task = asyncio.create_task(asyncio.to_thread(pump))
    _background.add(task)
    task.add_done_callback(_background.discard)
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, _Failed):
                raise item.exc
            yield item
    finally:
        stop.set()
