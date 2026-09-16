"""Server-sent event streams: OpenAI chat chunks and the TUI's stats feed."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterator

from acceptrate.serve.bridge import iterate_in_thread
from acceptrate.serve.schemas import (
    ChatCompletionChunk,
    ChunkChoice,
    Delta,
    ErrorBody,
    ErrorResponse,
)
from acceptrate.serve.session import Event, Session

HEARTBEAT_S = 1.0
"""Idle interval between stats pushes on /stats/stream."""

DONE_LINE = "data: [DONE]\n\n"
STATS_EVENT = "stats"
HEARTBEAT_EVENT = "heartbeat"

log = logging.getLogger(__name__)


def sse(data: str, event: str | None = None) -> str:
    """One SSE frame. OpenAI chunks carry no event name; the stats feed does."""
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {data}\n\n"


def _chunk(
    completion_id: str, created: int, model: str, delta: Delta, finish: str | None = None
) -> str:
    chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=delta, finish_reason=finish)],
    )
    # OpenAI keeps "finish_reason": null on every chunk but omits unset delta fields.
    choice = chunk.choices[0].model_dump()
    payload = {
        **chunk.model_dump(),
        "choices": [{**choice, "delta": delta.model_dump(exclude_none=True)}],
    }
    return sse(json.dumps(payload))


def _error_frame(message: str) -> str:
    body = ErrorResponse(error=ErrorBody(message=message, type="server_error", code=500))
    return sse(body.model_dump_json())


async def chat_chunks(
    events: Iterator[Event], completion_id: str, created: int, model: str
) -> AsyncIterator[str]:
    """OpenAI streaming: a role chunk, content deltas, a finish chunk, then [DONE]."""
    yield _chunk(completion_id, created, model, Delta(role="assistant", content=""))
    try:
        async for event in iterate_in_thread(events):
            if event.text:
                yield _chunk(completion_id, created, model, Delta(content=event.text))
            if event.finish_reason is not None:
                yield _chunk(completion_id, created, model, Delta(), event.finish_reason)
    except Exception:
        log.exception("generation failed mid-stream (completion %s)", completion_id)
        yield _error_frame("generation failed; see server log")
    yield DONE_LINE


async def stats_events(
    session: Session, heartbeat_s: float = HEARTBEAT_S, limit: int | None = None
) -> AsyncIterator[str]:
    """Current stats immediately, then after every window, else a heartbeat every heartbeat_s."""
    version = session.version
    sent = 0
    yield sse(session.stats().model_dump_json(), STATS_EVENT)
    sent += 1
    while limit is None or sent < limit:
        latest = await asyncio.to_thread(session.wait_for_change, version, heartbeat_s)
        name = STATS_EVENT if latest != version else HEARTBEAT_EVENT
        version = latest
        yield sse(session.stats().model_dump_json(), name)
        sent += 1
