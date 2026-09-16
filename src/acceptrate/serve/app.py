"""`create_app(session)`: the OpenAI-compatible endpoint plus the live stats feed.

Localhost only, no auth, no secrets: the runner binds 127.0.0.1 by default.
One generation at a time — a second request while one runs gets 429 with an
OpenAI-style error body rather than queueing (see serve/session.py for why).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from acceptrate.serve.schemas import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    ErrorBody,
    ErrorResponse,
    Health,
    Message,
    ModelCard,
    ModelList,
    Usage,
)
from acceptrate.serve.session import Event, Session, SessionBusyError
from acceptrate.serve.sse import HEARTBEAT_S, chat_chunks, stats_events
from acceptrate.serve.stats import Stats

SSE_MEDIA_TYPE = "text/event-stream"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

SAMPLING_NOT_IMPLEMENTED = (
    "temperature > 0 is not implemented yet: sampling arrives with the distributional "
    "losslessness test; send temperature 0 (greedy) for now"
)
BUSY_MESSAGE = (
    "a generation is already running on the single local GPU; requests are not queued "
    "because interleaving would corrupt the per-window timing stats — retry shortly"
)


def _error(status: int, kind: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(message=message, type=kind, code=status))
    return JSONResponse(body.model_dump(), status_code=status)


def _new_completion_id() -> str:
    return "chatcmpl-" + uuid4().hex[:24]


def _collect(events: Iterator[Event]) -> tuple[str, int, str]:
    """Drain a generation on the worker thread: (text, completion tokens, finish reason)."""
    parts: list[str] = []
    n_tokens = 0
    finish = "length"
    for event in events:
        parts.append(event.text)
        n_tokens += len(event.tokens)
        if event.finish_reason is not None:
            finish = event.finish_reason
    return "".join(parts), n_tokens, finish


async def _chat_completion(session: Session, req: ChatCompletionRequest):
    """Non-stream: drain on a worker thread. Stream: SSE chunks. Busy: SessionBusyError -> 429."""
    if req.temperature > 0:
        return _error(400, "invalid_request_error", SAMPLING_NOT_IMPLEMENTED)
    messages = tuple(m.model_dump() for m in req.messages)
    events = session.generate(messages, req.completion_budget)
    completion_id, created = _new_completion_id(), int(time.time())
    if req.stream:
        stream = chat_chunks(events, completion_id, created, session.model)
        return StreamingResponse(stream, media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)
    text, n_tokens, finish = await asyncio.to_thread(_collect, events)
    prompt_tokens = session.prompt_tokens(messages)
    return ChatCompletion(
        id=completion_id,
        created=created,
        model=session.model,
        choices=[Choice(message=Message(role="assistant", content=text), finish_reason=finish)],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=n_tokens,
            total_tokens=prompt_tokens + n_tokens,
        ),
    )


def create_app(session: Session, *, heartbeat_s: float = HEARTBEAT_S) -> FastAPI:
    app = FastAPI(title="acceptrate serve", docs_url=None, redoc_url=None)
    started = int(time.time())

    @app.exception_handler(SessionBusyError)
    async def _busy(_: Request, __: SessionBusyError) -> JSONResponse:
        return _error(429, "server_busy", BUSY_MESSAGE)

    @app.post("/v1/chat/completions", response_model=ChatCompletion)
    async def chat_completions(req: ChatCompletionRequest):
        return await _chat_completion(session, req)

    @app.get("/v1/models", response_model=ModelList)
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=session.model, created=started)])

    @app.get("/stats", response_model=Stats)
    async def stats() -> Stats:
        return session.stats()

    @app.get("/stats/stream")
    async def stats_stream(
        limit: int | None = Query(default=None, ge=1, description="stop after N events"),
    ) -> StreamingResponse:
        stream = stats_events(session, heartbeat_s, limit)
        return StreamingResponse(stream, media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)

    @app.get("/healthz", response_model=Health)
    async def healthz() -> Health:
        return Health(busy=session.busy)

    return app
