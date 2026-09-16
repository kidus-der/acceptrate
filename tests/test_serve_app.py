"""serve/app.py — the OpenAI-compatible endpoint and the stats feed, on fakes only.

No model is ever loaded here: FakeBackend + FakeTokenizer behind a Session,
FastAPI's TestClient in front. TestClient buffers streaming bodies, so SSE
tests read the whole response and split it into events.
"""

from __future__ import annotations

import json
import threading

import pytest
from fastapi.testclient import TestClient

from acceptrate.runtime.engine import GenerationContext, generate_plain
from acceptrate.serve.app import create_app
from acceptrate.serve.session import Session
from tests.fakes import FakeBackend, FakeTokenizer
from tests.serve_fakes import HoldableBackend

V = 32
MODEL = "target-fake"
DRAFT = "draft-fake"
MESSAGES = [{"role": "user", "content": "hello there"}]
CTX = GenerationContext(run_id="x", workload_tag="chat", prompt_id="x", rep=0)


class _Snap:
    mem_pressure = 0
    page_ins = 0
    thermal_level = 0


def _expected_text(max_tokens: int) -> str:
    tok = FakeTokenizer()
    prompt = tok.encode_chat(MESSAGES)
    result = generate_plain(FakeBackend(V), prompt, max_tokens, tok.eos_token_ids, CTX, _Snap)
    return tok.decode(result.tokens)


def _session(target=None, k: int = 3) -> Session:
    return Session(
        target or FakeBackend(V),
        FakeBackend(V, step=3),
        FakeTokenizer(),
        choose_k=lambda: k,
        model=MODEL,
        draft_name=DRAFT,
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(_session(), heartbeat_s=0.01))


def _sse_payloads(body: str) -> list[str]:
    return [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]


def test_non_stream_completion_returns_the_greedy_text(client: TestClient) -> None:
    r = client.post("/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 12})

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == MODEL
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": _expected_text(12)}
    assert choice["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 12
    assert body["usage"]["prompt_tokens"] == len(FakeTokenizer().encode_chat(MESSAGES))
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 12


def test_stream_completion_sends_chunks_then_done_and_the_same_text(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 12, "stream": True}
    )

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    payloads = _sse_payloads(r.text)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert {c["object"] for c in chunks} == {"chat.completion.chunk"}
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == _expected_text(12)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_temperature_above_zero_is_a_400_until_sampling_lands(client: TestClient) -> None:
    r = client.post("/v1/chat/completions", json={"messages": MESSAGES, "temperature": 0.7})

    assert r.status_code == 400
    assert "sampling" in r.json()["error"]["message"].lower()


def test_temperature_zero_and_unknown_openai_fields_are_accepted(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={"messages": MESSAGES, "max_tokens": 4, "temperature": 0, "top_p": 0.9, "n": 1},
    )

    assert r.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {"messages": [{"role": "robot", "content": "x"}]},
        {"messages": [{"role": "user"}]},
        {"messages": MESSAGES, "max_tokens": 0},
        {"messages": MESSAGES, "max_tokens": "lots"},
    ],
)
def test_malformed_bodies_are_422(client: TestClient, body: dict) -> None:
    r = client.post("/v1/chat/completions", json=body)

    assert r.status_code == 422
    assert "detail" in r.json()


def test_max_completion_tokens_is_honoured_as_the_openai_alias(client: TestClient) -> None:
    r = client.post("/v1/chat/completions", json={"messages": MESSAGES, "max_completion_tokens": 5})

    assert r.json()["usage"]["completion_tokens"] == 5


def test_a_concurrent_request_is_refused_with_429() -> None:
    backend = HoldableBackend(V)
    session = _session(target=backend)
    client = TestClient(create_app(session))
    backend.gate.clear()
    first: list = []
    worker = threading.Thread(
        target=lambda: first.append(
            client.post("/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 6})
        ),
        daemon=True,
    )
    worker.start()
    assert backend.entered.wait(2.0)

    second = client.post("/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 6})

    assert second.status_code == 429
    assert second.json()["error"]["type"] == "server_busy"
    backend.gate.set()
    worker.join(2.0)
    assert first and first[0].status_code == 200
    assert client.get("/healthz").json()["busy"] is False


def test_models_lists_the_one_served_model(client: TestClient) -> None:
    r = client.get("/v1/models")

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == [MODEL]
    assert body["data"][0]["object"] == "model"


def test_stats_reflect_the_windows_of_a_completed_request(client: TestClient) -> None:
    client.post("/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 10})

    stats = client.get("/stats").json()

    assert stats["model"] == MODEL
    assert stats["draft"] == DRAFT
    assert stats["k_current"] == 3
    assert stats["windows_total"] == 9  # a bad draft commits one token per window
    assert stats["proposed_total"] == 27
    assert stats["accepted_total"] == 0
    assert stats["alpha_ewma"] == pytest.approx(0.0)
    assert stats["tok_s_recent"] > 0
    assert len(stats["last_windows"]) == 9
    assert set(stats["last_windows"][0]) == {
        "k_proposed",
        "n_accepted",
        "draft_ms",
        "verify_ms",
        "window_ms",
    }
    assert stats["busy"] is False


def test_stats_stream_sends_stats_first_then_heartbeats_when_idle(client: TestClient) -> None:
    r = client.get("/stats/stream", params={"limit": 3})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [e for e in r.text.split("\n\n") if e.strip()]
    assert len(events) == 3
    assert events[0].startswith("event: stats\n")
    assert all(e.startswith("event: heartbeat\n") for e in events[1:])
    for e in events:
        payload = json.loads(e.split("data: ", 1)[1])
        assert payload["model"] == MODEL


def test_stats_stream_rejects_a_non_positive_limit(client: TestClient) -> None:
    assert client.get("/stats/stream", params={"limit": 0}).status_code == 422


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")

    assert r.status_code == 200
    assert r.json() == {"status": "ok", "busy": False}
