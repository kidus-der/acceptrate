"""serve/runner.py — programmatic uvicorn, bound to localhost by default, never with reload."""

from __future__ import annotations

from fastapi import FastAPI

from acceptrate.serve import runner
from acceptrate.serve.session import Session
from tests.fakes import FakeBackend, FakeTokenizer


def _session() -> Session:
    return Session(FakeBackend(), None, FakeTokenizer(), choose_k=lambda: 0, model="m")


def test_run_binds_localhost_and_the_default_port(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(runner.uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))

    runner.run(_session())

    assert isinstance(captured["app"], FastAPI)
    assert captured["host"] == runner.DEFAULT_HOST == "127.0.0.1"
    assert captured["port"] == runner.DEFAULT_PORT == 8321
    assert captured.get("reload", False) is False


def test_run_passes_host_port_and_log_level_through(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(runner.uvicorn, "run", lambda app, **kw: captured.update(kw))

    runner.run(_session(), host="0.0.0.0", port=9000, log_level="warning")

    assert (captured["host"], captured["port"], captured["log_level"]) == (
        "0.0.0.0",
        9000,
        "warning",
    )
