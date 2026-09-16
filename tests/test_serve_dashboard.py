"""serve/app.py — the dashboard's static mount, on fakes only.

The built Svelte app lives in src/acceptrate/serve/static/ (committed, shipped
in the wheel). When it is missing the mount degrades to a clear 404 JSON body
rather than a bare Starlette error, so `acceptrate serve` still works without Node.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from acceptrate.serve.app import DASHBOARD_MISSING, STATIC_DIR, create_app
from acceptrate.serve.session import Session
from tests.fakes import FakeBackend, FakeTokenizer

V = 32


def _session() -> Session:
    return Session(
        FakeBackend(V),
        FakeBackend(V, step=3),
        FakeTokenizer(),
        choose_k=lambda: 3,
        model="target-fake",
        draft_name="draft-fake",
    )


def _client(static_dir: Path) -> TestClient:
    return TestClient(create_app(_session(), static_dir=static_dir))


def _built(tmp_path: Path) -> Path:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>dash</title>")
    (static / "assets" / "app.js").write_text("console.log('hi')")
    return static


def test_the_committed_build_exists_with_an_index_and_assets() -> None:
    assert (STATIC_DIR / "index.html").is_file()
    assert any((STATIC_DIR / "assets").glob("*.js"))


def test_dashboard_serves_index_html_when_the_build_exists(tmp_path: Path) -> None:
    client = _client(_built(tmp_path))

    r = client.get("/dashboard/")

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "dash" in r.text


def test_dashboard_serves_assets_relative_to_the_mount(tmp_path: Path) -> None:
    client = _client(_built(tmp_path))

    r = client.get("/dashboard/assets/app.js")

    assert r.status_code == 200
    assert "hi" in r.text


def test_root_redirects_to_the_dashboard(tmp_path: Path) -> None:
    client = _client(_built(tmp_path))

    r = client.get("/", follow_redirects=False)

    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/dashboard/"


def test_dashboard_is_a_clear_404_json_when_the_build_is_missing(tmp_path: Path) -> None:
    client = _client(tmp_path / "nowhere")

    for path in ("/dashboard/", "/dashboard/assets/app.js"):
        r = client.get(path)
        assert r.status_code == 404, path
        body = r.json()
        assert body["error"]["type"] == "not_found"
        assert body["error"]["message"] == DASHBOARD_MISSING
        assert "npm run build" in DASHBOARD_MISSING


def test_the_api_still_answers_when_the_build_is_missing(tmp_path: Path) -> None:
    client = _client(tmp_path / "nowhere")

    assert client.get("/healthz").status_code == 200
    assert client.get("/stats").status_code == 200
