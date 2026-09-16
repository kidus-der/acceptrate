"""`run(session)`: serve the app with uvicorn, programmatically.

Localhost by default — there is no auth, so binding wider is an explicit
choice the caller makes. Never `reload`: it would re-import the process and
reload gigabytes of weights.
"""

from __future__ import annotations

import uvicorn

from acceptrate.serve.app import create_app
from acceptrate.serve.session import Session

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8321


def run(
    session: Session,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_level: str = "info",
) -> None:
    """Block serving `session` until interrupted."""
    uvicorn.run(create_app(session), host=host, port=port, log_level=log_level)
