"""`acceptrate serve`: an OpenAI-compatible local endpoint plus a live stats feed.

This package is the IPC boundary the Go TUI consumes and the adoption path for
Open WebUI / Zed / continue.dev (one base-URL change). It sits above the
runtime seam: it talks to `Backend` / `Tokenizer` via `runtime/streaming.py`
and never imports mlx.
"""
