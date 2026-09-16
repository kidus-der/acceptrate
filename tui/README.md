# `acceptrate chat` — the Go + Bubble Tea v2 client

A separate process that talks HTTP to `acceptrate serve`. It never loads a
model and never runs during `bench` (CLAUDE.md trap 5).

```sh
cd tui
go run ./cmd/acceptrate-tui --mock                   # demo, no model, no server
go run ./cmd/acceptrate-tui --url http://127.0.0.1:8321 --baseline 16.6
go run ./cmd/acceptrate-tui --benchmark-frames 300   # prints achieved fps
go run ./cmd/acceptrate-tui --stats-only             # panel without the chat
```

| flag | default | meaning |
|---|---|---|
| `--url` | `http://127.0.0.1:8321` | `acceptrate serve` base URL |
| `--fps` | 30 | animation frame rate (1–120) |
| `--no-color` | off | ASCII colour profile; `NO_COLOR` is honoured too |
| `--stats-only` | off | instrument panel only, no prompt |
| `--benchmark-frames N` | 0 | render N synthetic busy frames to a discarded writer, print fps, exit |
| `--mock` | off | start the in-process mock (`internal/mock`) and connect to it |
| `--baseline TOK_S` | 0 | plain-decode tok/s from `bench`; enables the ghost marker and speedup. Omitted → nothing is faked |

Keys: type and `enter` to send · `pgup`/`pgdown`/`↑`/`↓` scroll the
transcript · `ctrl+c` / `esc` quit.

## Layout (brief tab 5, "Instrument")

```
acceptrate chat — Llama-3.1-8B-Instruct-4bit ← draft Llama-3.2-1B-Instruct-4bit
draft window #31 · generating
●●●●◆●●○●●●●●●◆●●●●○●●●●●●◆●●●●●○
acceptance  ████████████████░░░░░░░░ 0.71
throughput  ███████████████┆█████░░░ 28.4 tok/s  1.71× vs baseline
draft depth ██████████████████░░░░░░ K = 6  ↑ adapting from 4
windows 31 · accepted 118/160 · last window 5/6 · draft 1.2 ms · verify 9.8 ms
────────────────────────────────────────────────────────────────────────────
<transcript viewport, soft-wrapped, code fences styled>
> ask the model something…
```

The strip expands `last_windows` into tokens: `●` accepted, `○` the
rejection point, `◆` the bonus token after a fully accepted window — glyphs
from `theme`, never colour alone. The footer shows only what `/stats`
provides; memory, thermal and page-ins are not in the feed, so they are
omitted rather than invented.

## Draft-depth spring

`design/tokens.json` gives a Framer-Motion-style spring (stiffness k = 170,
damping c = 26, unit mass). Harmonica wants angular frequency and damping
ratio (`internal/ui/spring.go`):

```
ω = sqrt(k / m)        = sqrt(170)      ≈ 13.04 rad/s
ζ = c / (2 sqrt(k m))  = 26 / 26.08     ≈ 0.997
```

so `harmonica.NewSpring(harmonica.FPS(fps), ω, ζ)` — a hair under critical
damping: sub-second settle, barely visible overshoot. A tick command runs
only while the spring is moving; idle frames cost nothing.

Reduced motion: the tokens set `reduced_motion: true`, meaning *honour the
user's preference*; `ACCEPTRATE_REDUCED_MOTION=1` then snaps K instead of
springing.

## Tests

```sh
cd tui && go vet ./... && gofmt -l . && go test ./...
```

Everything runs against `internal/mock`, an `httptest` server replaying the
demo scenario (alpha 0.45 → 0.80, K 4 → 6, prose flowing into a code block)
with the exact frame shapes of `serve/sse.py`. The fps gate is asserted in
`internal/ui/bench_test.go` (300 frames < 10 s); on the M4 it renders ~1700
model+view frames per second at 120×40.

## Demo recording

`vhs tui/demo/chat.tape` (VHS is not installed here; the tape is checked in).
