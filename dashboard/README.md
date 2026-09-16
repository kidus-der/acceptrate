# AcceptRate dashboard

The brief's speedup calculator made live: `speedup(α, K, c)` plotted against
K where α and c are *measured* (from `GET /stats/stream`) and the current K is
marked on the curve. Svelte 5 (runes) + Vite + uPlot + Tailwind v4, built to
static assets that live inside the Python package and are served by
`acceptrate serve` at `/dashboard/`. **Users never need Node** — the built
output is committed and shipped in the wheel.

## Build

```sh
cd dashboard
npm install          # once
npm run build        # -> ../src/acceptrate/serve/static/  (commit the result)
npm test             # vitest, jsdom, no browser
```

From the repo root, `make dashboard` runs the build. After changing anything
under `dashboard/src`, rebuild and commit the regenerated
`src/acceptrate/serve/static/` alongside the source change; `pytest` checks the
build exists and FastAPI serves it.

`npm run dev` starts Vite's dev server; the SSE client resolves
`../stats/stream` relative to the page, so run it behind `acceptrate serve` or
open `http://localhost:5173/?demo`.

## Runtime

- `/dashboard/` (and `/` redirects there). No network beyond the local API:
  no webfonts, no CDNs. Fonts fall back from the token names to local
  monospace / system UI.
- `?demo` replays the scripted scenario from `tui/internal/mock/scenario.go`
  (α 0.45 → 0.80, K 4 → 6) with no server. The header's *demo* button does
  the same at any time; it is offered automatically when the feed is
  unreachable.
- `?theme=dark|light` pins the colour scheme; otherwise `prefers-color-scheme`
  decides. `prefers-reduced-motion` disables the busy pulse.

## Layout of `src/`

| file | role |
|---|---|
| `tokens.css` | **generated** from `design/tokens.json` — never edit; `uv run python -m acceptrate.design --write` |
| `app.css` | Tailwind import, token font stacks, panel primitives |
| `lib/speedup.ts` | port of `acceptrate/model/speedup.py`; tests pin the same values as `tests/test_speedup.py` |
| `lib/stats.ts` | `parseStats` (strict validation of the `/stats` payload) and `costRatio` (median of `(draft_ms / K) / verify_ms`) |
| `lib/client.ts` | EventSource client for `/stats/stream`: `stats` / `heartbeat` events, capped exponential backoff |
| `lib/history.ts` | immutable ring of (window #, tok/s, α) for the time-series |
| `lib/strip.ts` | window strip cells: ● accepted, ○ rejection point, ◆ bonus (mirrors `tui/internal/ui/strip.go`) |
| `lib/demo.ts` | the scripted scenario |
| `lib/charts.ts` | pure uPlot option/data builders |
| `lib/store.svelte.ts` | the one `$state` store, fed by the client or the demo |
| `lib/theme.svelte.ts` | colour-scheme tracking and reading the token palette for canvas |
| `lib/components/` | `Header`, `Calculator` (+ `Readout`), `Timeseries`, `WindowStrip`, `Explainer`, `Chart` |

Tokens are consumed as plain `var(--color-…)` custom properties in component
styles; Tailwind is used for layout and spacing only, so the palette stays
single-sourced in `design/tokens.json`. Accept / reject / bonus are never
encoded by colour alone — the glyph tokens carry the meaning too.
