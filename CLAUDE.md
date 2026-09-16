# AcceptRate — rules for every agent working in this repo

Read this whole file before touching code. It is the contract. If something here
conflicts with your instincts, this file wins.

## What this is

A **measurement framework** for speculative decoding on Apple Silicon that happens
to include an adaptive runtime. The runtime re-picks draft depth K per draft
window from a live acceptance-rate estimate. The framework answers *when*
speculation pays off on a 16 GB M4 — and a negative answer is a valid, publishable
result. Never quietly drop a configuration that looked bad.

Design brief (source of truth for decisions): https://claude.ai/artifact/7K2oAJoHdFcENv9FAtxqq3

## Hardware

- Dev + target: M4 Mac mini, 10-core, 16 GB unified, ~120 GB/s.
- Model pair: `mlx-community/Llama-3.1-8B-Instruct-4bit` (target) +
  `mlx-community/Llama-3.2-1B-Instruct-4bit` (draft). Same Llama-3 tokenizer.
- Budget $0. Everything runs locally. No hosted services, no Docker (no Metal
  passthrough on macOS — impossible, not merely unnecessary).

## Stack (decided — do not re-litigate)

Python 3.12 · uv · `mlx==0.32.2` · `mlx-lm==0.31.3` (exact pins — a version bump
can silently invalidate the dataset) · Polars → Parquet · DuckDB · Pydantic v2 ·
Typer · FastAPI + uvicorn · scikit-learn `HistGradientBoostingRegressor` ·
pytest + hypothesis · ruff · matplotlib.

TUI: **Go + Bubble Tea v2**, a separate client process talking to
`acceptrate serve`. The brief's tab-4 tree mentions `tui/app.py  # Textual`;
that is superseded by tab 5. **No Textual.**

Dashboard: Svelte 5 + Vite + uPlot + Tailwind v4, built to static assets
embedded in the wheel and served by FastAPI. Users never need Node.

Design direction "Instrument". One `design/tokens.json` codegen'd to both
Lip Gloss constants and CSS custom properties.

**Not used:** PyTorch, Rust extensions, Weights & Biases, MLflow, Pandas,
Postgres, any React build chain, Docker, Textual, the TinyFish MCP.

## Five traps that silently ruin this

1. **MLX IS LAZY.** Timing a step without forcing evaluation measures queueing,
   not compute. Every timing boundary needs an explicit `mx.eval()` /
   `mx.synchronize()` *before* the clock stops. `backend/mlx_backend.py` is the
   only place timing boundaries touch MLX, and a test asserts the pattern.
2. **CHARM MODULE PATHS CHANGED.** `charm.land/bubbletea/v2` (v2.0.9),
   `charm.land/lipgloss/v2`, `charm.land/bubbles/v2` — NOT
   `github.com/charmbracelet/*`. Lip Gloss v2 needs manual light/dark
   detection. Verify every Charm API against current docs before writing Go.
3. **No Go toolchain existed on this machine.** It is installed via Homebrew in
   P0. Check `go version` before assuming.
4. **THE HOT LOOP ALLOCATES NOTHING.** Buffer trace events as tuples; flush per
   generation, never per window. No logging, no f-strings, no dict building
   inside the draft/verify cycle. System guards run on a separate 1 Hz thread
   that sets an int flag; the loop reads the int and never shells out.
5. **THE TUI NEVER RUNS DURING `bench`.** `chat` gets the live view; `bench`
   prints one progress line and writes parquet. This is architecture (separate
   process), not discipline.

## Two boundaries that carry the repo (tests enforce both)

- **HORIZONTAL:** `acceptrate/backend/protocol.py` (a `typing.Protocol` with
  `prefill` / `decode_step` / `verify` / `trim`) is the only thing `runtime/`
  knows about. `acceptrate/backend/mlx_backend.py` is the **single**
  Apple-specific file. Nothing above the seam imports `mlx` or `mlx_lm`.
- **VERTICAL:** `bench/` never imports `runtime/` or `model/`; `runtime/` and
  `model/` never import `bench/`. They meet only at the trace schema
  (`acceptrate/trace/schema.py`). You must not be able to benchmark a runtime
  that knows it is being benchmarked.

`tests/test_boundaries.py` scans the AST of `src/` and fails on any violation.
It runs in CI. Do not weaken it.

**How bench drives the engine without importing it:** `cli.py` is the
composition root. It imports both sides and hands `bench/` a plain callable
(`Generator`) that runs one generation and returns trace rows. `bench/` sees
only that callable and the schema. Dependency inversion, not a loophole.

## The trace schema

**One row per draft window, never per request.** Per-request averages destroy
the signal (α is high mid-code-block, low at a sentence boundary) and it cannot
be recovered later. Fields (brief tab 4):

| field | type | why |
|---|---|---|
| run_id | str | groups a sweep cell; carries the full config hash |
| window_idx | int | which draft/verify round within the generation |
| token_pos | int | position in the output — α drifts with depth |
| k_proposed | int | draft depth actually used this window |
| n_accepted | int | the numerator of everything |
| draft_ms | f32 | measured, not assumed |
| verify_ms | f32 | measured, not assumed |
| workload_tag | str | code · json · prose · chat · summarize · reason |
| mem_pressure | u8 | 0 normal / 1 warn / 2 critical; >0 excluded from headlines |
| page_ins | int | delta over the window; non-zero invalidates the sample |
| thermal_level | u8 | from pmset; throttled run = discarded run |
| prompt_id | str | (added) which corpus prompt — needed to tell generations apart |
| rep | int | (added) repeat index of the prompt within the run |
| window_ms | f32 | (added) wall clock of the whole window incl. Python overhead; speedup is measured from this, not draft_ms + verify_ms |

Rows are tuples in `trace/schema.py` `FIELD_NAMES` order. The definitive
schema is `acceptrate.trace.schema.SCHEMA`.

## Why the engine owns the loop

mlx-lm's built-in `draft_model=` path yields tokens with a `from_draft` flag but
exposes no per-window `draft_ms` / `verify_ms` and no way to change K
mid-generation. So `runtime/engine.py` drives `Backend` directly. mlx-lm's own
speculative path is an *external reference* only (P2 cross-check, one column in
P3). Do not "simplify" by calling `mlx_lm.generate(draft_model=...)` in the
runtime.

## Measurement rules (brief tab 6)

- Interleave A/B, never all-baseline-then-all-speculative (thermal drift).
- Discard warmup windows.
- Abort on page-ins or elevated memory pressure; do not quietly report.
- Report median and IQR, never the mean.
- Re-run the grid twice on different days and publish both.
- Losslessness: greedy equivalence is exact token match. The one platform
  caveat is measured, not assumed: batched-verify and sequential-decode Metal
  kernels disagree by up to ~0.10 in fp16 logits, so a divergence at a top-2
  margin within `calibration/noise_floor.json` is a near-tie, reported, not a
  failure (docs/gates/P2.md). Never widen that floor by hand; re-run
  `acceptrate verify calibrate` if the hardware or mlx version changes.

## Phases and gates (do not advance without evidence; two failures → stop and report)

| phase | gate |
|---|---|
| P0 env + skeleton | `acceptrate bench --smoke` streams 128 tokens from both models, exits 0 |
| P1 measurement rig | same config, two invocations: median tok/s within 2% |
| P2 losslessness | 20 prompts at T=0: token-identical to baseline, where a divergence counts only as a near-tie if its sequential top-2 margin is within the calibrated noise floor (`calibration/noise_floor.json`); every near-tie printed; 0 divergent |
| P3 the sweep | ≥ 50 000 clean draft windows in parquet, 0 rows with page_ins > 0 |
| P4 analytical fit | predicted vs measured speedup R² > 0.85, residuals vs mem_pressure plotted |
| P5 adaptive runtime | adaptive K beats best fixed K by ≥ 5% on held-out mixed workload |
| P6 predictor + TUI | predictor beats constant prior; TUI ≥ 30 fps while generating |
| P7 report | `make reproduce` regenerates every figure from raw traces on clean checkout |

CI runs correctness gates only (`pytest -m "not model"` + ruff). Performance
numbers never come from CI. Tests needing model weights carry `@pytest.mark.model`.

## Commit discipline (the owner cares about this — a lot)

- **Commit as often as possible. One logical change per commit.** If the
  message needs "and", split it.
- TDD rhythm = commit rhythm: `test:` (red) → `feat:`/`fix:` (green) →
  `refactor:`. Three commits per cycle is normal.
- Also commit separately: each new file skeleton, config change, dependency,
  doc update, schema change, CI change.
- Format: `<type>: <description>` with type ∈ feat, fix, refactor, docs, test,
  chore, perf, ci. Say what changed and why, specifically.
- Never let more than one small coherent change sit uncommitted.
- Push at every phase gate and at meaningful milestones.

### Parallel subagents and git (required)

- Parallel subagents **never** commit into the main working tree
  (`.git/index.lock` collisions). Each gets its own `git worktree` on its own
  branch (e.g. `feat/p1-guards`) under `../acceptrate-wt/<branch>`.
- Commit freely and often on the branch. Red commits are fine on branches.
- Merge back: rebase onto `main`, then fast-forward. **Never squash.** Never
  merge-commit. Test suite must be green at every merge point.
- The orchestrator's own sequential work commits directly on `main`.
- Do not touch `main` from a worktree; do not touch a worktree branch from
  `main`.

## Tooling

- Use `uv run` for everything Python. Never `pip install`.
- `make test` / `make lint` / `make smoke` / `make reproduce`.
- Web access: WebFetch + WebSearch. **Not** the TinyFish MCP.
- Branch is `main`, not `master`.

## Style

- Immutable data, small functions (< 50 lines), small files (< 400 lines
  typical), early returns, named constants, explicit error handling, Pydantic
  validation at every boundary.
- No `print` outside CLI entry points. No logging inside the hot loop.
