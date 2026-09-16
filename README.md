# AcceptRate

Adaptive speculative decoding for Apple Silicon — and the measurement
framework that proves *when* it pays off.

mlx-lm already ships speculative decoding. Published Mac results range from
2.43× faster to a loss in 24 of 24 configurations, because nobody wrote down
the boundary. This repo measures that boundary honestly on a 16 GB M4, then
ships a runtime that re-picks draft depth K per request from a live acceptance
estimate. A negative result is a valid result; negative cells are published
alongside positive ones.

## Status

| phase | gate | status |
|---|---|---|
| P0 env + skeleton | `acceptrate bench --smoke` streams 128 tokens, exits 0 | **pass** — [evidence](docs/gates/P0.md) |
| P1 measurement rig | same config twice, median tok/s within 2% | **pass** — 0.14%, [evidence](docs/gates/P1.md) |
| P2 losslessness | 20 prompts at T=0, token-identical, 20/20 | **pass** — 19 identical + 1 near-tie within the calibrated 0.10 noise floor, [evidence](docs/gates/P2.md) |
| P3 the sweep | ≥ 50 000 clean draft windows, 0 rows page_ins > 0 | **pass** — 70,290 clean windows, [evidence](docs/gates/P3.md) |
| P4 analytical fit | predicted vs measured R² > 0.85 | **pass** — R² 0.970 with the measured verify factor (brief's form: −12.5), [evidence](docs/gates/P4.md) |
| P5 adaptive runtime | adaptive K beats best fixed K by ≥ 5% | **fail ×2** — adaptive −1.1% vs fixed K=2; a per-prompt oracle gains 0.0%, [evidence](docs/gates/P5.md) |
| P6 predictor + TUI | predictor beats constant prior; TUI ≥ 30 fps | **pass** — MAE −19.1% vs prior; 1,816 fps, [evidence](docs/gates/P6.md) |
| P7 report | `make reproduce` regenerates every figure | |

## Layout

```
src/acceptrate/
  backend/   protocol.py is the seam; mlx_backend.py is the only Apple-specific file
  runtime/   engine + scheduler; talks to Backend only
  model/     closed-form speedup, EWMA estimator, cold-start predictor
  bench/     measurement harness; never imports the runtime
  trace/     the one-row-per-draft-window schema both sides share
  verify/    losslessness tests
  analysis/  figures and tables
tui/         Go + Bubble Tea v2 client (P6)
```

See [CLAUDE.md](CLAUDE.md) for the rules every contributor and agent follows.
