<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg?v=1">
  <img alt="acceptrate — adaptive speculative decoding for Apple Silicon, measured honestly" src="./assets/hero-light.svg?v=1" width="100%">
</picture>

<p>
  <a href="https://github.com/kidus-der/acceptrate/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/kidus-der/acceptrate/ci.yml?branch=main&style=flat-square&label=ci&labelColor=1f2328&color=52B9C8" alt="ci"></a>
  <img src="https://img.shields.io/badge/python-3.12-1f2328?style=flat-square&logo=python&logoColor=52B9C8" alt="python 3.12">
  <img src="https://img.shields.io/badge/mlx-0.32.2-1f2328?style=flat-square&logoColor=52B9C8" alt="mlx 0.32.2">
  <img src="https://img.shields.io/badge/go-1.27-1f2328?style=flat-square&logo=go&logoColor=52B9C8" alt="go 1.27">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-1f2328?style=flat-square&logoColor=52B9C8" alt="MIT"></a>
  <a href="./docs/report.md"><img src="https://img.shields.io/badge/read-the%20report-1f2328?style=flat-square&color=E3A44C" alt="the report"></a>
</p>

A small model guesses the next few words; the big model checks them all in one pass. Apple's mlx-lm already ships this trick. Nobody had measured **when it actually helps on a normal Mac** — published numbers ran from 2.4× faster to a loss in 24 of 24 configs. This repo is the measurement, the map, and a runtime that uses it.

### The one curve that explains everything

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/verify-cost-dark.svg?v=1">
  <img alt="Cost of one verify pass in plain decode steps: 1.00, 1.05, 1.30, 1.57, 1.98, 2.43, 2.48, 2.96 for K = 1 to 8" src="./assets/verify-cost-light.svg?v=1" width="100%">
</picture>

The textbook assumes checking K guesses costs the same as writing one word. On a base M4 it does — for two guesses. Then the cost climbs almost linearly, and every result below follows from it: the published closed form scores **R² = −12.5** against our grid; with this measured cost in its denominator it scores **R² = 0.97**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/curve-dark.svg?v=1">
  <img alt="Measured speedup against draft depth for six kinds of text; every line peaks at K=2, chat and prose lose past K=5" src="./assets/curve-light.svg?v=1" width="100%">
</picture>

| kind of text | best K | speedup | worst cell |
|---|---|---|---|
| code · json | 2 | **1.9×** | 0.92× · 1.03× |
| reasoning | 2 | 1.8× | 0.90× |
| summaries | 4 | 1.7× | 0.70× |
| prose · chat | 2 | 1.6× | **0.66× · 0.58×** |

Losing cells are published, not dropped. The full grid — 3 drafts × K 0…8 × 6 workloads, 90,654 windows with every guard reading — is in [`traces/`](./traces) and every figure regenerates from it with one command.

### Try it

```sh
git clone git@github.com:kidus-der/acceptrate.git && cd acceptrate
uv sync && make test                    # ~500 tests, no model needed
uv run acceptrate models pull           # 8B 4-bit target + 1B 4-bit draft, ~5 GB
uv run acceptrate serve --adaptive      # OpenAI-compatible endpoint + dashboard at http://127.0.0.1:8321
uv run acceptrate chat                  # the live instrument panel (Go), or: make chat-demo — no model
make reproduce                          # every figure and table, from the raw traces
```

Point Open WebUI, Zed or continue.dev at `http://127.0.0.1:8321/v1` and it just works.

### What's inside

```mermaid
flowchart LR
  subgraph measure ["bench/ — the rig"]
    G[guards · 1 Hz<br/>pressure · page-ins · heat] --> R[interleaved runner]
    R --> T[(one row per draft window)]
  end
  subgraph run ["runtime/ — the product"]
    E[engine · draft / verify] --> S[scheduler · picks K live]
    S --> V[serve · TUI · dashboard]
  end
  T -.-> A[analysis · fit · figures · report]
  E -.-> T
  B[backend/protocol.py<br/>the only seam] --> E
  M[mlx_backend.py<br/>the only Apple-specific file] --> B
  style measure fill:none,stroke:#E3A44C
  style run fill:none,stroke:#52B9C8
```

The measuring side and the running side never import each other; they meet only at the trace schema, so the benchmark cannot be flattered by the thing it measures. Every timing forces the GPU to finish first (MLX is lazy — [the trap, quantified](./tests/test_laziness.py)). Guards run on their own thread; the hot loop reads an int.

### Honest results, including the ones that hurt

Eight gates, each machine-checked, evidence in [`docs/gates/`](./docs/gates). Seven passed. The adaptive-K gate did not: the scheduler finds K=2 on its own and stays there, because on this chip K=2 is already optimal for every workload — a per-prompt oracle gains 0.0%. That is the finding, and it stays marked failed. Losslessness is exact up to the platform's own fp16 kernel noise, which is calibrated rather than assumed ([P2](./docs/gates/P2.md)).

The map is drawn from one machine on one day; the second-day rerun, a sharper sampling test and the polish on the terminal and browser views are still ahead. The numbers here will move a little — the shape will not.

<p>
  <a href="./docs/report.md">Report</a> · <a href="./docs/gates">Gate evidence</a> · <a href="./docs/findings">Findings</a> · <a href="./traces">Dataset</a> · <a href="./CLAUDE.md">Rules of the repo</a> · <a href="https://claude.ai/artifact/7K2oAJoHdFcENv9FAtxqq3">Design brief</a>
</p>
