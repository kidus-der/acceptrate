# When does speculative decoding pay off on a base M4? — AcceptRate report

*Draft; regenerate every figure and table with `make reproduce`. Sections
marked ⏳ are filled in as their phase gate passes.*

## The claim

On a 16 GB M4 Mac mini with an 8B 4-bit target and a 1B 4-bit draft,
speculative decoding pays off on every workload we measured **at K = 2**,
from 1.6× (chat, prose) to 1.9× (code, json) — and it loses, sometimes badly,
past K ≈ 4 on conversational text (0.58× at K = 8 for chat). The reason is
not acceptance: it is that on this hardware the target's verification pass
stops being free after two tokens. Verifying K+1 tokens costs 1.0, 1.05,
1.3, 1.6, 2.0, 2.4, 2.5, 3.0 plain decode steps for K = 1…8. The published
closed-form model assumes 1.0 and scores R² = −12.5 against our grid; with
the measured verify cost in the denominator it scores R² = 0.970.

## How it was measured

- **Rig** (P1, `docs/gates/P1.md`): interleaved arms with a rotating leader,
  two warmup generations discarded, one trace row per draft window, guards
  on a 1 Hz thread (page-ins, kernel memory pressure, thermal). Repeatability
  of the plain baseline across two invocations: 0.14%. Every timing boundary
  forces `mx.eval`; the laziness trap is demonstrated in
  `tests/test_laziness.py` (0.32 ms lazy vs 13.0 ms evaluated).
- **Losslessness** (P2, `docs/gates/P2.md`): greedy output token-identical
  to plain decoding on 19/20 prompts; the twentieth diverges at an exact
  fp16 tie (margin 0.0000) that batched and sequential Metal kernels resolve
  differently. The cross-kernel noise floor is calibrated (max 0.10 over 400
  positions) and a divergence within it is reported as a near-tie. Our plain
  greedy matches mlx-lm's own on 9/9 prompts.
- **Sweep** (P3, `docs/gates/P3.md`): 3 drafts × K 0…8 × 6 workload tags ×
  36 prompts × 200 tokens; 70,290 clean windows kept, 20,364 excluded with
  their guard fields intact (never deleted). The first session's second half
  was excluded by the memory-pressure guard: MLX's default buffer cache is
  the whole machine and grew until the kernel reported pressure
  (`docs/findings/sweep-session-1.md`). Capping it at 512 MB and pacing on
  guard state fixed it.
- **Model** (P4, `docs/gates/P4.md`): α = accepted / examined per cell,
  c = draft cost per token relative to the verify pass, v(K) = verify pass
  relative to a plain step — all measured, none fitted.

## The map

Per-draft heatmaps: `docs/figures/speedup_heatmap_*.png`. The per-cell
table with every losing cell: `docs/figures/cells.md`.

| tag | best K | best speedup | worst | losing cells / 32 | α range |
|---|---|---|---|---|---|
| code | 2 | 1.93× | 0.92× | 2 | 0.84–0.90 |
| json | 2 | 1.92× | 1.03× | 0 | 0.87–0.93 |
| reason | 2 | 1.84× | 0.90× | 3 | 0.84–0.91 |
| summarize | 4 | 1.72× | 0.70× | 10 | 0.76–0.85 |
| prose | 2 | 1.61× | 0.66× | 15 | 0.70–0.82 |
| chat | 2 | 1.60× | 0.58× | 14 | 0.73–0.80 |

Drafts: the 1B-4bit draft wins everywhere at K = 2; the 3B-4bit draft peaks
at 1.30× and loses from K = 5 (c = 0.46); the 1B-8bit draft is strictly
worse than the 4-bit one at every K (c = 0.33 vs 0.19, same acceptance).

## The memory ceiling

The 8B-4bit + 1B-4bit pair needs ~6.3 GB and leaves ~5 GB of the post-OS
budget; a 14B-4bit target with the same draft would leave 1.4 GB and is
refused by the startup guard. Even the safe pair needs MLX's buffer cache
capped: with the default (unbounded) cache, a 40-minute run drove the
kernel into memory-pressure "warn" and 55% of windows had to be excluded.

## The adaptive runtime ⏳ (P5)

The scheduler re-solves argmax_K of the corrected form every window from
an EWMA of per-token acceptance and a measured draft cost; K never drops
below 1 so the estimate keeps observing. Result on the held-out mixed
workload: *pending — `docs/gates/P5.md`.*

## The cold-start prior ⏳ (P6)

*pending — `docs/gates/P6.md`.*

## Reproduce

```
make reproduce      # P3 gate, P4 fit tables and every figure from traces/
```
Raw traces (`traces/sweep`, one parquet cell per arm per invocation with a
manifest) are the dataset; nothing in this report is computed anywhere else.
