# Sweep session 1 (2026-09-16 12:07–12:47Z) — draft Llama-3.2-1B-Instruct-4bit, K 0–8

First full-grid session, kept in the dataset for its clean rows; its second
half is excluded by the memory-pressure guard (see "what went wrong").

## Headline shape (clean rows only, 9,586 windows, per-cell medians)

| tag | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | α (K=2) |
|---|---|---|---|---|---|---|---|---|---|
| code | 1.59 | **1.93** | 1.82 | 1.85 | 1.66 | 1.48 | 1.50 | 1.34 | 0.88 |
| json | 1.56 | **1.92** | 1.91 | 1.87 | 1.68 | 1.52 | 1.58 | 1.47 | 0.92 |
| reason | 1.54 | **1.84** | 1.76 | 1.63 | 1.46 | 1.34 | 1.39 | 1.19 | 0.88 |
| summarize | 1.45 | 1.63 | 1.39 | **1.72** | 1.27 | 1.20 | 1.06 | 1.02 | 0.78 |
| prose | 1.45 | **1.61** | 1.44 | 1.25 | 1.07 | 0.88 | 0.84 | 0.74 | 0.72 |
| chat | 1.44 | **1.56** | 1.46 | 1.26 | 1.07 | 0.94 | 0.91 | 0.72 | 0.74 |

Measured speedup vs the interleaved K=0 baseline (22.4–22.6 tok/s). Bold =
best K per tag. **Losing cells exist and are published:** prose and chat
lose from K=6 up; K=8 loses on both by ~26%.

Measured draft cost ratio c falls with K (0.19 at K=1 → 0.06 at K=8):
per-drafted-token cost drops because the draft's fixed per-window overhead
(the `pending` forward) is amortised. The closed form treats c as constant;
P4 will fit against the measured per-cell c.

Arm-level medians across all 36 prompts (dirty rows included):
plain 22.49 · K1 33.26 · K2 38.10 · K3 35.22 · K4 32.65 · K5 28.09 ·
K6 25.32 · K7 24.78 · K8 22.32 tok/s.

## What went wrong, and the fix

23,145 windows written; 13,559 (59%) dirty — 2,021 by page-ins (the usual
background 5–9%), **12,653 by `mem_pressure == 1`**, zero thermal. The
kernel flipped to "warn" around prompt 20 and stayed there: MLX's default
buffer-cache limit is all of physical memory, so its Metal pool grew for 40
minutes until free pages ran out (~150 MB free, 8.1 GB wired observed
live). Throughput was unaffected (plain IQR 22.31–22.58 across all 36
prompts) — it was the kernel's free-page signal, not paging — but the rule
is the rule: rows above pressure 0 are excluded, never "corrected".

Fix (commits 52b88d5, 938dfa3, b39a6aa): `MLXBackend.load` caps the cache
at 512 MB; the runner gained a `before_job` hook; sweep/adaptive runs release
cached buffers and wait for a clean guard state (max 120 s) before every
generation, and report `pacing_waits`. Session 2 relaunched 12:50Z with all
three drafts.
