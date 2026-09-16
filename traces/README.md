# The trace dataset

One row per draft window, never per request. Schema:
`acceptrate.trace.schema.SCHEMA` (14 fields, documented in CLAUDE.md). Each
run directory holds `manifest.json` (config hash → run_id, environment,
git sha) and `part-*.parquet`.

| directory | what | rows |
|---|---|---|
| `sweep/` | P3 grid: 3 drafts × K 0…8 × 6 tags × 36 train prompts × 200 tokens, two sessions for the 1B-4bit draft | 90,654 (70,290 clean) |
| `p5/` | held-out mixed workload: adaptive vs fixed K 1…8, two attempts | 2 × ~21k |
| `0c472f962077-*` | P1 repeatability runs (plain baseline, twice) | 2 × 2,143 |

Dirty rows (page_ins > 0, mem_pressure > 0 or thermal_level > 0) are kept
with their guard fields; every published number filters them with
`acceptrate.analysis.load.clean`. Machine: M4 Mac mini 16 GB, mlx 0.32.2,
mlx-lm 0.31.3, 2026-09-16. Load everything with
`acceptrate.analysis.load.load_sweep("traces/sweep")`.
