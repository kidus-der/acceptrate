# mlx-lm as an external reference (P2 cross-check)

CLAUDE.md: mlx-lm's own generation paths are an *external reference only*.
This page records two uses of them. **Everything here is a reference number,
not a measurement**: single run, no memory/thermal/page-in guards, no
warmup discard, no interleaving, tok/s exactly as mlx-lm reports it. The
framework's measured numbers come from `acceptrate bench`, never from here.

Environment: M4 Mac mini 16 GB, mlx 0.32.2, mlx-lm 0.31.3, target
`mlx-community/Llama-3.1-8B-Instruct-4bit`, draft
`mlx-community/Llama-3.2-1B-Instruct-4bit` (both bfloat16 4-bit, group 64).
Prompts: the first corpus prompt of each tag, chat template applied
(`backend.tokenizer.encode_chat(as_chat_messages(prompt))`), 64 tokens,
temperature 0 (`make_sampler(temp=0.0)`, i.e. argmax).

Code: `tests/reference_mlx_lm.py` (the only file outside
`backend/mlx_backend.py` that imports `mlx_lm`; it lives under `tests/` so the
horizontal boundary scan of `src/` stays intact) and
`src/acceptrate/verify/reference.py` (framework-free comparison).

## 1. Our greedy path vs mlx-lm's greedy path

Claim: `MLXBackend` + `runtime.engine.generate_plain` produce exactly the
tokens `mlx_lm.stream_generate(..., sampler=make_sampler(temp=0.0))` produces
for the same prompt token ids, with the same stop rule (eos token included
when produced; hard cap at `max_tokens`).

```
$ uv run pytest -m model tests/test_reference.py -q
............                                                             [100%]
```

| model | prompt | tokens | result |
|---|---|---|---|
| 1B | code-001 | 64 | identical |
| 1B | json-001 | 64 | identical |
| 1B | prose-001 | 64 | identical |
| 1B | chat-001 | 64 | identical |
| 1B | summarize-001 | 64 | identical |
| 1B | reason-001 | 64 | identical |
| 8B | code-001 | 64 | identical |
| 8B | prose-001 | 64 | identical |
| 8B | chat-001 | 64 | identical |

Result: 9/9 prompts token-identical at every index; no divergence to
report a margin for.

Two places where mlx-lm's path differs from ours and *could* flip a
near-tie, checked in `mlx_lm/generate.py` (0.31.3) and found not to matter
on these prompts:

- **Prefill chunking.** `generate_step` runs the first `L-1` prompt tokens
  through the model, then the last prompt token on its own inside `_step`.
  We run all `L` tokens in one pass. Different batch shapes take different
  Metal kernel paths (see `tests/test_mlx_backend.py`, `FP16_ATOL`).
- **Argmax dtype.** mlx-lm computes `logits - logsumexp(logits)` in the
  model dtype (bfloat16 here, ~3 significant digits) and argmaxes that. We
  upcast the raw logits to float32 and argmax. The shift preserves order but
  bf16 rounding can merge two close logits into a tie, which first-index
  argmax then resolves by position.

Neither is a correctness difference in the losslessness sense; both are
platform numerics. If a future prompt diverges, `describe(report)` names the
index and both tokens, and the top-2 logit margin at that index is the
thing to look at before anything else.

Also confirmed: no logits processors, repetition penalty, or KV-cache
quantisation are applied by default in `stream_generate`; the only
difference in stop handling is that mlx-lm never yields the eos token
in-loop but does carry it in its final `finish_reason="stop"` response, so
collecting `.token` over every response equals our token list.

## 2. mlx-lm's built-in speculative path (K = 4, 8B target + 1B draft)

External data point for the claim that greedy speculation is lossless: mlx-lm's
own `draft_model=` path should be token-identical to its plain greedy path.

Losslessness, from the same test run above (3 prompts, 64 tokens, K = 4):
`code-001`, `prose-001`, `chat-001` — all three identical to plain greedy.

Draft fraction and mlx-lm-reported tok/s come from:

```
$ uv run python -m tests.reference_mlx_lm
```

Single run, 2026-09-16, machine otherwise idle, no guards:

| prompt | tokens | finish | from draft | draft frac | spec tok/s | plain tok/s | vs plain |
|---|---|---|---|---|---|---|---|
| code-001 | 64 | length | 51 | 0.797 | 52.99 | 23.95 | identical |
| prose-001 | 64 | length | 40 | 0.625 | 27.99 | 23.95 | identical |
| chat-001 | 64 | length | 43 | 0.672 | 34.26 | 23.89 | identical |

Reading, with the usual caveat that this is one unguarded run of 64 tokens
per prompt: mlx-lm's own path shows the workload dependence the brief
predicts — code drafts well (0.80 of tokens from the draft, ~2.2× its own
plain rate) while prose barely clears break-even (0.63, ~1.2×). The plain
rate mlx-lm reports here (~24 tok/s) is above the ~20.5 tok/s median from the
P1 rig because it excludes prefill, has no guard thread, and times from the
first generated token. Compare speedups within this table only; the framework's
own speedup numbers come from `window_ms` in the traces.

Columns: `from draft` is the count of tokens with `GenerationResponse.from_draft`
set; `draft frac` divides that by the tokens generated; `spec tok/s` and
`plain tok/s` are `generation_tps` from the final response of the speculative
and plain `stream_generate` calls respectively; `vs plain` is the
`verify.reference.describe` verdict.
